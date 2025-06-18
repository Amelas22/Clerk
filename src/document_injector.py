"""
Main document injector module.
Orchestrates the entire document processing pipeline from Box to vector storage.
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_processing.box_client import BoxClient, BoxDocument
from src.document_processing.pdf_extractor import PDFExtractor
from src.document_processing.chunker import DocumentChunker
from src.document_processing.deduplicator import DocumentDeduplicator
from src.document_processing.context_generator import ContextGenerator
from src.vector_storage.embeddings import EmbeddingGenerator
from src.vector_storage.vector_store import VectorStore
from src.vector_storage.fulltext_search import FullTextSearchManager
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of processing a single document"""
    document_id: str
    file_name: str
    case_name: str
    status: str  # "success", "duplicate", "failed"
    chunks_created: int
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

class DocumentInjector:
    """Main orchestrator for document processing pipeline"""
    
    def __init__(self, enable_cost_tracking: bool = True):
        """Initialize all components
        
        Args:
            enable_cost_tracking: Whether to track API costs
        """
        logger.info("Initializing Document Injector")
        
        # Initialize components
        self.box_client = BoxClient()
        self.pdf_extractor = PDFExtractor()
        self.chunker = DocumentChunker()
        self.deduplicator = DocumentDeduplicator()
        self.context_generator = ContextGenerator()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.fulltext_search = FullTextSearchManager()
        
        # Cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        if enable_cost_tracking:
            self.cost_tracker = CostTracker()
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "duplicates": 0,
            "failed": 0
        }
    
    def process_case_folder(self, parent_folder_id: str, 
                          max_documents: Optional[int] = None) -> List[ProcessingResult]:
        """Process all documents in a case folder
        
        Args:
            parent_folder_id: Box folder ID for the case
            max_documents: Maximum number of documents to process (for testing)
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting processing for folder ID: {parent_folder_id}")
        results = []
        
        try:
            # Get all PDFs in folder and subfolders
            documents = list(self.box_client.traverse_folder(parent_folder_id))
            
            if max_documents:
                documents = documents[:max_documents]
            
            logger.info(f"Found {len(documents)} PDF documents to process")
            
            # Process each document
            for i, box_doc in enumerate(documents, 1):
                logger.info(f"Processing document {i}/{len(documents)}: {box_doc.name}")
                
                result = self._process_single_document(box_doc)
                results.append(result)
                
                # Update statistics
                self.stats["total_processed"] += 1
                if result.status == "success":
                    self.stats["successful"] += 1
                elif result.status == "duplicate":
                    self.stats["duplicates"] += 1
                else:
                    self.stats["failed"] += 1
                
                # Log progress
                if i % 10 == 0:
                    self._log_progress()
            
            self._log_final_statistics()
            
            # Generate and save cost report
            if self.enable_cost_tracking:
                logger.info("Generating cost report...")
                self.cost_tracker.print_summary()
                report_path = self.cost_tracker.save_report()
                logger.info(f"Cost report saved to: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error processing folder: {str(e)}")
            raise
    
    def _process_single_document(self, box_doc: BoxDocument) -> ProcessingResult:
        """Process a single document through the entire pipeline
        
        Args:
            box_doc: BoxDocument object
            
        Returns:
            ProcessingResult
        """
        start_time = datetime.utcnow()
        
        # Start cost tracking for this document
        if self.enable_cost_tracking:
            doc_cost = self.cost_tracker.start_document(
                box_doc.name, box_doc.file_id, box_doc.case_name
            )
        
        try:
            # Step 1: Download document
            logger.debug(f"Downloading {box_doc.name}")
            content = self.box_client.download_file(box_doc.file_id)
            
            # Step 2: Check for duplicates
            doc_hash = self.deduplicator.calculate_document_hash(content)
            exists, existing_record = self.deduplicator.check_document_exists(doc_hash)
            
            if exists:
                # Handle duplicate
                logger.info(f"Duplicate found: {box_doc.name}")
                self.deduplicator.add_duplicate_location(
                    doc_hash, box_doc.path, box_doc.case_name
                )
                
                return ProcessingResult(
                    document_id=existing_record["document_hash"],
                    file_name=box_doc.name,
                    case_name=box_doc.case_name,
                    status="duplicate",
                    chunks_created=0,
                    processing_time=(datetime.utcnow() - start_time).total_seconds()
                )
            
            # Step 3: Register new document
            doc_record = self.deduplicator.register_new_document(
                doc_hash, box_doc.name, box_doc.path, box_doc.case_name,
                metadata={
                    "file_size": box_doc.size,
                    "modified_at": box_doc.modified_at.isoformat(),
                    "folder_path": box_doc.folder_path
                }
            )
            
            # Step 4: Extract text
            extracted = self.pdf_extractor.extract_text(content, box_doc.name)
            
            if not self.pdf_extractor.validate_extraction(extracted):
                raise ValueError("Text extraction failed or produced invalid results")
            
            # Step 5: Chunk document
            doc_metadata = {
                "case_name": box_doc.case_name,  # CRITICAL for case isolation
                "document_name": box_doc.name,
                "document_path": box_doc.path,
                "document_hash": doc_hash,
                "page_count": extracted.page_count
            }
            
            chunks = self.chunker.chunk_document(extracted.text, doc_metadata)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 6: Generate contexts for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            chunks_with_context, context_usage = self.context_generator.generate_contexts_sync(
                chunk_texts, extracted.text
            )
            
            # Track context generation costs
            if self.enable_cost_tracking:
                self.cost_tracker.track_context_usage(
                    box_doc.file_id, 
                    context_usage["prompt_tokens"],
                    context_usage["completion_tokens"],
                    model=self.context_generator.model
                )
            
            # Step 7: Generate embeddings and prepare for storage
            storage_chunks = []
            
            for chunk, context_chunk in zip(chunks, chunks_with_context):
                # Prepare search text
                search_text = self.fulltext_search.prepare_text_for_search(
                    context_chunk.combined_content
                )
                
                # Generate embedding
                embedding, embedding_tokens = self.embedding_generator.generate_embedding(
                    context_chunk.combined_content
                )
                
                # Track embedding costs
                if self.enable_cost_tracking:
                    self.cost_tracker.track_embedding_usage(
                        box_doc.file_id, embedding_tokens,
                        model=self.embedding_generator.model
                    )
                
                # Prepare chunk data
                chunk_data = {
                    "content": context_chunk.combined_content,
                    "embedding": embedding,
                    "search_text": search_text,  # For full-text search
                    "metadata": {
                        **chunk.metadata,
                        "has_context": bool(context_chunk.context),
                        "original_length": len(chunk.content),
                        "context_length": len(context_chunk.context)
                    }
                }
                storage_chunks.append(chunk_data)
            
            # Step 8: Store in vector database
            chunk_ids = self.vector_store.store_document_chunks(
                box_doc.case_name,  # CRITICAL: Pass case name
                doc_hash,
                storage_chunks
            )
            
            # Step 9: Verify case isolation
            if not self.vector_store.verify_case_isolation(box_doc.case_name):
                logger.error(f"CRITICAL: Case isolation verification failed for {box_doc.case_name}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Successfully processed {box_doc.name} in {processing_time:.2f}s")
            
            # Finish cost tracking
            if self.enable_cost_tracking:
                self.cost_tracker.finish_document(
                    box_doc.file_id, len(chunk_ids), processing_time
                )
            
            return ProcessingResult(
                document_id=doc_hash,
                file_name=box_doc.name,
                case_name=box_doc.case_name,
                status="success",
                chunks_created=len(chunk_ids),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing {box_doc.name}: {str(e)}")
            
            # Finish cost tracking even on failure
            if self.enable_cost_tracking:
                self.cost_tracker.finish_document(
                    box_doc.file_id, 0,
                    (datetime.utcnow() - start_time).total_seconds()
                )
            
            return ProcessingResult(
                document_id="",
                file_name=box_doc.name,
                case_name=box_doc.case_name,
                status="failed",
                chunks_created=0,
                error_message=str(e),
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _log_progress(self):
        """Log current processing progress"""
        logger.info(
            f"Progress: {self.stats['total_processed']} processed "
            f"({self.stats['successful']} successful, "
            f"{self.stats['duplicates']} duplicates, "
            f"{self.stats['failed']} failed)"
        )
    
    def _log_final_statistics(self):
        """Log final processing statistics"""
        logger.info("=" * 50)
        logger.info("Processing Complete!")
        logger.info(f"Total documents processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Duplicates: {self.stats['duplicates']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        # Get deduplication stats
        dedup_stats = asyncio.run(self.deduplicator.get_statistics_async())
        logger.info(f"Total unique documents in system: {dedup_stats['total_unique_documents']}")
        logger.info(f"Total duplicate instances: {dedup_stats['total_duplicate_instances']}")
        logger.info("=" * 50)
    
    def process_multiple_cases(self, folder_ids: List[str]) -> Dict[str, List[ProcessingResult]]:
        """Process multiple case folders
        
        Args:
            folder_ids: List of Box folder IDs
            
        Returns:
            Dictionary mapping folder ID to processing results
        """
        all_results = {}
        
        for folder_id in folder_ids:
            logger.info(f"\nProcessing case folder: {folder_id}")
            results = self.process_case_folder(folder_id)
            all_results[folder_id] = results
        
        return all_results
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get the current cost report
        
        Returns:
            Cost report dictionary or empty dict if cost tracking disabled
        """
        if self.enable_cost_tracking:
            return self.cost_tracker.get_session_report()
        return {}


# Utility function for testing
def test_connection():
    """Test all connections and basic functionality"""
    logger.info("Testing Document Injector connections...")
    
    try:
        injector = DocumentInjector()
        
        # Test Box connection
        if injector.box_client.check_connection():
            logger.info("✓ Box connection successful")
        else:
            logger.error("✗ Box connection failed")
        
        # Test Supabase connection
        try:
            injector.deduplicator.get_statistics()
            logger.info("✓ Supabase connection successful")
        except:
            logger.error("✗ Supabase connection failed")
        
        # Test OpenAI connection
        try:
            test_embedding = injector.embedding_generator.generate_embedding("test")
            logger.info("✓ OpenAI connection successful")
        except:
            logger.error("✗ OpenAI connection failed")
        
        logger.info("Connection tests complete")
        
    except Exception as e:
        logger.error(f"Error during connection test: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run connection test
    test_connection()