"""
Main document injector module.
Orchestrates the entire document processing pipeline from Box to Qdrant.
All database operations now use Qdrant exclusively.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.document_processing.box_client import BoxClient, BoxDocument
from src.document_processing.pdf_extractor import PDFExtractor
from src.document_processing.chunker import DocumentChunker
from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator
from src.document_processing.context_generator import ContextGenerator
from src.vector_storage.embeddings import EmbeddingGenerator
from src.vector_storage.qdrant_store import QdrantVectorStore, SearchResult
from src.vector_storage.sparse_encoder import SparseVectorEncoder, LegalQueryAnalyzer
from config.settings import settings
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
    
    def __init__(self, enable_cost_tracking: bool = True, no_context: bool = False):
        """Initialize all components
        
        Args:
            enable_cost_tracking: Whether to track API costs
            no_context: Whether to skip context generation
        """
        logger.info("Initializing Document Injector with Qdrant backend")
        
        # Initialize components
        self.box_client = BoxClient()
        self.pdf_extractor = PDFExtractor()
        self.chunker = DocumentChunker()
        self.deduplicator = QdrantDocumentDeduplicator()
        self.context_generator = ContextGenerator()
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize Qdrant vector store
        self.vector_store = QdrantVectorStore()
        
        # Initialize sparse encoder for hybrid search
        self.sparse_encoder = SparseVectorEncoder()
        self.query_analyzer = LegalQueryAnalyzer(self.sparse_encoder)
        
        # Cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        if enable_cost_tracking:
            self.cost_tracker = CostTracker()
        
        # Conditional context generation
        self.no_context = no_context
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "duplicates": 0,
            "failed": 0
        }
    
    def process_case_folder(self, folder_id: str, 
                              max_documents: Optional[int] = None) -> List[ProcessingResult]:
        """Process all documents in a case folder (Box folder)
        
        Args:
            folder_id: Box folder ID to process
            max_documents: Maximum number of documents to process (for testing)
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting to process case folder: {folder_id}")
        
        # Get all PDFs from folder
        documents = list(self.box_client.traverse_folder(folder_id))
        
        if max_documents:
            documents = documents[:max_documents]
            logger.info(f"Limited to processing {max_documents} documents")
        
        logger.info(f"Found {len(documents)} documents to process")
        
        # Process each document
        results = []
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
        
        # Log summary
        self._log_processing_summary()
        
        return results

    def _process_single_document(self, box_doc: BoxDocument) -> ProcessingResult:
        """Process a single document through the entire pipeline
        
        Args:
            box_doc: Box document to process
            
        Returns:
            Processing result
        """
        start_time = datetime.utcnow()
        
        # Track costs if enabled
        if self.enable_cost_tracking:
            doc_cost = self.cost_tracker.start_document(
                box_doc.name, box_doc.file_id, box_doc.case_name
            )
        
        try:
            # Step 1: Download document
            logger.debug(f"Downloading {box_doc.name}")
            content = self.box_client.download_file(box_doc.file_id)
            logger.debug(f"Downloaded content type: {type(content).__name__}, length: {len(content)}")
            
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
                    "folder_path": box_doc.folder_path,
                    "subfolder_name": box_doc.subfolder_name  # Track subfolder
                }
            )
            
            # Step 4: Extract text
            extracted = self.pdf_extractor.extract_text(content, box_doc.name)
            
            if not self.pdf_extractor.validate_extraction(extracted):
                raise ValueError("Text extraction failed or produced invalid results")
            
            # Step 5: Get document link
            doc_link = self.box_client.get_shared_link(box_doc.file_id)
            
            # Step 6: Chunk document
            doc_metadata = {
                "case_name": box_doc.case_name,  # CRITICAL for case isolation
                "document_name": box_doc.name,
                "document_path": box_doc.path,
                "document_hash": doc_hash,
                "document_link": doc_link,  # Add Box link
                "page_count": extracted.page_count,
                "document_type": self._determine_document_type(box_doc.name, extracted.text),
                "folder_path": "/".join(box_doc.folder_path),
                "subfolder": box_doc.subfolder_name or "root",  # Track subfolder
                "file_size": box_doc.size,
                "modified_at": box_doc.modified_at.isoformat()
            }
            
            chunks = self.chunker.chunk_document(extracted.text, doc_metadata)
            logger.info(f"Created {len(chunks)} chunks from {box_doc.subfolder_name or 'root'}/{box_doc.name}")
            
            # Step 7: Conditionally generate contexts for chunks
            if self.no_context:
                # Skip context generation
                chunks_with_context = chunks
                context_usage = None
            else:
                chunk_texts = [chunk.content for chunk in chunks]
                chunks_with_context, context_usage = self.context_generator.generate_contexts_sync(
                    chunk_texts, extracted.text
                )
            
            # Track context generation costs
            if self.enable_cost_tracking and context_usage:
                self.cost_tracker.track_context_usage(
                    box_doc.file_id, 
                    context_usage["prompt_tokens"],
                    context_usage["completion_tokens"],
                    model=self.context_generator.model
                )
            
            # Step 8: Generate embeddings and prepare for storage
            storage_chunks = []
            
            for chunk, context_chunk in zip(chunks, chunks_with_context):
                # Prepare search text for full-text search
                search_text = self.sparse_encoder.prepare_search_text(
                    getattr(context_chunk, 'combined_content', context_chunk.original_chunk)
                )
                
                # Generate dense embedding
                embedding, embedding_tokens = self.embedding_generator.generate_embedding(
                    getattr(context_chunk, 'combined_content', context_chunk.original_chunk)
                )
                
                # Generate sparse vectors for hybrid search
                keyword_sparse, citation_sparse = self.sparse_encoder.encode_for_hybrid_search(
                    getattr(context_chunk, 'combined_content', context_chunk.original_chunk)
                )
                
                # Track embedding costs
                if self.enable_cost_tracking:
                    self.cost_tracker.track_embedding_usage(
                        box_doc.file_id, embedding_tokens,
                        model=self.embedding_generator.model
                    )
                
                # Extract legal entities for metadata
                entities = self.sparse_encoder.extract_legal_entities(
                    getattr(context_chunk, 'combined_content', context_chunk.original_chunk)
                )
                
                # Prepare chunk data with all vectors and metadata
                chunk_data = {
                    "content": getattr(context_chunk, 'combined_content', context_chunk.original_chunk),
                    "embedding": embedding,
                    "search_text": search_text,
                    "keywords_sparse": keyword_sparse,
                    "citations_sparse": citation_sparse,
                    "metadata": {
                        **chunk.metadata,
                        "has_context": bool(getattr(context_chunk, 'context', None)),
                        "original_length": len(chunk.content),
                        "context_length": len(getattr(context_chunk, 'context', '')),
                        "has_citations": bool(entities["citations"]),
                        "citation_count": len(entities["citations"]),
                        "has_monetary": bool(entities["monetary"]),
                        "has_dates": bool(entities["dates"]),
                        "subfolder": box_doc.subfolder_name or "root"  # Ensure subfolder is tracked
                    }
                }
                storage_chunks.append(chunk_data)
            
            # Step 9: Store in Qdrant vector database (using case name for collection)
            chunk_ids = self.vector_store.store_document_chunks(
                case_name=box_doc.case_name,  # Use case name, not folder name
                document_id=doc_hash,
                chunks=storage_chunks,
                use_hybrid=True  # Enable hybrid collection storage
            )
            
            logger.info(f"Successfully stored {len(chunk_ids)} chunks for {box_doc.name} in case '{box_doc.case_name}'")
            
            # Complete cost tracking
            if self.enable_cost_tracking:
                self.cost_tracker.finish_document(
                    doc_hash,
                    len(chunk_ids),
                    (datetime.utcnow() - start_time).total_seconds()
                )
            
            return ProcessingResult(
                document_id=doc_hash,
                file_name=box_doc.name,
                case_name=box_doc.case_name,
                status="success",
                chunks_created=len(chunk_ids),
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Error processing {box_doc.name}: {str(e)}")
            
            # Track failed document
            if self.enable_cost_tracking and 'doc_hash' in locals():
                self.cost_tracker.finish_document(
                    doc_hash,
                    0,
                    (datetime.utcnow() - start_time).total_seconds()
                )
            
            return ProcessingResult(
                document_id=box_doc.file_id,
                file_name=box_doc.name,
                case_name=box_doc.case_name,
                status="failed",
                chunks_created=0,
                error_message=str(e),
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _determine_document_type(self, filename: str, content: str) -> str:
        """Determine document type based on filename and content
        
        Args:
            filename: Document filename
            content: Document content (first portion)
            
        Returns:
            Document type string
        """
        filename_lower = filename.lower()
        content_lower = content[:1000].lower()  # Check first 1000 chars
        
        # Check filename patterns
        if "motion" in filename_lower:
            return "motion"
        elif "complaint" in filename_lower:
            return "complaint"
        elif "order" in filename_lower:
            return "court_order"
        elif "expert" in filename_lower:
            return "expert_report"
        elif "medical" in filename_lower or "record" in filename_lower:
            return "medical_record"
        elif "discovery" in filename_lower:
            return "discovery"
        elif "deposition" in filename_lower:
            return "deposition"
        
        # Check content patterns
        if "motion to" in content_lower:
            return "motion"
        elif "complaint" in content_lower and "plaintiff" in content_lower:
            return "complaint"
        elif "ordered" in content_lower and "court" in content_lower:
            return "court_order"
        elif "diagnosis" in content_lower or "treatment" in content_lower:
            return "medical_record"
        
        return "general"
    
    def search_case(self, case_name: str, query: str, 
                   limit: int = 10, use_hybrid: bool = True) -> List[SearchResult]:
        """Search within a specific case using vector or hybrid search
        
        Args:
            case_name: Case to search within
            query: Search query
            limit: Maximum results
            use_hybrid: Whether to use hybrid search
            
        Returns:
            List of search results
        """
        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)
        logger.info(f"Query type: {query_analysis['query_type']}")
        
        # Generate query embedding
        query_embedding, _ = self.embedding_generator.generate_embedding(query)
        
        if use_hybrid and settings.legal["enable_hybrid_search"]:
            # Generate sparse vectors
            keyword_sparse, citation_sparse = self.sparse_encoder.encode_for_hybrid_search(query)
            
            # Perform hybrid search
            results = self.vector_store.hybrid_search(
                case_name=case_name,
                query_text=query,
                query_embedding=query_embedding,
                keyword_indices=keyword_sparse,
                citation_indices=citation_sparse,
                limit=limit,
                filters=query_analysis.get("filters", {})
            )
        else:
            # Standard vector search
            results = self.vector_store.search_case_documents(
                case_name=case_name,
                query_embedding=query_embedding,
                limit=limit,
                filters=query_analysis.get("filters", {})
            )
        
        return results
    
    def _log_processing_summary(self):
        """Log processing summary statistics"""
        logger.info("=" * 50)
        logger.info("Processing Summary:")
        logger.info(f"Total documents processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Duplicates: {self.stats['duplicates']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        # Get deduplication stats
        dedup_stats = self.deduplicator.get_statistics()
        logger.info(f"Total unique documents in system: {dedup_stats['total_unique_documents']}")
        logger.info(f"Total duplicate instances: {dedup_stats['total_duplicate_instances']}")
        
        # Get case statistics from Qdrant
        try:
            # This would need to be implemented to get all unique cases
            logger.info("Qdrant collections initialized and ready")
        except Exception as e:
            logger.warning(f"Could not get Qdrant statistics: {str(e)}")
        
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
    
    def delete_document(self, case_name: str, document_id: str) -> int:
        """Delete a document from vector storage (for versioning)
        
        Args:
            case_name: Case name
            document_id: Document ID to delete
            
        Returns:
            Number of vectors deleted
        """
        return self.vector_store.delete_document_vectors(case_name, document_id)
    
    def optimize_vector_store(self):
        """Optimize vector store for better performance"""
        logger.info("Optimizing Qdrant collections...")
        self.vector_store.optimize_collection()
        logger.info("Optimization complete")


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
        
        # Test Qdrant connection (both vector store and deduplicator)
        try:
            # Test vector store collections
            collections = injector.vector_store.client.get_collections()
            logger.info(f"✓ Qdrant vector store connection successful - Collections: {[c.name for c in collections.collections]}")
            
            # Test deduplicator collection
            registry_info = injector.deduplicator.client.get_collection(
                injector.deduplicator.collection_name
            )
            logger.info(f"✓ Qdrant deduplicator connection successful - Registry points: {registry_info.points_count}")
        except Exception as e:
            logger.error(f"✗ Qdrant connection failed: {str(e)}")
        
        # Test OpenAI connection
        try:
            test_embedding, _ = injector.embedding_generator.generate_embedding("test")
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