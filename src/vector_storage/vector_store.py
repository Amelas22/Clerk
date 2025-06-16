"""
Vector storage module.
Manages storing and retrieving vectors from Supabase with case isolation.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from supabase import create_client, Client
from pgvector.utils import to_db_vector

from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector storage in Supabase with strict case isolation"""
    
    def __init__(self):
        """Initialize vector store with Supabase connection"""
        self.supabase: Client = create_client(
            settings.database.supabase_url,
            settings.database.supabase_service_key  # Use service key for admin access
        )
        self.collection_name = settings.vector.collection_name
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the vector collection exists with proper schema"""
        # Note: In production, use proper migrations
        try:
            # Test query
            self.supabase.table(self.collection_name).select("id").limit(1).execute()
            logger.info(f"Vector collection '{self.collection_name}' exists")
        except Exception as e:
            logger.warning(f"Vector collection might not exist: {str(e)}")
    
    def store_document_chunks(self, case_name: str, document_id: str,
                            chunks: List[Dict[str, Any]]) -> List[str]:
        """Store multiple chunks for a document
        
        Args:
            case_name: Name of the case (CRITICAL for isolation)
            document_id: Unique identifier for the document
            chunks: List of chunk dictionaries with content, embedding, search_text, and metadata
            
        Returns:
            List of stored chunk IDs
        """
        if not chunks:
            return []
        
        logger.info(f"Storing {len(chunks)} chunks for document {document_id} in case {case_name}")
        
        stored_ids = []
        
        try:
            # Prepare records for insertion
            records = []
            
            for chunk in chunks:
                # Generate unique ID for chunk
                chunk_id = str(uuid.uuid4())
                
                # Prepare record with STRICT case isolation
                record = {
                    "id": chunk_id,
                    "case_name": case_name,  # CRITICAL: Always set case_name
                    "document_id": document_id,
                    "content": chunk["content"],
                    "embedding": chunk["embedding"],  # Supabase handles vector type
                    "search_text": chunk.get("search_text", chunk["content"]),  # Full-text search
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "case_name": case_name,  # Double-ensure case isolation
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                }
                
                records.append(record)
                stored_ids.append(chunk_id)
            
            # Batch insert
            result = self.supabase.table(self.collection_name).insert(records).execute()
            
            logger.info(f"Successfully stored {len(result.data)} chunks")
            return stored_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise
    
    def search_case_documents(self, case_name: str, query_embedding: List[float],
                            limit: int = 10, threshold: float = 0.7) -> List[Dict]:
        """Search for similar chunks ONLY within a specific case
        
        Args:
            case_name: Name of the case to search within (CRITICAL)
            query_embedding: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            # CRITICAL: Always filter by case_name first
            # Use Supabase's vector similarity search with RPC function
            response = self.supabase.rpc(
                "search_case_vectors",
                {
                    "case_name_filter": case_name,
                    "query_embedding": query_embedding,
                    "match_threshold": threshold,
                    "match_count": limit
                }
            ).execute()
            
            results = []
            for item in response.data:
                # Double-check case isolation
                if item.get("case_name") != case_name:
                    logger.error(
                        f"CRITICAL: Case isolation breach detected! "
                        f"Expected case '{case_name}', got '{item.get('case_name')}'"
                    )
                    continue
                
                results.append({
                    "id": item["id"],
                    "content": item["content"],
                    "similarity": item["similarity"],
                    "metadata": item["metadata"],
                    "document_id": item["document_id"]
                })
            
            logger.info(f"Found {len(results)} matching chunks in case '{case_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise
    
    def delete_document_chunks(self, case_name: str, document_id: str) -> int:
        """Delete all chunks for a specific document
        
        Args:
            case_name: Name of the case (for verification)
            document_id: Document ID to delete
            
        Returns:
            Number of deleted chunks
        """
        try:
            # CRITICAL: Always include case_name in deletion
            result = self.supabase.table(self.collection_name).delete().match({
                "case_name": case_name,
                "document_id": document_id
            }).execute()
            
            count = len(result.data) if result.data else 0
            logger.info(f"Deleted {count} chunks for document {document_id}")
            
            return count
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            raise
    
    def get_case_statistics(self, case_name: str) -> Dict[str, Any]:
        """Get statistics for a specific case
        
        Args:
            case_name: Name of the case
            
        Returns:
            Dictionary with case statistics
        """
        try:
            # Get chunk count
            count_result = self.supabase.table(self.collection_name).select(
                "id", count="exact"
            ).eq("case_name", case_name).execute()
            
            # Get unique documents
            docs_result = self.supabase.table(self.collection_name).select(
                "document_id"
            ).eq("case_name", case_name).execute()
            
            unique_docs = set(item["document_id"] for item in docs_result.data or [])
            
            return {
                "case_name": case_name,
                "total_chunks": count_result.count or 0,
                "total_documents": len(unique_docs),
                "document_ids": list(unique_docs)
            }
            
        except Exception as e:
            logger.error(f"Error getting case statistics: {str(e)}")
            raise
    
    def verify_case_isolation(self, case_name: str, sample_size: int = 10) -> bool:
        """Verify that case isolation is working correctly
        
        Args:
            case_name: Case to verify
            sample_size: Number of chunks to sample
            
        Returns:
            True if isolation is verified
        """
        try:
            # Get sample of chunks
            result = self.supabase.table(self.collection_name).select(
                "case_name, metadata"
            ).eq("case_name", case_name).limit(sample_size).execute()
            
            # Verify all chunks belong to correct case
            for chunk in result.data or []:
                if chunk["case_name"] != case_name:
                    logger.error(
                        f"Case isolation violation: Expected '{case_name}', "
                        f"found '{chunk['case_name']}'"
                    )
                    return False
                
                # Also check metadata
                metadata_case = chunk.get("metadata", {}).get("case_name")
                if metadata_case and metadata_case != case_name:
                    logger.error(
                        f"Case isolation violation in metadata: Expected '{case_name}', "
                        f"found '{metadata_case}'"
                    )
                    return False
            
            logger.info(f"Case isolation verified for '{case_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying case isolation: {str(e)}")
            return False