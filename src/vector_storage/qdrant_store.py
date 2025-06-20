"""
Qdrant vector storage module.
Manages storing and retrieving vectors from Qdrant with folder-based isolation and hybrid search.
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import re
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    PointStruct, Filter, FieldCondition, MatchValue,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    SparseVectorParams, SparseIndexParams, NamedVector, NamedSparseVector,
    QueryRequest, SparseVector
)
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """Result from vector search"""
    id: str
    content: str
    case_name: str
    document_id: str
    score: float
    metadata: Dict[str, Any]
    search_type: str = "vector"  # "vector", "keyword", or "hybrid"


class QdrantVectorStore:
    """Manages vector storage in Qdrant with folder-based isolation and hybrid search"""
    
    def __init__(self):
        """Initialize Qdrant client"""
        self.config = settings.qdrant
        
        # Initialize synchronous client
        self.client = QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            prefer_grpc=self.config.prefer_grpc,
            timeout=self.config.timeout
        )
        
        # Initialize async client for batch operations
        self.async_client = AsyncQdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            prefer_grpc=self.config.prefer_grpc,
            timeout=self.config.timeout
        )
    
    def get_collection_name(self, folder_name: str) -> str:
        """Generate safe collection name from folder name"""
        # Sanitize folder name to valid Qdrant collection name
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', folder_name)
        return f"{sanitized}_hybrid" if settings.legal["enable_hybrid_search"] else sanitized
    
    def ensure_collection_exists(self, folder_name: str):
        """Ensure collection exists for a specific folder"""
        collection_name = self.get_collection_name(folder_name)
        
        # Check if collection exists
        if not self.client.collection_exists(collection_name):
            self.create_collection(collection_name)
            logger.info(f"Created collection for folder '{folder_name}': {collection_name}")
        
        return collection_name
    
    def create_collection(self, collection_name: str):
        """Create a new collection with hybrid configuration"""
        if settings.legal["enable_hybrid_search"]:
            self._create_hybrid_collection(collection_name)
        else:
            self._create_standard_collection(collection_name)
    
    def _create_standard_collection(self, collection_name: str):
        """Create standard vector collection"""
        quantization_config = None
        if hasattr(settings.vector, 'quantization') and settings.vector.quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.vector.embedding_dimensions,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=settings.vector.hnsw_m,
                ef_construct=settings.vector.hnsw_ef_construct,
                on_disk=False,
                max_indexing_threads=8
            ),
            quantization_config=quantization_config,
            on_disk_payload=False
        )
        self._create_payload_indexes(collection_name)
    
    def _create_hybrid_collection(self, collection_name: str):
        """Create hybrid collection with multiple vector types"""
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "semantic": VectorParams(
                    size=settings.vector.embedding_dimensions,
                    distance=Distance.COSINE
                ),
                "legal_concepts": VectorParams(
                    size=settings.vector.embedding_dimensions,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "keywords": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                "citations": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            hnsw_config=HnswConfigDiff(
                m=settings.vector.hnsw_m,
                ef_construct=settings.vector.hnsw_ef_construct,
                on_disk=False,
                max_indexing_threads=8
            ),
            on_disk_payload=False
        )
        self._create_payload_indexes(collection_name)
    
    def _create_payload_indexes(self, collection_name: str):
        """Create payload indexes for efficient filtering"""
        # Essential legal document fields
        index_fields = [
            ("case_name", "keyword"),        # CRITICAL for case isolation
            ("document_id", "keyword"),      # Document tracking
            ("document_type", "keyword"),    # Document categorization
            ("jurisdiction", "keyword"),     # Legal jurisdiction
            ("practice_areas", "keyword"),   # Multi-value field
            ("date_filed", "datetime"),      # Temporal filtering
            ("court_level", "keyword"),      # Court hierarchy
            ("has_citations", "bool"),       # Citation presence
            ("chunk_index", "integer"),      # Chunk ordering
            ("created_at", "datetime")       # Processing time
        ]
        
        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"Created index for {field_name} in {collection_name}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index {field_name} might already exist: {str(e)}")
    
    def index_document(self, folder_name: str, document):
        """Index a document in folder-specific collection"""
        collection_name = self.ensure_collection_exists(folder_name)
        
        if not document:
            return []
        
        logger.info(f"Indexing document in folder '{folder_name}'")
        
        stored_ids = []
        points = []
        
        try:
            # Generate unique ID for document
            document_id = str(uuid.uuid4())
            stored_ids.append(document_id)
            
            # Get document metadata safely
            document_metadata = document.get("metadata", {})
            
            # Build payload with folder-based isolation
            payload = {
                # Primary fields for filtering - DO NOT NEST THESE
                "folder_name": folder_name,
                "document_id": document_id,
                
                # Content
                "content": document["content"],
                "search_text": document.get("search_text", document["content"]),
                
                # Document metadata
                "document_type": document.get("document_type", ""),
                
                # System metadata
                "indexed_at": datetime.utcnow().isoformat(),
                "vector_version": "1.0"
            }
            
            # Add document metadata fields individually to avoid overwriting critical fields
            # Skip folder_name if it exists in metadata to prevent overwriting
            for key, value in document_metadata.items():
                if key not in ["folder_name", "document_id"]:
                    payload[key] = value
            
            # Create point for standard collection
            if settings.legal["enable_hybrid_search"]:
                # For hybrid collections, we need to specify multiple vectors
                point = PointStruct(
                    id=document_id,
                    vector={
                        "semantic": document["embedding"],
                        "legal_concepts": document["embedding"]
                    },
                    payload=payload
                )
            else:
                # For standard collections, use single vector
                point = PointStruct(
                    id=document_id,
                    vector=document["embedding"],
                    payload=payload
                )
            
            points.append(point)
            
            # Batch upload
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Successfully indexed {len(stored_ids)} documents")
            return stored_ids
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    def _store_hybrid_document(self, folder_name: str, document, document_ids: List[str]):
        """Store document in hybrid collection with multiple vector types"""
        try:
            hybrid_points = []
            
            # Extract sparse vectors if available
            keywords_sparse = document.get("keywords_sparse", {})
            citations_sparse = document.get("citations_sparse", {})
            
            # Get document metadata safely
            document_metadata = document.get("metadata", {})
            
            # Build payload - same structure as standard collection
            payload = {
                # Primary fields for filtering - DO NOT NEST THESE
                "folder_name": folder_name,
                "document_id": document_ids[0],
                "content": document["content"],
                "search_text": document.get("search_text", document["content"]),
            }
            
            # Add metadata fields individually
            for key, value in document_metadata.items():
                if key not in ["folder_name", "document_id"]:
                    payload[key] = value
            
            # Prepare sparse vectors with proper formatting
            sparse_vectors = {}
            
            # Format keywords sparse vector if available
            if keywords_sparse:
                sparse_vectors["keywords"] = {
                    "indices": [int(idx) for idx in keywords_sparse.keys()],
                    "values": [float(val) for val in keywords_sparse.values()]
                }
            else:
                # Provide empty sparse vector structure
                sparse_vectors["keywords"] = {"indices": [], "values": []}
            
            # Format citations sparse vector if available
            if citations_sparse:
                sparse_vectors["citations"] = {
                    "indices": [int(idx) for idx in citations_sparse.keys()],
                    "values": [float(val) for val in citations_sparse.values()]
                }
            else:
                # Provide empty sparse vector structure
                sparse_vectors["citations"] = {"indices": [], "values": []}
            
            # Create hybrid point with vectors in dictionary format
            point = PointStruct(
                id=document_ids[0],
                vector={
                    "semantic": document["embedding"],
                    "legal_concepts": document["embedding"],  # Same for now
                    **sparse_vectors  # Add sparse vectors
                },
                payload=payload
            )
            
            hybrid_points.append(point)
            
            # Batch upload to hybrid collection
            self.client.upsert(
                collection_name=self.get_collection_name(folder_name),
                points=hybrid_points,
                wait=True
            )
            
        except Exception as e:
            logger.error(f"Error storing hybrid document: {str(e)}")
            # Don't raise - hybrid is optional enhancement
    
    def search_documents(self, folder_name: str, query_embedding: List[float],
                        limit: int = 10, threshold: float = 0.7,
                        filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents ONLY within a specific folder
        
        Args:
            folder_name: Name of the folder to search within (CRITICAL)
            query_embedding: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Additional filters to apply
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Build folder isolation filter
            folder_filter = Filter(
                must=[
                    FieldCondition(
                        key="folder_name",
                        match=MatchValue(value=folder_name)
                    )
                ]
            )
            
            # Add additional filters if provided
            if filters:
                for key, value in filters.items():
                    folder_filter.must.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            # Log filter for debugging
            logger.debug(f"Searching with filter: folder_name='{folder_name}'")
            
            # Perform search
            results = self.client.search(
                collection_name=self.get_collection_name(folder_name),
                query_vector=query_embedding,
                query_filter=folder_filter,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to SearchResult objects
            search_results = []
            for point in results:
                # CRITICAL: Double-check folder isolation
                point_folder_name = point.payload.get("folder_name")
                if point_folder_name != folder_name:
                    logger.error(
                        f"CRITICAL: Folder isolation breach detected! "
                        f"Expected: {folder_name}, Got: {point_folder_name}"
                    )
                    continue
                
                search_results.append(SearchResult(
                    id=str(point.id),
                    content=point.payload.get("content", ""),
                    case_name=point.payload.get("case_name", ""),
                    document_id=point.payload.get("document_id", ""),
                    score=point.score,
                    metadata=point.payload,
                    search_type="vector"
                ))
            
            logger.debug(f"Found {len(search_results)} results for folder '{folder_name}'")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    async def hybrid_search(
        self, 
        folder_name: str, 
        query: str,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform hybrid search in folder-specific collection"""
        collection_name = self.ensure_collection_exists(folder_name)
        
        # Perform search
        search_results = await self.async_client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(
                name="semantic",
                vector=query_embedding
            ),
            limit=limit
        )
        
        # Convert to SearchResult
        results = []
        for hit in search_results:
            payload = hit.payload or {}
            results.append(SearchResult(
                id=hit.id,
                score=hit.score,
                content=payload.get("content", ""),
                metadata=payload
            ))
        
        return results
    
    def _process_results(self, results):
        # Convert to SearchResult objects
        search_results = []
        for point in results:
            search_results.append(SearchResult(
                id=str(point.id),
                content=point.payload.get("content", ""),
                case_name=point.payload.get("case_name", ""),
                document_id=point.payload.get("document_id", ""),
                score=point.score,
                metadata=point.payload,
                search_type="hybrid"
            ))
        
        return search_results
    
    def _combine_search_results(self, results: List, limit: int):
        """Combine search results by sorting by score, handling both ScoredPoint and tuples"""
        # Convert tuples to ScoredPoint if necessary
        from qdrant_client.models import ScoredPoint
        converted_results = []
        for item in results:
            if isinstance(item, ScoredPoint):
                converted_results.append(item)
            elif isinstance(item, tuple):
                # Try to convert tuple to ScoredPoint
                # Assuming the tuple has the same order as ScoredPoint fields
                if len(item) >= 6:
                    # We'll create a ScoredPoint object from the tuple
                    try:
                        scored_point = ScoredPoint(
                            id=item[0],
                            version=item[1],
                            score=item[2],
                            payload=item[3],
                            vector=item[4] if len(item) > 4 else None,
                            vector_name=item[5] if len(item) > 5 else None
                        )
                        converted_results.append(scored_point)
                    except Exception as e:
                        logger.error(f"Error converting tuple to ScoredPoint: {e}")
                else:
                    logger.error(f"Cannot convert tuple to ScoredPoint: insufficient fields")
            else:
                logger.error(f"Unsupported result type: {type(item)}")
        
        # Sort by score
        sorted_results = sorted(converted_results, key=lambda x: x.score, reverse=True)
        return sorted_results[:limit]
    
    def delete_document_vectors(self, folder_name: str, document_id: str) -> int:
        """Delete all vectors for a specific document
        
        Args:
            folder_name: Folder name (for verification)
            document_id: Document to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            # Build filter for document within folder
            standard_filter = Filter(
                must=[
                    FieldCondition(
                        key="folder_name",
                        match=MatchValue(value=folder_name)
                    ),
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Count before deletion
            count_before = self.client.count(
                collection_name=self.get_collection_name(folder_name),
                count_filter=standard_filter
            ).count
            
            # Delete points
            self.client.delete(
                collection_name=self.get_collection_name(folder_name),
                points_selector=standard_filter
            )
            
            # Also delete from hybrid collection
            if settings.legal["enable_hybrid_search"]:
                try:
                    self.client.delete(
                        collection_name=self.get_collection_name(folder_name),
                        points_selector=standard_filter
                    )
                except Exception as e:
                    logger.warning(f"Could not delete from hybrid collection: {str(e)}")
            
            logger.info(f"Deleted {count_before} vectors for document {document_id}")
            return count_before
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {str(e)}")
            raise
    
    def get_folder_statistics(self, folder_name: str) -> Dict[str, Any]:
        """Get statistics for a specific folder
        
        Args:
            folder_name: Folder to get statistics for
            
        Returns:
            Dictionary with folder statistics
        """
        try:
            folder_filter = Filter(
                must=[
                    FieldCondition(
                        key="folder_name",
                        match=MatchValue(value=folder_name)
                    )
                ]
            )
            
            # Get counts
            total_chunks = self.client.count(
                collection_name=self.get_collection_name(folder_name),
                count_filter=folder_filter
            ).count
            
            # Get unique documents (this is approximate)
            # In production, you might want to maintain a separate index
            search_results = self.client.scroll(
                collection_name=self.get_collection_name(folder_name),
                scroll_filter=folder_filter,
                limit=10000,  # Adjust based on expected folder size
                with_payload=["document_id"],
                with_vectors=False
            )
            
            unique_documents = set()
            for point in search_results[0]:
                unique_documents.add(point.payload.get("document_id"))
            
            return {
                "folder_name": folder_name,
                "total_chunks": total_chunks,
                "unique_documents": len(unique_documents),
                "document_ids": list(unique_documents)
            }
            
        except Exception as e:
            logger.error(f"Error getting folder statistics: {str(e)}")
            return {
                "folder_name": folder_name,
                "total_chunks": 0,
                "unique_documents": 0,
                "error": str(e)
            }
    
    def optimize_collection(self, folder_name: str):
        """Optimize collection for better performance"""
        try:
            logger.info("Starting collection optimization...")
            
            # Update optimizer config for faster indexing
            self.client.update_collection(
                collection_name=self.get_collection_name(folder_name),
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=50000,
                    flush_interval_sec=5,
                    max_optimization_threads=8
                )
            )
            
            # Trigger optimization
            self.client.update_collection(
                collection_name=self.get_collection_name(folder_name),
                optimizers_config=OptimizersConfigDiff(
                    max_optimization_threads=8
                )
            )
            
            logger.info("Collection optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing collection: {str(e)}")
            raise
    
    def close(self):
        """Close client connections"""
        try:
            if hasattr(self.async_client, 'close'):
                asyncio.run(self.async_client.close())
            logger.info("Closed Qdrant connections")
        except Exception as e:
            logger.warning(f"Error closing connections: {str(e)}")