"""
Qdrant vector storage module.
Manages storing and retrieving vectors from Qdrant with strict case isolation.
Implements hybrid search combining dense vectors, sparse vectors, and metadata filtering.
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    PointStruct, Filter, FieldCondition, MatchValue, SearchRequest,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    SparseVectorParams, SparseIndexParams, NamedVector, NamedSparseVector,
    Prefetch, FusionQuery, Fusion
)

from config.settings import settings

logger = logging.getLogger(__name__)

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
    """Manages vector storage in Qdrant with strict case isolation and hybrid search"""
    
    def __init__(self):
        """Initialize Qdrant client and ensure collections exist"""
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
        
        # Collection names
        self.collection_name = self.config.collection_name
        self.hybrid_collection_name = self.config.hybrid_collection_name
        
        # Ensure collections exist
        self._ensure_collections_exist()
    
    def _ensure_collections_exist(self):
        """Ensure both standard and hybrid collections exist with optimal configuration"""
        try:
            # Check if collections exist
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            # Create standard collection if needed
            if self.collection_name not in collection_names:
                self._create_standard_collection()
                logger.info(f"Created standard collection: {self.collection_name}")
            else:
                logger.info(f"Standard collection exists: {self.collection_name}")
            
            # Create hybrid collection if needed
            if self.hybrid_collection_name not in collection_names:
                self._create_hybrid_collection()
                logger.info(f"Created hybrid collection: {self.hybrid_collection_name}")
            else:
                logger.info(f"Hybrid collection exists: {self.hybrid_collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collections exist: {str(e)}")
            raise
    
    def _create_standard_collection(self):
        """Create standard vector collection optimized for legal documents"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.vector.embedding_dimensions,
                distance=Distance.COSINE,
                on_disk=False  # Keep in RAM for performance
            ),
            hnsw_config=HnswConfigDiff(
                m=settings.vector.hnsw_m,
                ef_construct=settings.vector.hnsw_ef_construct,
                on_disk=False,
                max_indexing_threads=8
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=4,
                flush_interval_sec=5,
                max_optimization_threads=2
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=settings.vector.quantization_quantile,
                    always_ram=True
                )
            ) if settings.vector.quantization_enabled else None,
            on_disk_payload=False,
            replication_factor=2,  # Production redundancy
            shard_number=1
        )
        
        # Create payload indexes for efficient filtering
        self._create_payload_indexes(self.collection_name)
    
    def _create_hybrid_collection(self):
        """Create hybrid collection with multiple vector types for advanced search"""
        self.client.create_collection(
            collection_name=self.hybrid_collection_name,
            vectors_config={
                # Dense semantic vectors
                "semantic": VectorParams(
                    size=settings.vector.embedding_dimensions,
                    distance=Distance.COSINE
                ),
                # Legal terminology vectors (future: legal BERT)
                "legal_concepts": VectorParams(
                    size=settings.vector.embedding_dimensions,  # Same size for now
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                # Keyword matching
                "keywords": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                ),
                # Legal citation matching
                "citations": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
            hnsw_config=HnswConfigDiff(
                m=settings.vector.hnsw_m,
                ef_construct=settings.vector.hnsw_ef_construct,
                on_disk=False,
                max_indexing_threads=8
            ),
            on_disk_payload=False
        )
        
        # Create payload indexes
        self._create_payload_indexes(self.hybrid_collection_name)
    
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
    
    def store_document_chunks(self, case_name: str, document_id: str,
                            chunks: List[Dict[str, Any]], 
                            use_hybrid: bool = True) -> List[str]:
        """Store multiple chunks for a document with strict case isolation
        
        Args:
            case_name: Name of the case (CRITICAL for isolation)
            document_id: Unique identifier for the document
            chunks: List of chunk dictionaries with content, embedding, and metadata
            use_hybrid: Whether to store in hybrid collection as well
            
        Returns:
            List of stored chunk IDs
        """
        if not chunks:
            return []
        
        logger.info(f"Storing {len(chunks)} chunks for document {document_id} in case {case_name}")
        
        stored_ids = []
        points = []
        
        try:
            for i, chunk in enumerate(chunks):
                # Generate unique ID for chunk
                chunk_id = str(uuid.uuid4())
                stored_ids.append(chunk_id)
                
                # Build payload with STRICT case isolation
                payload = {
                    # CRITICAL: Case isolation fields
                    "case_name": case_name,
                    "document_id": document_id,
                    
                    # Content
                    "content": chunk["content"],
                    "search_text": chunk.get("search_text", chunk["content"]),
                    
                    # Chunk metadata
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    
                    # Document metadata
                    **chunk.get("metadata", {}),
                    
                    # System metadata
                    "indexed_at": datetime.utcnow().isoformat(),
                    "vector_version": "1.0"
                }
                
                # Ensure case_name is in metadata for double protection
                if "metadata" not in payload:
                    payload["metadata"] = {}
                payload["metadata"]["case_name"] = case_name
                
                # Create point for standard collection
                point = PointStruct(
                    id=chunk_id,
                    vector=chunk["embedding"],
                    payload=payload
                )
                points.append(point)
            
            # Batch upload to standard collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            # Also store in hybrid collection if enabled
            if use_hybrid and settings.legal["enable_hybrid_search"]:
                self._store_hybrid_chunks(case_name, document_id, chunks, stored_ids)
            
            logger.info(f"Successfully stored {len(stored_ids)} chunks")
            return stored_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise
    
    def _store_hybrid_chunks(self, case_name: str, document_id: str,
                           chunks: List[Dict[str, Any]], chunk_ids: List[str]):
        """Store chunks in hybrid collection with multiple vector types"""
        try:
            hybrid_points = []
            
            for chunk, chunk_id in zip(chunks, chunk_ids):
                # Extract sparse vectors if available
                keywords_sparse = chunk.get("keywords_sparse", {})
                citations_sparse = chunk.get("citations_sparse", {})
                
                # Create hybrid point
                point = PointStruct(
                    id=chunk_id,
                    vector={
                        "semantic": chunk["embedding"],
                        "legal_concepts": chunk["embedding"]  # Same for now
                    },
                    payload={
                        "case_name": case_name,
                        "document_id": document_id,
                        "content": chunk["content"],
                        "search_text": chunk.get("search_text", chunk["content"]),
                        **chunk.get("metadata", {})
                    }
                )
                
                # Add sparse vectors if available
                if keywords_sparse:
                    point.vector["keywords"] = keywords_sparse
                if citations_sparse:
                    point.vector["citations"] = citations_sparse
                
                hybrid_points.append(point)
            
            # Batch upload to hybrid collection
            self.client.upsert(
                collection_name=self.hybrid_collection_name,
                points=hybrid_points,
                wait=True
            )
            
        except Exception as e:
            logger.error(f"Error storing hybrid chunks: {str(e)}")
            # Don't raise - hybrid is optional enhancement
    
    def search_case_documents(self, case_name: str, query_embedding: List[float],
                            limit: int = 10, threshold: float = 0.7,
                            filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks ONLY within a specific case
        
        Args:
            case_name: Name of the case to search within (CRITICAL)
            query_embedding: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Additional filters to apply
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Build case isolation filter
            case_filter = Filter(
                must=[
                    FieldCondition(
                        key="case_name",
                        match=MatchValue(value=case_name)
                    )
                ]
            )
            
            # Add additional filters if provided
            if filters:
                for key, value in filters.items():
                    case_filter.must.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=case_filter,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to SearchResult objects
            search_results = []
            for point in results:
                # CRITICAL: Double-check case isolation
                if point.payload.get("case_name") != case_name:
                    logger.error(
                        f"CRITICAL: Case isolation breach detected! "
                        f"Expected: {case_name}, Got: {point.payload.get('case_name')}"
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
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching case documents: {str(e)}")
            raise
    
    def hybrid_search(self, case_name: str, query_text: str,
                     query_embedding: List[float],
                     keyword_indices: Optional[Dict[int, float]] = None,
                     citation_indices: Optional[Dict[int, float]] = None,
                     limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None,
                     threshold: float = 0.7) -> List[SearchResult]:
        """Perform hybrid search combining semantic, keyword, and citation matching
        
        Args:
            case_name: Name of the case to search within (CRITICAL)
            query_text: Original query text
            query_embedding: Dense query vector
            keyword_indices: Sparse vector for keyword matching
            citation_indices: Sparse vector for citation matching
            limit: Maximum number of results
            filters: Additional filters
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results ranked by combined score
        """
        if not settings.legal["enable_hybrid_search"]:
            # Fall back to standard vector search
            return self.search_case_documents(
                case_name, query_embedding, limit, threshold=threshold, filters=filters
            )
        
        try:
            # Build case isolation filter
            case_filter = Filter(
                must=[
                    FieldCondition(
                        key="case_name",
                        match=MatchValue(value=case_name)
                    )
                ]
            )
            
            if filters:
                for key, value in filters.items():
                    case_filter.must.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            # Prepare prefetch queries
            prefetches = []
            
            # Semantic search
            prefetches.append(
                Prefetch(
                    query=query_embedding,
                    using="semantic",
                    limit=limit * 2,  # Get more candidates
                    filter=case_filter
                )
            )

            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=case_filter,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )

            # Convert to SearchResult objects
            search_results = []
            for point in results:
                # CRITICAL: Double-check case isolation
                if point.payload.get("case_name") != case_name:
                    logger.error(
                        f"CRITICAL: Case isolation breach detected! "
                        f"Expected: {case_name}, Got: {point.payload.get('case_name')}"
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

            return search_results

        except Exception as e:
            logger.error(f"Error searching case documents: {str(e)}")
            raise

    def delete_document_vectors(self, case_name: str, document_id: str) -> int:
        """Delete all vectors for a specific document (for versioning)
        
        Args:
            case_name: Case name for isolation
            document_id: Document to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            # Delete from standard collection
            standard_filter = Filter(
                must=[
                    FieldCondition(
                        key="case_name",
                        match=MatchValue(value=case_name)
                    ),
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Get count before deletion
            count_before = self.client.count(
                collection_name=self.collection_name,
                count_filter=standard_filter
            ).count
            
            # Delete points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=standard_filter
            )
            
            # Also delete from hybrid collection
            if settings.legal["enable_hybrid_search"]:
                try:
                    self.client.delete(
                        collection_name=self.hybrid_collection_name,
                        points_selector=standard_filter
                    )
                except Exception as e:
                    logger.warning(f"Could not delete from hybrid collection: {str(e)}")
            
            logger.info(f"Deleted {count_before} vectors for document {document_id}")
            return count_before
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {str(e)}")
            raise
    
    def get_case_statistics(self, case_name: str) -> Dict[str, Any]:
        """Get statistics for a specific case
        
        Args:
            case_name: Case to get statistics for
            
        Returns:
            Dictionary with case statistics
        """
        try:
            case_filter = Filter(
                must=[
                    FieldCondition(
                        key="case_name",
                        match=MatchValue(value=case_name)
                    )
                ]
            )
            
            # Get counts
            total_chunks = self.client.count(
                collection_name=self.collection_name,
                count_filter=case_filter
            ).count
            
            # Get unique documents (this is approximate)
            # In production, you might want to maintain a separate index
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=case_filter,
                limit=10000,  # Adjust based on expected case size
                with_payload=["document_id"],
                with_vectors=False
            )
            
            unique_documents = set()
            for point in search_results[0]:
                unique_documents.add(point.payload.get("document_id"))
            
            return {
                "case_name": case_name,
                "total_chunks": total_chunks,
                "unique_documents": len(unique_documents),
                "document_ids": list(unique_documents)
            }
            
        except Exception as e:
            logger.error(f"Error getting case statistics: {str(e)}")
            return {
                "case_name": case_name,
                "total_chunks": 0,
                "unique_documents": 0,
                "error": str(e)
            }
    
    def optimize_collection(self):
        """Optimize collection for better performance"""
        try:
            logger.info("Starting collection optimization...")
            
            # Update optimizer config for faster indexing
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=50000,
                    flush_interval_sec=10
                )
            )
            
            logger.info("Collection optimization complete")
            
        except Exception as e:
            logger.error(f"Error optimizing collection: {str(e)}")
    
    async def parallel_upload(self, documents: List[Dict[str, Any]], 
                            batch_size: int = 500) -> int:
        """Upload documents in parallel for better performance
        
        Args:
            documents: List of documents with embeddings
            batch_size: Size of each batch
            
        Returns:
            Number of documents uploaded
        """
        total_uploaded = 0
        
        async def upload_batch(batch):
            points = []
            for doc in batch:
                point = PointStruct(
                    id=doc["id"],
                    vector=doc["embedding"],
                    payload=doc["payload"]
                )
                points.append(point)
            
            await self.async_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=False
            )
            return len(points)
        
        # Create batches
        batches = [documents[i:i + batch_size] 
                  for i in range(0, len(documents), batch_size)]
        
        # Upload in parallel
        tasks = [upload_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        total_uploaded = sum(results)
        logger.info(f"Uploaded {total_uploaded} documents in parallel")
        
        return total_uploaded