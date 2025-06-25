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
import hashlib

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    PointStruct, Filter, FieldCondition, MatchValue,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    SparseVectorParams, SparseIndexParams, NamedVector, NamedSparseVector,
    QueryRequest, SparseVector, models
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
        return f"{sanitized}" if settings.legal["enable_hybrid_search"] else sanitized
    
    def ensure_collection_exists(self, folder_name: str):
        """Ensure collection exists for a specific folder"""
        collection_name = self.get_collection_name(folder_name)
        
        try:
            # Check if collection exists
            exists = self.client.collection_exists(collection_name)
            logger.debug(f"Collection '{collection_name}' exists: {exists}")
            
            if not exists:
                logger.info(f"Creating new collection: {collection_name}")
                self.create_collection(collection_name)
                
                # Verify creation
                if self.client.collection_exists(collection_name):
                    logger.info(f"Successfully created collection: {collection_name}")
                else:
                    raise Exception(f"Collection creation failed - collection still doesn't exist: {collection_name}")
            
            return collection_name
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            logger.error(f"Folder: {folder_name}, Collection: {collection_name}")
            logger.error(f"Qdrant URL: {self.config.url}")
            
            # Try to create collection with minimal config as fallback
            try:
                logger.warning("Attempting to create collection with minimal configuration...")
                self._create_minimal_collection(collection_name)
                return collection_name
            except Exception as fallback_error:
                logger.error(f"Fallback collection creation also failed: {str(fallback_error)}")
                raise Exception(f"Failed to create collection '{collection_name}': {str(e)}")
    
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

    def store_document_chunks(self, case_name: str, document_id: str, 
                            chunks: List[Dict[str, Any]], use_hybrid: bool = False) -> List[str]:
        """Store multiple chunks for a document in the case-specific collection
        
        Args:
            case_name: Case name (used as collection name after sanitization)
            document_id: Unique document identifier
            chunks: List of chunk dictionaries with embeddings and metadata
            use_hybrid: Whether to use hybrid search features
            
        Returns:
            List of chunk IDs that were stored
        """
        if not chunks:
            return []
        
        # Ensure collection exists for this case
        collection_name = self.ensure_collection_exists(case_name)
        
        logger.info(f"Storing {len(chunks)} chunks for document {document_id} in collection '{collection_name}'")
        
        stored_ids = []
        points = []
        
        # Generate base ID using timestamp + hash for uniqueness
        # This ensures unique IDs even if multiple documents are processed simultaneously
        timestamp_part = int(datetime.utcnow().timestamp() * 1000)  # Millisecond precision
        hash_part = abs(hash(document_id)) % 1000000  # 6 digits from doc hash
        base_id = timestamp_part * 1000000 + hash_part
        
        try:
            for i, chunk in enumerate(chunks):
                # Generate unique integer ID for this chunk
                chunk_id = base_id + i
                stored_ids.append(str(chunk_id))
                
                # Get chunk metadata safely
                chunk_metadata = chunk.get("metadata", {})
                
                # Build comprehensive payload
                payload = {
                    # Primary identifiers
                    "case_name": case_name,
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_reference": f"{document_id}_{i}",
                    
                    # Content fields
                    "content": chunk["content"],
                    "search_text": chunk.get("search_text", chunk["content"]),
                    
                    # Document metadata
                    "document_name": chunk_metadata.get("document_name", ""),
                    "document_type": chunk_metadata.get("document_type", ""),
                    "document_path": chunk_metadata.get("document_path", ""),
                    "document_link": chunk_metadata.get("document_link", ""),
                    "subfolder": chunk_metadata.get("subfolder", "root"),
                    "folder_path": chunk_metadata.get("folder_path", ""),
                    
                    # Processing metadata
                    "has_context": chunk_metadata.get("has_context", False),
                    "original_length": chunk_metadata.get("original_length", 0),
                    "context_length": chunk_metadata.get("context_length", 0),
                    
                    # Legal entity flags
                    "has_citations": chunk_metadata.get("has_citations", False),
                    "citation_count": chunk_metadata.get("citation_count", 0),
                    "has_monetary": chunk_metadata.get("has_monetary", False),
                    "has_dates": chunk_metadata.get("has_dates", False),
                    
                    # Document info
                    "page_count": chunk_metadata.get("page_count", 0),
                    "file_size": chunk_metadata.get("file_size", 0),
                    "modified_at": chunk_metadata.get("modified_at", ""),
                    
                    # System metadata
                    "indexed_at": datetime.utcnow().isoformat(),
                    "vector_version": "1.0",
                    "total_chunks": chunk_metadata.get("total_chunks", len(chunks))
                }
                
                # Create point based on collection type
                if settings.legal.get("enable_hybrid_search", False):
                    # For hybrid collections - prepare vectors dictionary
                    vectors = {
                        "semantic": chunk["embedding"]
                    }
                    
                    # Only add legal_concepts if it's different from semantic
                    # For now, they're the same, so we'll skip to save space
                    # vectors["legal_concepts"] = chunk["embedding"]
                    
                    # Handle sparse vectors if present and valid
                    if "keywords_sparse" in chunk and isinstance(chunk["keywords_sparse"], dict):
                        sparse_indices, sparse_values = self._convert_sparse_to_lists(chunk["keywords_sparse"])
                        if sparse_indices:  # Only add if we have valid data
                            try:
                                vectors["keywords"] = SparseVector(
                                    indices=sparse_indices,
                                    values=sparse_values
                                )
                            except Exception as e:
                                logger.warning(f"Could not add keywords sparse vector: {e}")
                    
                    if "citations_sparse" in chunk and isinstance(chunk["citations_sparse"], dict):
                        sparse_indices, sparse_values = self._convert_sparse_to_lists(chunk["citations_sparse"])
                        if sparse_indices:
                            try:
                                vectors["citations"] = SparseVector(
                                    indices=sparse_indices,
                                    values=sparse_values
                                )
                            except Exception as e:
                                logger.warning(f"Could not add citations sparse vector: {e}")
                    
                    # Create point with multiple vectors
                    point = PointStruct(
                        id=chunk_id,
                        vector=vectors,
                        payload=payload
                    )
                else:
                    # Standard collection - single embedding vector
                    point = PointStruct(
                        id=chunk_id,
                        vector=chunk["embedding"],
                        payload=payload
                    )
                
                points.append(point)
            
            # Batch upload all points
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Successfully stored {len(stored_ids)} chunks in collection '{collection_name}'")
            
            return stored_ids
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            # Only log collection name if it was successfully created
            if 'collection_name' in locals():
                logger.error(f"Collection: {collection_name}, Document: {document_id}")
            else:
                logger.error(f"Case: {case_name}, Document: {document_id}")
            if 'points' in locals() and points:
                logger.debug(f"Failed with {len(points)} points prepared")
            raise

    def _convert_sparse_to_lists(self, sparse_dict: Dict[Any, float]) -> Tuple[List[int], List[float]]:
        """Convert sparse vector dictionary to lists of indices and values
        
        Args:
            sparse_dict: Dictionary with either int or string keys
            
        Returns:
            Tuple of (indices, values) lists
        """
        indices = []
        values = []
        
        try:
            for key, value in sparse_dict.items():
                if isinstance(key, int):
                    # Already an integer index
                    indices.append(key)
                    values.append(float(value))
                elif isinstance(key, str) and key.isdigit():
                    # String that represents an integer
                    indices.append(int(key))
                    values.append(float(value))
                else:
                    # String key - hash to index
                    # Use consistent hashing for reproducibility
                    idx = abs(hash(str(key))) % 100000
                    indices.append(idx)
                    values.append(float(value))
            
            # Sort by indices for consistency
            if indices:
                sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
                indices, values = zip(*sorted_pairs)
                return list(indices), list(values)
            
        except Exception as e:
            logger.warning(f"Error converting sparse vector: {e}")
        
        return [], []

    def _convert_sparse_vector_for_storage(self, sparse_dict: Dict[str, float]) -> Dict[str, List]:
        """Convert string-keyed sparse vector to Qdrant format
        
        The sparse encoder returns vectors with string keys (tokens),
        but Qdrant needs integer indices.
        
        Args:
            sparse_dict: Dictionary with string keys and float values
            
        Returns:
            Dictionary with 'indices' and 'values' lists
        """
        # For now, we'll use a simple hash-based approach
        # In production, you'd want a consistent vocabulary mapping
        indices = []
        values = []
        
        for token, value in sparse_dict.items():
            # Simple hash to get an index (mod by large prime for distribution)
            index = abs(hash(token)) % 1000000
            indices.append(index)
            values.append(float(value))
        
        return {
            "indices": indices,
            "values": values
        }
        
    def sanitize_collection_name(self, case_name: str) -> str:
        """Sanitize and truncate collection name for Qdrant
        
        Args:
            case_name: Original case/folder name
            
        Returns:
            Sanitized collection name (max 63 chars, alphanumeric + underscore)
        """
        # Remove special characters, keep only alphanumeric and spaces
        sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', case_name)
        
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized.strip())
        
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Convert to lowercase for consistency
        sanitized = sanitized.lower()
        
        # Handle empty result
        if not sanitized:
            sanitized = "unnamed_case"
        
        # If name is too long, truncate and add hash
        max_length = 63  # Safe limit for most filesystems
        
        if len(sanitized) > max_length:
            # Keep first part of name and add hash suffix
            hash_suffix = hashlib.md5(case_name.encode()).hexdigest()[:8]
            
            # Calculate how much of the name we can keep
            # Format: name_hash (1 underscore + 8 chars for hash)
            available_length = max_length - 9
            
            sanitized = f"{sanitized[:available_length]}_{hash_suffix}"
        
        # Ensure it doesn't start with a number (some systems don't like this)
        if sanitized[0].isdigit():
            sanitized = f"case_{sanitized}"
        
        # Final length check
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        logger.debug(f"Collection name: '{case_name}' -> '{sanitized}'")
        
        return sanitized

    def get_case_name_mapping(self) -> Dict[str, str]:
        """Get mapping of sanitized collection names to original case names
        
        Returns:
            Dictionary mapping collection names to original case names
        """
        try:
            # Get all collections
            collections = self.client.get_collections().collections
            
            mapping = {}
            for collection in collections:
                # Try to get original case name from collection metadata
                try:
                    # First, check if we stored the original name in the collection
                    # This would require updating the create_collection method
                    info = self.client.get_collection(collection.name)
                    
                    # For now, just store the collection name
                    mapping[collection.name] = collection.name
                    
                except Exception as e:
                    logger.debug(f"Could not get info for collection {collection.name}: {e}")
                    mapping[collection.name] = collection.name
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error getting case name mapping: {str(e)}")
            return {}

    def list_cases(self) -> List[Dict[str, Any]]:
        """List all cases (collections) in the system
        
        Returns:
            List of case information dictionaries
        """
        try:
            collections = self.client.get_collections().collections
            cases = []
            
            for collection in collections:
                try:
                    # Get collection info
                    info = self.client.get_collection(collection.name)
                    
                    cases.append({
                        "collection_name": collection.name,
                        "original_name": collection.name,  # Would be from metadata
                        "vector_count": info.vectors_count,
                        "points_count": info.points_count,
                        "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
                        "status": info.status
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not get details for {collection.name}: {e}")
                    cases.append({
                        "collection_name": collection.name,
                        "error": str(e)
                    })
            
            return cases
            
        except Exception as e:
            logger.error(f"Error listing cases: {str(e)}")
            return []
            
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