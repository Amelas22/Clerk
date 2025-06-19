"""
Vector storage package for Clerk legal AI system.
Provides Qdrant vector storage with hybrid search capabilities.
"""

from .embeddings import EmbeddingGenerator
from .qdrant_store import QdrantVectorStore, SearchResult
from .sparse_encoder import SparseVectorEncoder, LegalQueryAnalyzer

# Legacy imports for backward compatibility (will be removed)
try:
    from .vector_store import VectorStore
    from .fulltext_search import FullTextSearchManager
except ImportError:
    VectorStore = None
    FullTextSearchManager = None

__all__ = [
    "EmbeddingGenerator",
    "QdrantVectorStore",
    "SearchResult",
    "SparseVectorEncoder", 
    "LegalQueryAnalyzer",
    # Legacy
    "VectorStore",
    "FullTextSearchManager"
]

__version__ = "2.0.0"  # Major version bump for Qdrant migration