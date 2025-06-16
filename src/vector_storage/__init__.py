"""
Vector storage package for Clerk legal AI system.
Handles embeddings generation and vector database operations.
"""

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .fulltext_search import FullTextSearchManager, SearchResult

__all__ = [
    "EmbeddingGenerator",
    "VectorStore",
    "FullTextSearchManager",
    "SearchResult"
]

__version__ = "0.1.0"