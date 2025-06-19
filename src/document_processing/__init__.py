"""
Document processing package for Clerk legal AI system.
Handles PDF extraction, chunking, deduplication, and context generation.
All database operations now use Qdrant.
"""

from .box_client import BoxClient, BoxDocument
from .pdf_extractor import PDFExtractor, ExtractedDocument
from .chunker import DocumentChunker, DocumentChunk
from .qdrant_deduplicator import QdrantDocumentDeduplicator, DocumentRecord
from .context_generator import ContextGenerator, ChunkWithContext

# Backward compatibility alias
DocumentDeduplicator = QdrantDocumentDeduplicator

__all__ = [
    "BoxClient",
    "BoxDocument",
    "PDFExtractor", 
    "ExtractedDocument",
    "DocumentChunker",
    "DocumentChunk",
    "QdrantDocumentDeduplicator",
    "DocumentDeduplicator",  # Backward compatibility
    "DocumentRecord",
    "ContextGenerator",
    "ChunkWithContext"
]

__version__ = "0.2.0"  # Version bump for Qdrant-only implementation