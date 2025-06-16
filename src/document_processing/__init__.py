"""
Document processing package for Clerk legal AI system.
Handles PDF extraction, chunking, deduplication, and context generation.
"""

from .box_client import BoxClient, BoxDocument
from .pdf_extractor import PDFExtractor, ExtractedDocument
from .chunker import DocumentChunker, DocumentChunk
from .deduplicator import DocumentDeduplicator
from .context_generator import ContextGenerator, ChunkWithContext

__all__ = [
    "BoxClient",
    "BoxDocument",
    "PDFExtractor", 
    "ExtractedDocument",
    "DocumentChunker",
    "DocumentChunk",
    "DocumentDeduplicator",
    "ContextGenerator",
    "ChunkWithContext"
]

__version__ = "0.1.0"