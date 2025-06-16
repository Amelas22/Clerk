"""
Document chunking module.
Splits documents into chunks with overlap while preserving semantic boundaries.
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict

class DocumentChunker:
    """Splits documents into overlapping chunks with semantic awareness"""
    
    def __init__(self):
        """Initialize chunker with settings"""
        self.target_size = settings.chunking.target_chunk_size
        self.variance = settings.chunking.chunk_variance
        self.overlap = settings.chunking.overlap_size
        self.min_size = settings.chunking.min_chunk_size
        self.max_size = settings.chunking.max_chunk_size
        
        # Compile regex patterns for efficiency
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.word_pattern = re.compile(r'\s+')
    
    def chunk_document(self, text: str, doc_metadata: dict = None) -> List[DocumentChunk]:
        """Split document into chunks with overlap
        
        Args:
            text: Full document text
            doc_metadata: Metadata to attach to each chunk
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or len(text.strip()) < self.min_size:
            logger.warning("Document too short to chunk")
            return []
        
        # Clean text
        text = self._clean_text(text)
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # Find chunk boundaries
            chunk_start = max(0, current_pos - self.overlap if chunk_index > 0 else 0)
            chunk_end = self._find_chunk_end(text, chunk_start)
            
            # Extract chunk content
            chunk_content = text[chunk_start:chunk_end].strip()
            
            if len(chunk_content) >= self.min_size:
                chunk = DocumentChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata={
                        **(doc_metadata or {}),
                        "chunk_index": chunk_index,
                        "total_chunks": -1,  # Will be updated after all chunks created
                        "has_overlap": chunk_index > 0
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move position for next chunk
            current_pos = chunk_end
            
            # Break if we're at the end
            if current_pos >= len(text):
                break
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better chunking
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page markers for now (we'll handle them separately)
        text = re.sub(r'\[Page \d+\]', '', text)
        
        return text.strip()
    
    def _find_chunk_end(self, text: str, start_pos: int) -> int:
        """Find optimal end position for chunk
        
        Args:
            text: Full document text
            start_pos: Starting position for chunk
            
        Returns:
            End position for chunk
        """
        # Calculate target end position
        target_end = start_pos + self.target_size
        min_end = start_pos + self.target_size - self.variance
        max_end = min(start_pos + self.target_size + self.variance, len(text))
        
        # If we're near the end of document, just take the rest
        if max_end >= len(text) - self.min_size:
            return len(text)
        
        # Try to find paragraph boundary first
        paragraph_end = self._find_boundary(
            text, target_end, min_end, max_end, self.paragraph_pattern
        )
        if paragraph_end:
            return paragraph_end
        
        # Try sentence boundary
        sentence_end = self._find_boundary(
            text, target_end, min_end, max_end, self.sentence_pattern
        )
        if sentence_end:
            return sentence_end
        
        # Try word boundary
        word_end = self._find_boundary(
            text, target_end, min_end, max_end, self.word_pattern
        )
        if word_end:
            return word_end
        
        # Last resort: use target position
        return min(target_end, len(text))
    
    def _find_boundary(self, text: str, target: int, min_pos: int, 
                      max_pos: int, pattern: re.Pattern) -> Optional[int]:
        """Find boundary matching pattern within range
        
        Args:
            text: Full text
            target: Target position
            min_pos: Minimum acceptable position
            max_pos: Maximum acceptable position
            pattern: Regex pattern to match
            
        Returns:
            Position after boundary or None
        """
        # Search forward from target
        search_text = text[target:max_pos]
        match = pattern.search(search_text)
        if match:
            return target + match.end()
        
        # Search backward from target
        search_text = text[min_pos:target]
        matches = list(pattern.finditer(search_text))
        if matches:
            last_match = matches[-1]
            return min_pos + last_match.end()
        
        return None
    
    def create_overlap_chunk(self, previous_chunk: str, current_chunk: str) -> str:
        """Create chunk with overlap from previous chunk
        
        Args:
            previous_chunk: Content of previous chunk
            current_chunk: Content of current chunk
            
        Returns:
            Combined chunk with overlap
        """
        if not previous_chunk:
            return current_chunk
        
        # Take last portion of previous chunk
        overlap_start = max(0, len(previous_chunk) - self.overlap)
        overlap_text = previous_chunk[overlap_start:]
        
        # Find a good boundary in the overlap
        sentence_match = self.sentence_pattern.search(overlap_text)
        if sentence_match:
            overlap_text = overlap_text[sentence_match.end():]
        
        return overlap_text + current_chunk