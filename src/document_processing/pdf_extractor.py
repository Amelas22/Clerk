"""
PDF text extraction module.
Handles extraction of text from PDF documents with error handling.
"""

import io
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract

logger = logging.getLogger(__name__)

@dataclass
class ExtractedDocument:
    """Represents extracted text from a PDF"""
    text: str
    page_count: int
    extraction_method: str
    metadata: Dict[str, Any]

class PDFExtractor:
    """Extracts text from PDF documents using multiple methods"""
    
    def __init__(self):
        """Initialize PDF extractor"""
        self.extraction_methods = [
            ("pdfplumber", self._extract_with_pdfplumber),
            ("PyPDF2", self._extract_with_pypdf2),
            ("pdfminer", self._extract_with_pdfminer),
        ]
    
    def extract_text(self, pdf_content: bytes, filename: str = "unknown.pdf") -> ExtractedDocument:
        """Extract text from PDF content
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Name of the file for logging
            
        Returns:
            ExtractedDocument with extracted text and metadata
        """
        logger.info(f"Extracting text from {filename}")
        
        # Try each extraction method until one succeeds
        for method_name, method_func in self.extraction_methods:
            try:
                result = method_func(pdf_content)
                if result and result.text.strip():
                    logger.info(f"Successfully extracted text using {method_name}")
                    result.extraction_method = method_name
                    return result
            except Exception as e:
                logger.warning(f"Failed to extract with {method_name}: {str(e)}")
                continue
        
        # If all methods fail, return empty result
        logger.error(f"All extraction methods failed for {filename}")
        return ExtractedDocument(
            text="",
            page_count=0,
            extraction_method="failed",
            metadata={"error": "All extraction methods failed"}
        )
    
    def _extract_with_pdfplumber(self, pdf_content: bytes) -> ExtractedDocument:
        """Extract text using pdfplumber (best for tables and complex layouts)"""
        text_parts = []
        page_count = 0
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Add page marker for better chunking later
                        text_parts.append(f"\n[Page {page_num}]\n{page_text}")
                except Exception as e:
                    logger.debug(f"Error extracting page {page_num}: {str(e)}")
                    continue
        
        return ExtractedDocument(
            text="".join(text_parts),
            page_count=page_count,
            extraction_method="pdfplumber",
            metadata={"has_page_markers": True}
        )
    
    def _extract_with_pypdf2(self, pdf_content: bytes) -> ExtractedDocument:
        """Extract text using PyPDF2 (fast but less accurate)"""
        text_parts = []
        page_count = 0
        
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_count = len(pdf_reader.pages)
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"\n[Page {page_num}]\n{page_text}")
            except Exception as e:
                logger.debug(f"Error extracting page {page_num}: {str(e)}")
                continue
        
        return ExtractedDocument(
            text="".join(text_parts),
            page_count=page_count,
            extraction_method="PyPDF2",
            metadata={"has_page_markers": True}
        )
    
    def _extract_with_pdfminer(self, pdf_content: bytes) -> ExtractedDocument:
        """Extract text using pdfminer (good for complex PDFs)"""
        try:
            text = pdfminer_extract(io.BytesIO(pdf_content))
            
            # Try to count pages (approximate)
            page_count = text.count('\f') + 1  # Form feed character
            
            return ExtractedDocument(
                text=text,
                page_count=page_count,
                extraction_method="pdfminer",
                metadata={"has_page_markers": False}
            )
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {str(e)}")
            raise
    
    def validate_extraction(self, extracted: ExtractedDocument, min_chars: int = 100) -> bool:
        """Validate that extraction was successful
        
        Args:
            extracted: ExtractedDocument to validate
            min_chars: Minimum number of characters expected
            
        Returns:
            True if extraction appears valid
        """
        if not extracted.text:
            return False
            
        # Remove whitespace and check length
        cleaned_text = " ".join(extracted.text.split())
        
        if len(cleaned_text) < min_chars:
            logger.warning(f"Extracted text too short: {len(cleaned_text)} chars")
            return False
            
        # Check for common extraction failures
        if cleaned_text.count("�") > len(cleaned_text) * 0.1:  # >10% replacement chars
            logger.warning("High number of replacement characters detected")
            return False
            
        return True