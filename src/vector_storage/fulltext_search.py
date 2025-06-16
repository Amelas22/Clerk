"""
Full-text search module for hybrid search capabilities.
Prepares text for PostgreSQL full-text search and manages search operations.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from supabase import create_client, Client

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from hybrid search"""
    id: str
    content: str
    case_name: str
    document_id: str
    vector_similarity: Optional[float] = None
    text_rank: Optional[float] = None
    combined_score: Optional[float] = None
    metadata: Dict[str, Any] = None

class FullTextSearchManager:
    """Manages full-text search functionality for hybrid search"""
    
    def __init__(self):
        """Initialize full-text search manager"""
        self.supabase: Client = create_client(
            settings.database.supabase_url,
            settings.database.supabase_service_key
        )
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> set:
        """Load common legal stop words to exclude from search"""
        # Common legal and general stop words
        return {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'shall', 'can', 'a', 'an', 'this', 'that', 'these',
            'those', 'such', 'it', 'its', 'they', 'their', 'them',
            'pursuant', 'whereas', 'hereby', 'herein', 'thereof', 'thereto'
        }
    
    def prepare_text_for_search(self, text: str) -> str:
        """Prepare text for PostgreSQL full-text search
        
        Args:
            text: Raw text content
            
        Returns:
            Processed text optimized for search
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important legal punctuation
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)§]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Extract important terms (numbers, section references, etc.)
        important_terms = self._extract_important_terms(text)
        
        # Remove stop words for main content
        words = text.split()
        filtered_words = [w for w in words if w not in self.stop_words or w in important_terms]
        
        # Rejoin filtered text
        processed_text = ' '.join(filtered_words)
        
        # Add important terms at the end to boost their weight
        if important_terms:
            processed_text += ' ' + ' '.join(important_terms)
        
        return processed_text
    
    def _extract_important_terms(self, text: str) -> set:
        """Extract important legal terms that should be preserved
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of important terms
        """
        important = set()
        
        # Legal section references (e.g., "§ 1234", "Section 5.2")
        section_patterns = [
            r'§\s*\d+(?:\.\d+)?',
            r'section\s+\d+(?:\.\d+)?',
            r'article\s+[IVX]+',
            r'article\s+\d+',
            r'rule\s+\d+(?:\.\d+)?'
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                important.add(match.group().lower())
        
        # Case citations (simplified pattern)
        case_pattern = r'\d+\s+\w+\s+\d+'
        case_matches = re.finditer(case_pattern, text)
        for match in case_matches:
            important.add(match.group().lower())
        
        # Dollar amounts
        dollar_pattern = r'\$[\d,]+(?:\.\d{2})?'
        dollar_matches = re.finditer(dollar_pattern, text)
        for match in dollar_matches:
            important.add(match.group())
        
        # Dates in various formats
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                important.add(match.group().lower())
        
        # Medical terms (if medical record)
        if any(term in text.lower() for term in ['diagnosis', 'patient', 'treatment', 'medical']):
            medical_pattern = r'\b(?:diagnosis|prognosis|symptoms?|treatments?|medications?|procedures?)\b'
            medical_matches = re.finditer(medical_pattern, text, re.IGNORECASE)
            for match in medical_matches:
                important.add(match.group().lower())
        
        return important
    
    def create_search_vector(self, text: str) -> str:
        """Create a tsvector for PostgreSQL from text
        
        Args:
            text: Processed text
            
        Returns:
            Text ready for to_tsvector conversion
        """
        # PostgreSQL will handle the actual conversion to tsvector
        # We just prepare the text properly
        return self.prepare_text_for_search(text)
    
    def hybrid_search(self, case_name: str, query_text: str, 
                     query_embedding: List[float], limit: int = 20,
                     vector_weight: float = 0.7, text_weight: float = 0.3) -> List[SearchResult]:
        """Perform hybrid search combining vector and full-text search
        
        Args:
            case_name: Case to search within (CRITICAL for isolation)
            query_text: Text query for full-text search
            query_embedding: Vector embedding for similarity search
            limit: Maximum results to return
            vector_weight: Weight for vector similarity (0-1)
            text_weight: Weight for text search (0-1)
            
        Returns:
            List of search results with combined scoring
        """
        try:
            # Prepare query for full-text search
            processed_query = self.prepare_text_for_search(query_text)
            
            # Call hybrid search RPC function
            response = self.supabase.rpc(
                "hybrid_search_case_documents",
                {
                    "case_name_filter": case_name,
                    "text_query": processed_query,
                    "embedding_query": query_embedding,
                    "match_count": limit,
                    "vector_weight": vector_weight,
                    "text_weight": text_weight
                }
            ).execute()
            
            results = []
            for item in response.data:
                # CRITICAL: Verify case isolation
                if item.get("case_name") != case_name:
                    logger.error(
                        f"CRITICAL: Case isolation breach in hybrid search! "
                        f"Expected '{case_name}', got '{item.get('case_name')}'"
                    )
                    continue
                
                result = SearchResult(
                    id=item["id"],
                    content=item["content"],
                    case_name=item["case_name"],
                    document_id=item["document_id"],
                    vector_similarity=item.get("vector_similarity"),
                    text_rank=item.get("text_rank"),
                    combined_score=item.get("combined_score"),
                    metadata=item.get("metadata", {})
                )
                results.append(result)
            
            logger.info(f"Hybrid search found {len(results)} results for case '{case_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    def update_search_index(self, case_name: str, document_id: str):
        """Update search index for a specific document
        
        Args:
            case_name: Case name for isolation
            document_id: Document to reindex
        """
        try:
            # Call reindex RPC function
            self.supabase.rpc(
                "update_document_search_index",
                {
                    "case_name_filter": case_name,
                    "document_id_filter": document_id
                }
            ).execute()
            
            logger.info(f"Updated search index for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error updating search index: {str(e)}")
            raise
    
    def analyze_search_query(self, query: str) -> Dict[str, Any]:
        """Analyze a search query to extract key components
        
        Args:
            query: User's search query
            
        Returns:
            Dictionary with query analysis
        """
        analysis = {
            "original_query": query,
            "processed_query": self.prepare_text_for_search(query),
            "important_terms": list(self._extract_important_terms(query)),
            "has_legal_citations": bool(re.search(r'§\s*\d+|\d+\s+\w+\s+\d+', query)),
            "has_dates": bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', query)),
            "has_monetary_amounts": bool(re.search(r'\$[\d,]+', query)),
            "query_type": self._determine_query_type(query)
        }
        
        return analysis
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of search query
        
        Args:
            query: User's search query
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['motion', 'brief', 'complaint', 'answer']):
            return "legal_document"
        elif any(word in query_lower for word in ['diagnosis', 'treatment', 'medical', 'patient']):
            return "medical_record"
        elif any(word in query_lower for word in ['deposition', 'testimony', 'witness']):
            return "testimony"
        elif any(word in query_lower for word in ['expert', 'opinion', 'report']):
            return "expert_report"
        elif re.search(r'\$[\d,]+|\d+\s*dollars?', query_lower):
            return "financial"
        elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december', query_lower):
            return "temporal"
        else:
            return "general"