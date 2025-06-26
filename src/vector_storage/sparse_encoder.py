"""
Sparse vector encoder for hybrid search.
Creates sparse representations for keyword and citation matching.
Using hash-based consistent indexing.
"""

import re
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en import English

from config.settings import settings

logger = logging.getLogger(__name__)


class SparseVectorEncoder:
    """Encodes text into sparse vectors for keyword and citation matching"""
    
    def __init__(self):
        """Initialize the sparse encoder with legal-specific tokenization"""
        # Initialize spaCy for better tokenization
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except:
            logger.warning("spaCy model not found, using basic tokenizer")
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
        
        # Legal-specific patterns
        self.citation_pattern = re.compile(
            r'(?:'
            r'\d+\s+U\.S\.C\.\s+§\s*\d+|'  # USC citations
            r'\d+\s+F\.\d+d\s+\d+|'  # Federal Reporter
            r'\d+\s+S\.\s*Ct\.\s+\d+|'  # Supreme Court Reporter
            r'§\s*\d+(?:\.\d+)?|'  # Section references
            r'Rule\s+\d+(?:\.\d+)?|'  # Rule references
            r'\d+\s+[A-Z][a-z]+(?:\.\s*\d+d)?\s+\d+'  # State reporters
            r')'
        )
        
        self.monetary_pattern = re.compile(
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|k|m|b))?'
        )
        
        self.date_pattern = re.compile(
            r'(?:'
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|'
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}'
            r')'
        )
        
        # Legal stopwords (common words to exclude)
        self.legal_stopwords = set([
            "plaintiff", "defendant", "court", "case", "matter",
            "pursuant", "whereas", "therefore", "hereby", "herein",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "must", "can", "could", "chunk"
        ])
        
        # Initialize TF-IDF for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            use_idf=True,
            smooth_idf=True
        )
        
        # Use fixed dimension for sparse vectors
        self.sparse_dimension = 100000  # Large enough to avoid collisions
    
    def _hash_token_to_index(self, token: str, max_dim: int = None) -> int:
        """
        Hash a token to a consistent index using SHA-256.
        This ensures the same token always maps to the same index.
        
        Args:
            token: Token to hash
            max_dim: Maximum dimension for the sparse vector
            
        Returns:
            Integer index between 0 and max_dim-1
        """
        if max_dim is None:
            max_dim = self.sparse_dimension
            
        # Use SHA-256 for consistent hashing
        hash_object = hashlib.sha256(token.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        # Convert hex to integer and modulo by max dimension
        hash_int = int(hash_hex, 16)
        index = hash_int % max_dim
        
        return index
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text"""
        entities = {
            "citations": [],
            "monetary": [],
            "dates": [],
            "sections": [],
            "rules": []
        }
        
        # Extract citations
        citations = self.citation_pattern.findall(text)
        entities["citations"] = list(set(citations))
        
        # Extract monetary amounts
        monetary = self.monetary_pattern.findall(text)
        entities["monetary"] = list(set(monetary))
        
        # Extract dates
        dates = self.date_pattern.findall(text)
        entities["dates"] = list(set(dates))
        
        # Extract section references
        sections = re.findall(r'§\s*\d+(?:\.\d+)?', text)
        entities["sections"] = list(set(sections))
        
        # Extract rule references
        rules = re.findall(r'Rule\s+\d+(?:\.\d+)?', text, re.IGNORECASE)
        entities["rules"] = list(set(rules))
        
        return entities
    
    def tokenize_legal_text(self, text: str) -> List[str]:
        """Tokenize text with legal-specific handling"""
        # Process with spaCy
        doc = self.nlp(text.lower())
        
        tokens = []
        for token in doc:
            # Skip punctuation and spaces
            if token.is_punct or token.is_space:
                continue
            
            # Skip stopwords
            if token.text in self.legal_stopwords or len(token.text) < 2:
                continue
            
            # Keep legal terms intact
            if token.text.startswith("§") or token.text in ["u.s.c.", "f.2d", "f.3d"]:
                tokens.append(token.text)
            else:
                # Use lemma for consistency
                tokens.append(token.lemma_)
        
        return tokens
    
    def build_keyword_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        Build sparse vector for keyword matching using hash-based indexing.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping integer indices to float values
        """
        # Tokenize
        tokens = self.tokenize_legal_text(text)
        
        if not tokens:
            logger.warning("No tokens extracted from text")
            return {}
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Build sparse vector using hash-based indices
        sparse_vector = {}
        
        for token, count in token_counts.items():
            # Get consistent index using hash
            idx = self._hash_token_to_index(token)
            
            # TF-IDF style weighting
            # Using log normalization for term frequency
            tf = 1 + np.log(count)
            
            # Store in sparse vector
            sparse_vector[idx] = tf
        
        # L2 normalization
        if sparse_vector:
            norm = np.sqrt(sum(v**2 for v in sparse_vector.values()))
            if norm > 0:
                sparse_vector = {k: v/norm for k, v in sparse_vector.items()}
            else:
                logger.warning("Zero norm encountered in sparse vector")
        
        logger.debug(f"Created keyword sparse vector with {len(sparse_vector)} non-zero elements")
        
        return sparse_vector

    def build_citation_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        Build sparse vector for citation matching using hash-based indexing.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping integer indices to float values
        """
        # Extract legal entities
        entities = self.extract_legal_entities(text)
        
        # Combine all legal references
        all_citations = (
            entities["citations"] + 
            entities["sections"] + 
            entities["rules"]
        )
        
        if not all_citations:
            logger.debug("No citations found in text")
            return {}
        
        # Build sparse vector using hash-based indices
        sparse_vector = {}
        
        # Use a different prefix for citation indices to avoid collision with keywords
        citation_prefix = "CITATION_"
        
        for citation in all_citations:
            # Normalize citation
            normalized = citation.lower().strip()
            
            # Add prefix to distinguish from keywords
            prefixed_citation = f"{citation_prefix}{normalized}"
            
            # Get consistent index using hash
            idx = self._hash_token_to_index(prefixed_citation)
            
            # Use binary weighting for citations (present or not)
            sparse_vector[idx] = 1.0
        
        # Normalize
        if sparse_vector:
            # For citations, we can use L2 norm or keep binary
            norm = np.sqrt(sum(v**2 for v in sparse_vector.values()))
            if norm > 0:
                sparse_vector = {k: v/norm for k, v in sparse_vector.items()}
        
        logger.debug(f"Created citation sparse vector with {len(sparse_vector)} citations")
        
        return sparse_vector

    def encode_for_hybrid_search(self, text: str) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Encode text for both keyword and citation sparse vectors with hash-based indexing.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (keyword_sparse, citation_sparse) dictionaries with integer indices
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sparse encoding")
            return {}, {}
        
        keyword_sparse = self.build_keyword_sparse_vector(text)
        citation_sparse = self.build_citation_sparse_vector(text)
        
        logger.debug(f"Encoded text to sparse vectors: keywords={len(keyword_sparse)}, citations={len(citation_sparse)}")
        
        return keyword_sparse, citation_sparse
    
    def prepare_search_text(self, text: str) -> str:
        """Prepare text for full-text search with legal-specific preprocessing"""
        # Extract entities
        entities = self.extract_legal_entities(text)
        
        # Tokenize
        tokens = self.tokenize_legal_text(text)
        
        # Add extracted entities as additional tokens
        additional_tokens = []
        
        # Add normalized citations
        for citation in entities["citations"]:
            normalized = citation.replace(".", "").replace(" ", "_").lower()
            additional_tokens.append(f"CITE_{normalized}")
        
        # Add normalized monetary amounts
        for amount in entities["monetary"]:
            normalized = amount.replace("$", "").replace(",", "")
            additional_tokens.append(f"MONEY_{normalized}")
        
        # Add normalized dates
        for date in entities["dates"]:
            normalized = date.replace("/", "-").replace(" ", "_")
            additional_tokens.append(f"DATE_{normalized}")
        
        # Combine all tokens
        all_tokens = tokens + additional_tokens
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in all_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        return " ".join(unique_tokens)
    
    def extract_important_terms(self, text: str, max_terms: int = 10) -> List[Tuple[str, float]]:
        """Extract important terms from text using TF-IDF"""
        try:
            # Tokenize first
            tokens = self.tokenize_legal_text(text)
            if not tokens:
                return []
            
            # Join tokens back for TF-IDF
            processed_text = " ".join(tokens)
            
            # Fit TF-IDF on single document
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = scores.argsort()[-max_terms:][::-1]
            top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return top_terms
            
        except Exception as e:
            logger.error(f"Error extracting important terms: {str(e)}")
            return []
    
    def calculate_bm25_score(self, query_terms: List[str], doc_terms: List[str],
                           avg_doc_length: float = 1000, k1: float = 1.2, b: float = 0.75) -> float:
        """Calculate BM25 score for ranking"""
        doc_length = len(doc_terms)
        doc_term_counts = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                # Simplified IDF (would need corpus statistics in production)
                idf = np.log(1000 / (1 + tf))  # Assume 1000 docs
                
                # BM25 formula
                numerator = idf * tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                score += numerator / denominator
        
        return score


class LegalQueryAnalyzer:
    """Analyzes legal queries to optimize search strategy"""
    
    def __init__(self, sparse_encoder: SparseVectorEncoder):
        """Initialize with sparse encoder"""
        self.sparse_encoder = sparse_encoder
        
        # Query type patterns
        self.query_patterns = {
            "citation_search": re.compile(r'(?:cite|citation|§|\d+\s+U\.S\.C\.|Rule\s+\d+)', re.I),
            "date_search": re.compile(r'(?:on|before|after|between|during)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}[-/]\d{1,2})', re.I),
            "monetary_search": re.compile(r'(?:damages|settlement|amount|cost|fee|award).*\$[\d,]+', re.I),
            "definition_search": re.compile(r'(?:what is|define|definition of|meaning of)', re.I),
            "precedent_search": re.compile(r'(?:precedent|case law|prior cases|similar cases)', re.I),
            "procedure_search": re.compile(r'(?:how to|procedure|process|steps|filing)', re.I)
        }
    
    def analyze_query(self, query: str) -> Dict[str, any]:
        """Analyze query to determine optimal search strategy"""
        analysis = {
            "query_type": "general",
            "entities": self.sparse_encoder.extract_legal_entities(query),
            "important_terms": self.sparse_encoder.extract_important_terms(query, 5),
            "recommended_weights": {
                "vector": 0.7,
                "keyword": 0.2,
                "citation": 0.1
            },
            "filters": {}
        }
        
        # Determine query type
        for query_type, pattern in self.query_patterns.items():
            if pattern.search(query):
                analysis["query_type"] = query_type
                break
        
        # Adjust weights based on query type
        if analysis["query_type"] == "citation_search":
            analysis["recommended_weights"] = {
                "vector": 0.3,
                "keyword": 0.2,
                "citation": 0.5
            }
        elif analysis["query_type"] == "date_search":
            analysis["recommended_weights"] = {
                "vector": 0.5,
                "keyword": 0.4,
                "citation": 0.1
            }
            # Extract date for filtering
            dates = analysis["entities"]["dates"]
            if dates:
                analysis["filters"]["date_range"] = dates[0]
        elif analysis["query_type"] == "monetary_search":
            analysis["recommended_weights"] = {
                "vector": 0.5,
                "keyword": 0.4,
                "citation": 0.1
            }
        
        return analysis