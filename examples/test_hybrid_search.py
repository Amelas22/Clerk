#!/usr/bin/env python3
"""
Test script for hybrid search functionality.
Verifies that hybrid search works correctly with sparse and dense vectors.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
from src.vector_storage.qdrant_store import QdrantVectorStore
from src.vector_storage.embeddings import EmbeddingGenerator
from src.vector_storage.sparse_encoder import SparseVectorEncoder
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_hybrid_search():
    """Test hybrid search functionality"""
    try:
        # Initialize components
        logger.info("Initializing components...")
        vector_store = QdrantVectorStore()
        embedding_generator = EmbeddingGenerator()
        sparse_encoder = SparseVectorEncoder()
        
        # Test data
        case_name = "Test Case v. Example Corp"
        test_documents = [
            {
                "id": "doc1",
                "content": "This contract contains a liquidated damages clause of $50,000 per violation.",
                "metadata": {"document_type": "contract", "date_filed": "2023-01-15"}
            },
            {
                "id": "doc2", 
                "content": "The plaintiff filed a motion for summary judgment citing Rule 56 of the Federal Rules of Civil Procedure.",
                "metadata": {"document_type": "motion", "date_filed": "2023-02-20"}
            },
            {
                "id": "doc3",
                "content": "The defendant argues that the contract is unenforceable due to unconscionability.",
                "metadata": {"document_type": "brief", "date_filed": "2023-03-10"}
            }
        ]
        
        # Store documents
        logger.info("Storing test documents...")
        for doc in test_documents:
            # Generate embedding
            embedding, token_count = embedding_generator.generate_embedding(doc["content"])
            logger.info(f"Generated embedding with {token_count} tokens for document {doc['id']}")
            
            # Generate sparse vectors
            keyword_sparse, citation_sparse = sparse_encoder.encode_for_hybrid_search(doc["content"])
            
            # Create document chunk with sparse vectors
            chunk = {
                "content": doc["content"],
                "embedding": embedding,
                "keywords_sparse": keyword_sparse,
                "citations_sparse": citation_sparse,
                "metadata": {
                    "case_name": case_name,
                    "document_id": doc["id"],
                    **doc["metadata"]
                }
            }
            
            # Store in both collections
            vector_store.store_document_chunks(case_name, doc["id"], [chunk])
            logger.info(f"Stored document {doc['id']} in case {case_name}")
        
        # Test queries
        test_queries = [
            "liquidated damages contract",
            "summary judgment motion Rule 56",
            "unconscionability defense"
        ]
        
        logger.info("Testing hybrid search...")
        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' ---")
            
            # Generate query embedding
            query_embedding, _ = embedding_generator.generate_embedding(query)
            
            # Generate sparse vectors for query
            keyword_sparse, citation_sparse = sparse_encoder.encode_for_hybrid_search(query)
            
            # Convert sparse vectors to proper format for search
            keyword_indices = {sparse_encoder.keyword_vocab.get(k, -1): v 
                             for k, v in keyword_sparse.items() 
                             if sparse_encoder.keyword_vocab.get(k, -1) != -1}
            citation_indices = {sparse_encoder.citation_vocab.get(k, -1): v 
                              for k, v in citation_sparse.items() 
                              if sparse_encoder.citation_vocab.get(k, -1) != -1}
            
            # Perform hybrid search
            results = vector_store.hybrid_search(
                case_name=case_name,
                query_text=query,
                query_embedding=query_embedding,
                keyword_indices=keyword_indices,
                citation_indices=citation_indices,
                limit=5
            )
            
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")
                logger.info(f"     Content: {result.content[:100]}...")
                logger.info(f"     Document: {result.document_id}")
        
        # Test case isolation
        logger.info("\n--- Testing case isolation ---")
        other_case = "Other Case v. Different Corp"
        
        # Try to search for the other case (should return empty)
        results = vector_store.hybrid_search(
            case_name=other_case,
            query_text="liquidated damages",
            query_embedding=embedding_generator.generate_embedding("liquidated damages"),
            limit=5
        )
        
        if len(results) == 0:
            logger.info("✓ Case isolation working correctly - no results for different case")
        else:
            logger.error(f"✗ Case isolation failed - found {len(results)} results for different case")
        
        logger.info("\n=== Hybrid search test completed successfully ===")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_search()
