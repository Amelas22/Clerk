#!/usr/bin/env python
"""
Quick test script to verify Qdrant search is working after fixes
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_injector import DocumentInjector
from src.vector_storage import QdrantVectorStore
from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator


def quick_search_test():
    """Run a quick search test on available cases"""
    
    print("\nQUICK QDRANT SEARCH TEST")
    print("="*60)
    
    # Get available cases
    deduplicator = QdrantDocumentDeduplicator()
    stats = deduplicator.get_statistics()
    available_cases = stats.get("cases", [])
    
    if not available_cases:
        print("❌ No cases found!")
        return
    
    print(f"\nAvailable cases: {available_cases[:5]}")  # Show first 5
    
    # Test with first case
    test_case = 'Cerrtio v Test'
    print(f"\nTesting with case: '{test_case}'")
    
    # Initialize injector
    injector = DocumentInjector(enable_cost_tracking=False)
    
    # Get case statistics
    vector_store = QdrantVectorStore()
    case_stats = vector_store.get_case_statistics(test_case)
    print(f"Case has {case_stats['total_chunks']} chunks")
    
    # Simple test queries
    test_queries = [
        "What are the main allegations?",
        "damages",
        "motion",
        "complaint"
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: '{query}'")
        
        try:
            results = injector.search_case(
                case_name=test_case,
                query=query,
                limit=3,
                use_hybrid=False  # Start with simple vector search
            )
            
            print(f"Found {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:2], 1):
                    print(f"\nResult {i} (Score: {result.score:.3f}):")
                    print(f"  Document: {result.metadata.get('document_name', 'Unknown')}")
                    print(f"  Content preview: {result.content[:150]}...")
            else:
                print("  ❌ No results found!")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    quick_search_test()