#!/usr/bin/env python
"""
Updated example usage of the Clerk legal AI system with Qdrant vector database.
This version automatically detects available case names instead of using hardcoded values.
"""

import os
import sys
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_injector import DocumentInjector
from src.vector_storage import QdrantVectorStore, SparseVectorEncoder, LegalQueryAnalyzer
from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator


def get_available_cases():
    """Get all available case names from Qdrant"""
    deduplicator = QdrantDocumentDeduplicator()
    stats = deduplicator.get_statistics()
    return stats.get("cases", [])


def demonstrate_hybrid_search(case_name: str = None):
    """Demonstrate Qdrant's hybrid search capabilities"""
    print("\n" + "="*80)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*80)
    
    # Initialize components
    vector_store = QdrantVectorStore()
    sparse_encoder = SparseVectorEncoder()
    query_analyzer = LegalQueryAnalyzer(sparse_encoder)
    injector = DocumentInjector()
    
    # Get available cases if not specified
    if not case_name:
        available_cases = get_available_cases()
        
        if not available_cases:
            print("\n❌ No cases found in the system!")
            print("Please process some documents first:")
            print("  python -m src.document_injector --folder-id YOUR_FOLDER_ID")
            return
        
        print(f"\nAvailable cases in the system:")
        for i, case in enumerate(available_cases):
            print(f"  {i+1}. {case}")
        
        # Use the first available case
        case_name = available_cases[0]
        print(f"\nUsing case: '{case_name}'")
    else:
        print(f"\nUsing specified case: '{case_name}'")
    
    # Verify case exists
    case_stats = vector_store.get_case_statistics(case_name)
    if case_stats['total_chunks'] == 0:
        print(f"\n❌ No documents found for case '{case_name}'")
        return
    
    print(f"Case has {case_stats['total_chunks']} chunks from {case_stats['unique_documents']} documents")
    
    # Example queries demonstrating different search types
    test_queries = [
        {
            "query": "What are the key allegations in the complaint?",
            "type": "semantic"
        },
        {
            "query": "What motions have been filed in this case?",
            "type": "semantic"
        },
        {
            "query": "What are the damages claimed?",
            "type": "monetary_search"
        },
        {
            "query": "What legal citations are mentioned?",
            "type": "citation_search"
        }
    ]
    
    for test in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {test['query']}")
        print(f"Expected type: {test['type']}")
        
        # Analyze query
        analysis = query_analyzer.analyze_query(test['query'])
        print(f"Detected type: {analysis['query_type']}")
        print(f"Recommended weights: {analysis['recommended_weights']}")
        
        # Extract entities
        entities = analysis['entities']
        if entities['citations']:
            print(f"Citations found: {entities['citations']}")
        if entities['dates']:
            print(f"Dates found: {entities['dates']}")
        if entities['monetary']:
            print(f"Monetary amounts: {entities['monetary']}")
        
        # Perform search
        print("\nSearching...")
        start_time = time.time()
        
        try:
            results = injector.search_case(
                case_name=case_name,
                query=test['query'],
                limit=5,
                use_hybrid=True
            )
            
            search_time = time.time() - start_time
            
            print(f"Found {len(results)} results in {search_time:.3f} seconds")
            
            # Display top results
            for i, result in enumerate(results[:3], 1):
                print(f"\n  Result {i} (Score: {result.score:.3f}):")
                print(f"  Document: {result.metadata.get('document_name', 'Unknown')}")
                print(f"  Content: {result.content[:200]}...")
                
                # Show metadata
                if result.metadata.get('has_citations'):
                    print(f"  Has citations: Yes ({result.metadata.get('citation_count', 0)} found)")
                if result.metadata.get('chunk_index') is not None:
                    print(f"  Chunk: {result.metadata.get('chunk_index')} of document")
                    
        except Exception as e:
            print(f"Error during search: {str(e)}")


def demonstrate_case_isolation():
    """Demonstrate strict case isolation in Qdrant"""
    print("\n" + "="*80)
    print("CASE ISOLATION VERIFICATION")
    print("="*80)
    
    vector_store = QdrantVectorStore()
    
    # Get all available cases
    available_cases = get_available_cases()
    
    if len(available_cases) < 2:
        print("\nNeed at least 2 cases to demonstrate isolation.")
        print("Currently available cases:", available_cases)
        return
    
    print(f"\nTesting isolation between cases:")
    print(f"  Case 1: {available_cases[0]}")
    print(f"  Case 2: {available_cases[1] if len(available_cases) > 1 else 'N/A'}")
    
    # Generate test embedding
    from src.vector_storage.embeddings import EmbeddingGenerator
    embed_gen = EmbeddingGenerator()
    test_embedding, _ = embed_gen.generate_embedding("test query for case isolation")
    
    # Search in Case 1
    results_case1 = vector_store.search_case_documents(
        case_name=available_cases[0],
        query_embedding=test_embedding,
        limit=5
    )
    
    # Try to search with wrong case name
    results_wrong = vector_store.search_case_documents(
        case_name="DEFINITELY_WRONG_CASE_NAME",
        query_embedding=test_embedding,
        limit=5
    )
    
    print(f"\nResults for '{available_cases[0]}': {len(results_case1)} documents")
    print(f"Results for wrong case name: {len(results_wrong)} documents")
    
    if len(results_wrong) == 0:
        print("✓ Case isolation verified - no cross-case contamination")
    else:
        print("✗ WARNING: Case isolation may be compromised!")


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(
        description="Demonstrate Qdrant vector search for legal documents"
    )
    parser.add_argument(
        "--case-name",
        type=str,
        help="Specific case name to search (otherwise uses first available)"
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all available cases and exit"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\nCLERK LEGAL AI - QDRANT VECTOR DATABASE DEMONSTRATION")
    print("="*80)
    
    # List cases if requested
    if args.list_cases:
        cases = get_available_cases()
        print("\nAvailable cases:")
        for case in cases:
            print(f"  - {case}")
        return
    
    # Run demonstrations
    demonstrate_hybrid_search(args.case_name)
    demonstrate_case_isolation()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()