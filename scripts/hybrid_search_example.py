#!/usr/bin/env python
"""
Example script demonstrating hybrid search functionality.
Shows how to perform searches combining vector similarity and full-text search.
"""

import sys
import os
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_storage import FullTextSearchManager, EmbeddingGenerator
from src.vector_storage.vector_store import VectorStore

def demonstrate_search_types(case_name: str):
    """Demonstrate different types of searches"""
    
    # Initialize components
    search_manager = FullTextSearchManager()
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Example queries demonstrating different search scenarios
    queries = [
        {
            "query": "diagnosis of herniated disc lumbar spine",
            "description": "Medical terminology search",
            "vector_weight": 0.8,
            "text_weight": 0.2
        },
        {
            "query": "motion to dismiss filed on March 15, 2023",
            "description": "Date and document type search",
            "vector_weight": 0.5,
            "text_weight": 0.5
        },
        {
            "query": "$250,000 settlement offer",
            "description": "Monetary amount search",
            "vector_weight": 0.3,
            "text_weight": 0.7
        },
        {
            "query": "expert witness Dr. Johnson testimony regarding accident reconstruction",
            "description": "Named entity and topic search",
            "vector_weight": 0.7,
            "text_weight": 0.3
        },
        {
            "query": "§ 1983 civil rights violation claim",
            "description": "Legal citation search",
            "vector_weight": 0.4,
            "text_weight": 0.6
        }
    ]
    
    print(f"\nDemonstrating hybrid search for case: {case_name}")
    print("=" * 80)
    
    for query_info in queries:
        query = query_info["query"]
        print(f"\nQuery: {query}")
        print(f"Type: {query_info['description']}")
        print(f"Weights: Vector={query_info['vector_weight']}, Text={query_info['text_weight']}")
        print("-" * 60)
        
        # Analyze the query
        analysis = search_manager.analyze_search_query(query)
        print(f"Query type detected: {analysis['query_type']}")
        if analysis['important_terms']:
            print(f"Important terms: {', '.join(analysis['important_terms'])}")
        
        try:
            # Generate embedding for vector search
            query_embedding, _ = embedding_gen.generate_embedding(query)
            
            # Perform hybrid search
            results = search_manager.hybrid_search(
                case_name=case_name,
                query_text=query,
                query_embedding=query_embedding,
                limit=5,
                vector_weight=query_info['vector_weight'],
                text_weight=query_info['text_weight']
            )
            
            if results:
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results[:3], 1):  # Show top 3
                    print(f"\n{i}. Combined Score: {result.combined_score:.3f}")
                    print(f"   Vector Similarity: {result.vector_similarity:.3f}")
                    print(f"   Text Rank: {result.text_rank:.3f}")
                    print(f"   Document: {result.metadata.get('document_name', 'Unknown')}")
                    print(f"   Preview: {result.content[:150]}...")
            else:
                print("No results found.")
                
        except Exception as e:
            print(f"Error performing search: {str(e)}")
        
        print("\n" + "=" * 80)

def compare_search_methods(case_name: str, query: str):
    """Compare vector-only, text-only, and hybrid search results"""
    
    print(f"\nComparing search methods for query: '{query}'")
    print("=" * 80)
    
    # Initialize components
    search_manager = FullTextSearchManager()
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Generate embedding
    query_embedding, _ = embedding_gen.generate_embedding(query)
    
    # 1. Vector-only search
    print("\n1. VECTOR-ONLY SEARCH (Semantic similarity)")
    print("-" * 40)
    vector_results = vector_store.search_case_documents(
        case_name=case_name,
        query_embedding=query_embedding,
        limit=5
    )
    
    for i, result in enumerate(vector_results[:3], 1):
        print(f"{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
    
    # 2. Text-only search (using hybrid with 100% text weight)
    print("\n2. TEXT-ONLY SEARCH (Keyword matching)")
    print("-" * 40)
    text_results = search_manager.hybrid_search(
        case_name=case_name,
        query_text=query,
        query_embedding=query_embedding,
        limit=5,
        vector_weight=0.0,
        text_weight=1.0
    )
    
    for i, result in enumerate(text_results[:3], 1):
        print(f"{i}. Text Rank: {result.text_rank:.3f}")
        print(f"   Content: {result.content[:100]}...")
    
    # 3. Hybrid search
    print("\n3. HYBRID SEARCH (70% semantic + 30% keywords)")
    print("-" * 40)
    hybrid_results = search_manager.hybrid_search(
        case_name=case_name,
        query_text=query,
        query_embedding=query_embedding,
        limit=5,
        vector_weight=0.7,
        text_weight=0.3
    )
    
    for i, result in enumerate(hybrid_results[:3], 1):
        print(f"{i}. Combined Score: {result.combined_score:.3f}")
        print(f"   Vector: {result.vector_similarity:.3f}, Text: {result.text_rank:.3f}")
        print(f"   Content: {result.content[:100]}...")
    
    print("\n" + "=" * 80)

def main():
    """Main function to run search demonstrations"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate hybrid search capabilities"
    )
    parser.add_argument(
        "--case-name",
        required=True,
        help="Case name to search within"
    )
    parser.add_argument(
        "--demo-type",
        choices=["all", "compare", "examples"],
        default="all",
        help="Type of demonstration to run"
    )
    parser.add_argument(
        "--custom-query",
        help="Custom query for comparison demo"
    )
    
    args = parser.parse_args()
    
    if args.demo_type in ["all", "examples"]:
        demonstrate_search_types(args.case_name)
    
    if args.demo_type in ["all", "compare"]:
        query = args.custom_query or "medical malpractice negligence standard of care"
        compare_search_methods(args.case_name, query)

if __name__ == "__main__":
    main()