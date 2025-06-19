#!/usr/bin/env python
"""
Example usage of the Clerk legal AI system with Qdrant vector database.
Demonstrates:
- Document processing with Qdrant storage
- Hybrid search (vector + keyword + citation)
- Case isolation verification
- Performance comparison
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_injector import DocumentInjector
from src.vector_storage import QdrantVectorStore, SparseVectorEncoder, LegalQueryAnalyzer


def demonstrate_qdrant_features():
    """Demonstrate key Qdrant features for legal document processing"""
    print("\n" + "="*80)
    print("CLERK LEGAL AI - QDRANT VECTOR DATABASE DEMONSTRATION")
    print("="*80)
    
    # Initialize components
    print("\n1. Initializing Qdrant-powered document injector...")
    injector = DocumentInjector(enable_cost_tracking=True)
    
    # Check Qdrant connection
    try:
        collections = injector.vector_store.client.get_collections()
        print(f"✓ Connected to Qdrant")
        print(f"  Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"✗ Qdrant connection failed: {str(e)}")
        return
    
    # Process a sample folder
    print("\n2. Processing sample documents...")
    folder_id = "325242457476"  # Replace with your test folder
    
    start_time = time.time()
    results = injector.process_case_folder(folder_id, max_documents=3)
    processing_time = time.time() - start_time
    
    print(f"\nProcessing complete in {processing_time:.2f} seconds")
    for result in results:
        print(f"  - {result.file_name}: {result.status} ({result.chunks_created} chunks)")
    
    # Get case statistics
    if results and results[0].status == "success":
        case_name = results[0].case_name
        stats = injector.vector_store.get_case_statistics(case_name)
        print(f"\nCase statistics for '{case_name}':")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Unique documents: {stats['unique_documents']}")


def demonstrate_hybrid_search():
    """Demonstrate Qdrant's hybrid search capabilities"""
    print("\n" + "="*80)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*80)
    
    # Initialize components
    vector_store = QdrantVectorStore()
    sparse_encoder = SparseVectorEncoder()
    query_analyzer = LegalQueryAnalyzer(sparse_encoder)
    injector = DocumentInjector()
    
    # Test case name (use an actual case from your system)
    case_name = "Cerrito v Test"
    
    # Example queries demonstrating different search types
    test_queries = [
        {
            "query": "What is the purpose of the Motion about Irrelevant evidence?",
            "type": "semantic"
        },
        {
            "query": "Explain to me what was in the Motion about Irrelevant evidence",
            "type": "date_search"
        },
        {
            "query": "what case citations were in the Motion about Irrelevant evidence?",
            "type": "citation_search"
        },
        {
            "query": "what were the damanges?",
            "type": "monetary_search"
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


def demonstrate_case_isolation():
    """Demonstrate strict case isolation in Qdrant"""
    print("\n" + "="*80)
    print("CASE ISOLATION VERIFICATION")
    print("="*80)
    
    vector_store = QdrantVectorStore()
    
    # Get all unique cases (this is a simplified version)
    print("\nVerifying case isolation...")
    
    # Test search across cases (should return nothing)
    test_cases = ["Cerrito v Test", "Smith v Jones", "Doe v State"]
    
    for case in test_cases:
        try:
            stats = vector_store.get_case_statistics(case)
            if stats['total_chunks'] > 0:
                print(f"\n✓ Case '{case}': {stats['total_chunks']} chunks")
                
                # Try to search with wrong case name (should return empty)
                wrong_case = "WRONG_CASE_NAME"
                from src.vector_storage.embeddings import EmbeddingGenerator
                embed_gen = EmbeddingGenerator()
                test_embedding, _ = embed_gen.generate_embedding("test query")
                
                results = vector_store.search_case_documents(
                    case_name=wrong_case,
                    query_embedding=test_embedding,
                    limit=10
                )
                
                if len(results) == 0:
                    print(f"  ✓ Case isolation verified - no cross-case contamination")
                else:
                    print(f"  ✗ WARNING: Found {len(results)} results from wrong case!")
        except Exception as e:
            print(f"  - Case '{case}' not found or error: {str(e)}")


def demonstrate_performance_optimization():
    """Demonstrate Qdrant performance optimization features"""
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*80)
    
    vector_store = QdrantVectorStore()
    
    print("\n1. Collection Information:")
    try:
        # Get collection info
        info = vector_store.client.get_collection(vector_store.collection_name)
        print(f"  Vectors count: {info.vectors_count}")
        print(f"  Points count: {info.points_count}")
        print(f"  Segments count: {info.segments_count}")
        print(f"  Status: {info.status}")
        
        # Check configuration
        config = info.config
        print(f"\n2. Current Configuration:")
        print(f"  HNSW M parameter: {config.params.hnsw_config.m}")
        print(f"  HNSW ef_construct: {config.params.hnsw_config.ef_construct}")
        print(f"  Quantization enabled: {config.params.quantization_config is not None}")
        
        if config.params.quantization_config:
            print(f"  Quantization type: Scalar INT8")
        
    except Exception as e:
        print(f"  Error getting collection info: {str(e)}")
    
    print("\n3. Optimization Tips:")
    print("  - Use batch upload for initial data ingestion")
    print("  - Enable scalar quantization for 75% memory reduction")
    print("  - Set HNSW m=32 for legal document accuracy")
    print("  - Use gRPC for 15% better throughput")
    print("  - Keep vectors in RAM for optimal performance")


def demonstrate_backup_and_recovery():
    """Demonstrate Qdrant backup capabilities"""
    print("\n" + "="*80)
    print("BACKUP AND RECOVERY FEATURES")
    print("="*80)
    
    vector_store = QdrantVectorStore()
    
    print("\n1. Creating collection snapshot...")
    try:
        # Create snapshot
        snapshot_info = vector_store.client.create_snapshot(
            collection_name=vector_store.collection_name
        )
        print(f"  ✓ Snapshot created: {snapshot_info}")
        print("  Snapshots are stored in: /qdrant/snapshots/")
        
        # List snapshots
        snapshots = vector_store.client.list_snapshots(
            collection_name=vector_store.collection_name
        )
        print(f"\n2. Available snapshots: {len(snapshots)}")
        for snapshot in snapshots[:3]:  # Show first 3
            print(f"  - {snapshot}")
    
    except Exception as e:
        print(f"  Error creating snapshot: {str(e)}")
    
    print("\n3. Recovery procedures:")
    print("  - Snapshots can be used for point-in-time recovery")
    print("  - Use Docker volumes for persistent storage")
    print("  - Consider replication_factor=2 for redundancy")
    print("  - Regular backups to external storage recommended")


def main():
    """Main demonstration function"""
    print("\nCLERK LEGAL AI - QDRANT INTEGRATION DEMO")
    print("=========================================")
    
    # Run demonstrations
    demonstrate_qdrant_features()
    demonstrate_hybrid_search()
    demonstrate_case_isolation()
    demonstrate_performance_optimization()
    demonstrate_backup_and_recovery()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey advantages of Qdrant for legal document processing:")
    print("- 15x faster throughput than pgvector")
    print("- Native hybrid search combining vectors, keywords, and citations")
    print("- Scalar quantization reduces memory usage by 75%")
    print("- Production-ready with clustering and replication")
    print("- Strict case isolation with metadata filtering")
    print("- Built-in backup and recovery features")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()