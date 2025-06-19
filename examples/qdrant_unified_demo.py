#!/usr/bin/env python
"""
Demonstration of the unified Qdrant implementation for Clerk legal AI.
Shows how all database operations are now handled by Qdrant.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_injector import DocumentInjector
from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator
from src.vector_storage import QdrantVectorStore, EmbeddingGenerator


def demonstrate_unified_system():
    """Demonstrate the unified Qdrant system"""
    print("\n" + "="*80)
    print("CLERK LEGAL AI - UNIFIED QDRANT SYSTEM DEMONSTRATION")
    print("="*80)
    print("\nQdrant now handles ALL database operations:")
    print("- Vector storage and search")
    print("- Document deduplication")
    print("- Metadata management")
    print("- Case isolation")
    
    # Initialize components
    print("\n1. Initializing components...")
    deduplicator = QdrantDocumentDeduplicator()
    vector_store = QdrantVectorStore()
    embed_gen = EmbeddingGenerator()
    
    # Check collections
    try:
        collections = vector_store.client.get_collections().collections
        collection_names = [col.name for col in collections]
        print(f"\nActive Qdrant collections:")
        for name in collection_names:
            info = vector_store.client.get_collection(name)
            print(f"  - {name}: {info.points_count} points")
    except Exception as e:
        print(f"Error checking collections: {str(e)}")
        return


def demonstrate_deduplication():
    """Demonstrate document deduplication in Qdrant"""
    print("\n" + "="*80)
    print("DOCUMENT DEDUPLICATION DEMONSTRATION")
    print("="*80)
    
    deduplicator = QdrantDocumentDeduplicator()
    
    # Simulate processing documents
    print("\n1. Simulating document processing...")
    
    # First document
    test_content1 = b"This is the content of a legal motion to dismiss."
    doc_hash1 = deduplicator.calculate_document_hash(test_content1)
    
    exists, record = deduplicator.check_document_exists(doc_hash1)
    if not exists:
        print(f"  ✓ New document detected (hash: {doc_hash1[:8]}...)")
        record = deduplicator.register_new_document(
            doc_hash=doc_hash1,
            file_name="motion_to_dismiss.pdf",
            file_path="/cases/smith_v_jones/motion_to_dismiss.pdf",
            case_name="Smith v Jones",
            metadata={
                "document_type": "motion",
                "file_size": len(test_content1)
            }
        )
        print(f"  ✓ Registered in document registry")
    else:
        print(f"  ! Document already exists: {record['file_name']}")
    
    # Duplicate in different location
    print("\n2. Processing duplicate in different case...")
    exists, record = deduplicator.check_document_exists(doc_hash1)
    if exists:
        print(f"  ✓ Duplicate detected: {record['file_name']}")
        deduplicator.add_duplicate_location(
            doc_hash1,
            "/cases/doe_v_state/motion_to_dismiss_copy.pdf",
            "Doe v State"
        )
        print(f"  ✓ Recorded duplicate location")
    
    # Different document
    print("\n3. Processing different document...")
    test_content2 = b"This is a completely different document - a complaint."
    doc_hash2 = deduplicator.calculate_document_hash(test_content2)
    
    exists, record = deduplicator.check_document_exists(doc_hash2)
    if not exists:
        print(f"  ✓ New document detected (hash: {doc_hash2[:8]}...)")
        deduplicator.register_new_document(
            doc_hash=doc_hash2,
            file_name="complaint.pdf",
            file_path="/cases/smith_v_jones/complaint.pdf",
            case_name="Smith v Jones",
            metadata={"document_type": "complaint"}
        )
    
    # Show statistics
    print("\n4. Deduplication Statistics:")
    stats = deduplicator.get_statistics()
    print(f"  Total unique documents: {stats['total_unique_documents']}")
    print(f"  Total duplicate instances: {stats['total_duplicate_instances']}")
    print(f"  Total cases: {stats['total_cases']}")
    print(f"  Average duplicates per document: {stats['average_duplicates_per_document']:.2f}")


def demonstrate_case_management():
    """Demonstrate case-based operations"""
    print("\n" + "="*80)
    print("CASE MANAGEMENT DEMONSTRATION")
    print("="*80)
    
    deduplicator = QdrantDocumentDeduplicator()
    vector_store = QdrantVectorStore()
    
    case_name = "Demo v Example"
    
    print(f"\n1. Getting documents for case '{case_name}'...")
    case_docs = deduplicator.get_case_documents(case_name)
    
    if case_docs:
        print(f"  Found {len(case_docs)} documents:")
        for doc in case_docs[:5]:  # Show first 5
            print(f"  - {doc['file_name']} (added: {doc['first_seen_at']})")
    else:
        print("  No documents found for this case")
    
    print(f"\n2. Getting vector statistics for case '{case_name}'...")
    vector_stats = vector_store.get_case_statistics(case_name)
    print(f"  Total chunks: {vector_stats['total_chunks']}")
    print(f"  Unique documents: {vector_stats['unique_documents']}")


def demonstrate_integrated_workflow():
    """Demonstrate the complete integrated workflow"""
    print("\n" + "="*80)
    print("INTEGRATED WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Initialize document injector
    injector = DocumentInjector(enable_cost_tracking=True)
    
    print("\n1. Document Processing Pipeline:")
    print("  Box → Download → Deduplication Check (Qdrant) → ")
    print("  Extract Text → Chunk → Generate Context → ")
    print("  Create Embeddings → Store in Qdrant")
    
    print("\n2. Search Pipeline:")
    print("  Query → Analyze → Generate Embeddings → ")
    print("  Search Qdrant (Vector + Keyword + Citation) → ")
    print("  Rank Results → Return with Case Isolation")
    
    print("\n3. All operations use Qdrant:")
    print("  ✓ Document registry for deduplication")
    print("  ✓ Vector storage for semantic search")
    print("  ✓ Sparse vectors for keyword/citation matching")
    print("  ✓ Metadata filtering for case isolation")
    print("  ✓ Statistics and reporting")


def demonstrate_data_export():
    """Demonstrate exporting data from Qdrant"""
    print("\n" + "="*80)
    print("DATA EXPORT DEMONSTRATION")
    print("="*80)
    
    deduplicator = QdrantDocumentDeduplicator()
    
    print("\n1. Exporting document registry...")
    registry = deduplicator.export_registry()
    
    total_docs = sum(len(docs) for docs in registry.values())
    print(f"  Exported {len(registry)} cases with {total_docs} total documents")
    
    # Show sample
    for case, docs in list(registry.items())[:2]:
        print(f"\n  Case: {case}")
        for doc in docs[:3]:
            print(f"    - {doc['file_name']} ({doc['document_hash'][:8]}...)")


def cleanup_demo_data():
    """Clean up demonstration data"""
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    
    deduplicator = QdrantDocumentDeduplicator()
    vector_store = QdrantVectorStore()
    
    demo_cases = ["Demo v Example", "Smith v Jones", "Doe v State"]
    
    for case in demo_cases:
        # Clean deduplication records
        deleted_docs = deduplicator.cleanup_case(case)
        if deleted_docs > 0:
            print(f"  Cleaned {deleted_docs} documents from {case}")
        
        # Clean vectors
        deleted_vectors = vector_store.delete_document_vectors(case, "all")
        if deleted_vectors > 0:
            print(f"  Cleaned {deleted_vectors} vectors from {case}")


def main():
    """Main demonstration function"""
    print("\nCLERK LEGAL AI - UNIFIED QDRANT IMPLEMENTATION")
    print("=" * 80)
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstrations
    demonstrate_unified_system()
    demonstrate_deduplication()
    demonstrate_case_management()
    demonstrate_integrated_workflow()
    demonstrate_data_export()
    
    # Cleanup
    print("\nClean up demo data? (y/n): ", end="")
    if input().lower() == 'y':
        cleanup_demo_data()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Benefits of Unified Qdrant System:")
    print("- Single database for all operations")
    print("- Simplified infrastructure and deployment")
    print("- Consistent performance across all features")
    print("- Unified backup and recovery")
    print("- Reduced operational complexity")


if __name__ == "__main__":
    main()