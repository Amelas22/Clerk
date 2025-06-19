#!/usr/bin/env python
"""
Diagnostic script to verify Qdrant payload structure and search functionality
"""

import os
import sys
import json
from pprint import pprint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_storage.qdrant_store import QdrantVectorStore
from src.vector_storage.embeddings import EmbeddingGenerator
from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator


def diagnose_payload_structure():
    """Check the actual payload structure in Qdrant"""
    print("\n" + "="*80)
    print("DIAGNOSING QDRANT PAYLOAD STRUCTURE")
    print("="*80)
    
    store = QdrantVectorStore()
    
    # Get a sample of points from the collection
    print("\n1. Fetching sample points from collection...")
    try:
        # Use scroll to get some points
        results, _ = store.client.scroll(
            collection_name=store.collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        if not results:
            print("❌ No points found in collection!")
            return
        
        print(f"✓ Found {len(results)} points")
        
        # Examine payload structure
        print("\n2. Examining payload structure:")
        for i, point in enumerate(results[:2], 1):
            print(f"\n--- Point {i} (ID: {point.id}) ---")
            print("Payload keys:", list(point.payload.keys()))
            
            # Check critical fields
            case_name = point.payload.get("case_name")
            document_id = point.payload.get("document_id")
            
            print(f"case_name (top-level): '{case_name}'")
            print(f"document_id (top-level): '{document_id}'")
            
            # Check if case_name exists in nested metadata
            if "metadata" in point.payload:
                nested_case = point.payload["metadata"].get("case_name")
                print(f"case_name (in metadata): '{nested_case}'")
                if nested_case and nested_case != case_name:
                    print("⚠️  WARNING: case_name mismatch between top-level and metadata!")
            
            # Show first few fields
            print("\nFirst few payload fields:")
            for key, value in list(point.payload.items())[:5]:
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error fetching points: {str(e)}")
        return


def test_search_functionality():
    """Test search with different case names"""
    print("\n" + "="*80)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*80)
    
    store = QdrantVectorStore()
    deduplicator = QdrantDocumentDeduplicator()
    embed_gen = EmbeddingGenerator()
    
    # Get available cases
    stats = deduplicator.get_statistics()
    available_cases = stats.get("cases", [])
    
    if not available_cases:
        print("❌ No cases found in the system!")
        return
    
    print(f"\nAvailable cases: {available_cases}")
    
    # Test with first available case
    test_case = available_cases[0]
    print(f"\n3. Testing search for case: '{test_case}'")
    
    # Generate test embedding
    test_query = "test search query"
    test_embedding, _ = embed_gen.generate_embedding(test_query)
    
    # Try different filter approaches
    print("\n4. Testing different search approaches:")
    
    # Approach 1: Standard search
    print(f"\na) Standard search for case '{test_case}':")
    try:
        results = store.search_case_documents(
            case_name=test_case,
            query_embedding=test_embedding,
            limit=5,
            threshold=0.0  # Set low threshold to get any results
        )
        print(f"   Found {len(results)} results")
        if results:
            print(f"   First result case_name: '{results[0].case_name}'")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Approach 2: Direct search with manual filter
    print(f"\nb) Direct search with manual filter:")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Try exact match
        case_filter = Filter(
            must=[
                FieldCondition(
                    key="case_name",
                    match=MatchValue(value=test_case)
                )
            ]
        )
        
        results = store.client.search(
            collection_name=store.collection_name,
            query_vector=test_embedding,
            query_filter=case_filter,
            limit=5,
            with_payload=True
        )
        print(f"   Found {len(results)} results with exact match filter")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Approach 3: Count with filter
    print(f"\nc) Count points with case filter:")
    try:
        count = store.client.count(
            collection_name=store.collection_name,
            count_filter=case_filter
        ).count
        print(f"   Total points for case '{test_case}': {count}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Approach 4: Scroll to inspect all case_name values
    print(f"\nd) Inspecting unique case_name values in collection:")
    try:
        unique_cases = set()
        offset = None
        
        while True:
            results, next_offset = store.client.scroll(
                collection_name=store.collection_name,
                limit=100,
                offset=offset,
                with_payload=["case_name"],
                with_vectors=False
            )
            
            if not results:
                break
                
            for point in results:
                case = point.payload.get("case_name")
                if case:
                    unique_cases.add(case)
            
            offset = next_offset
            if not offset:
                break
        
        print(f"   Unique case names found: {list(unique_cases)[:5]}")  # Show first 5
        print(f"   Total unique cases: {len(unique_cases)}")
        
        # Check if our test case exists
        if test_case in unique_cases:
            print(f"   ✓ Test case '{test_case}' exists in collection")
        else:
            print(f"   ❌ Test case '{test_case}' NOT found in collection!")
            print(f"   Similar cases: {[c for c in unique_cases if test_case.lower() in c.lower()]}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")


def verify_fix():
    """Quick verification that the fix is working"""
    print("\n" + "="*80)
    print("VERIFYING FIX")
    print("="*80)
    
    store = QdrantVectorStore()
    embed_gen = EmbeddingGenerator()
    
    # Create a test chunk
    test_case = "TEST_DIAGNOSTIC_CASE"
    test_doc_id = "test_doc_diagnostic"
    
    print(f"\n5. Creating test data for case '{test_case}'...")
    
    test_chunk = {
        "content": "This is a test chunk for diagnostic purposes.",
        "embedding": embed_gen.generate_embedding("test diagnostic content")[0],
        "metadata": {
            "document_type": "test",
            "created_by": "diagnostic_script"
        }
    }
    
    # Store the chunk
    try:
        chunk_ids = store.store_document_chunks(
            case_name=test_case,
            document_id=test_doc_id,
            chunks=[test_chunk]
        )
        print(f"✓ Stored test chunk with ID: {chunk_ids[0]}")
        
        # Try to retrieve it
        results = store.search_case_documents(
            case_name=test_case,
            query_embedding=test_chunk["embedding"],
            limit=1,
            threshold=0.5
        )
        
        if results:
            print(f"✓ Successfully retrieved test chunk")
            print(f"  Case name in result: '{results[0].case_name}'")
            
            # Clean up
            deleted = store.delete_document_vectors(test_case, test_doc_id)
            print(f"✓ Cleaned up {deleted} test vectors")
        else:
            print("❌ Failed to retrieve test chunk!")
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\nQDRANT DIAGNOSTIC TOOL")
    print("="*80)
    
    # Run diagnostics
    diagnose_payload_structure()
    test_search_functionality()
    verify_fix()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)