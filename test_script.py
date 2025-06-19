#!/usr/bin/env python
"""
Verbose debugging of Qdrant search to see exactly what's happening
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.vector_storage.embeddings import EmbeddingGenerator
import json


def verbose_search_debug():
    """Step-by-step search debugging"""
    
    print("\nVERBOSE QDRANT SEARCH DEBUGGING")
    print("="*60)
    
    client = QdrantClient()
    collection_name = "legal_documents"
    test_case = "Cerrtio v Test"
    
    # Step 1: Verify exact case name in data
    print("\n1. VERIFYING EXACT CASE NAME IN DATA:")
    print("-"*40)
    
    sample = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=["case_name"],
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="case_name",
                    match=MatchValue(value=test_case)
                )
            ]
        )
    )
    
    if sample[0]:
        print(f"✓ Found {len(sample[0])} points with case_name = '{test_case}'")
        actual_case_name = sample[0][0].payload.get("case_name")
        print(f"  Actual value in DB: '{actual_case_name}'")
        print(f"  Length: {len(actual_case_name)} chars")
        print(f"  Repr: {repr(actual_case_name)}")  # Shows hidden characters
    else:
        print(f"✗ No points found with case_name = '{test_case}'")
        
        # Try to find what case names exist
        print("\n  Searching for available case names...")
        all_samples = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=["case_name"]
        )
        
        unique_cases = set()
        for point in all_samples[0]:
            case = point.payload.get("case_name")
            if case:
                unique_cases.add(case)
        
        print(f"  Found case names: {list(unique_cases)}")
        return
    
    # Step 2: Get a vector for testing
    print("\n2. GETTING TEST VECTOR:")
    print("-"*40)
    
    point_id = sample[0][0].id
    point_with_vector = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_vectors=True,
        with_payload=True
    )
    
    if not point_with_vector:
        print("✗ Failed to retrieve point with vector")
        return
    
    test_vector = point_with_vector[0].vector
    print(f"✓ Retrieved vector with dimension: {len(test_vector)}")
    print(f"  Point ID: {point_id}")
    print(f"  Content preview: {point_with_vector[0].payload.get('content', '')[:100]}...")
    
    # Step 3: Test search WITHOUT filter
    print("\n3. SEARCH WITHOUT FILTER:")
    print("-"*40)
    
    no_filter_results = client.search(
        collection_name=collection_name,
        query_vector=test_vector,
        limit=5
    )
    
    print(f"Found {len(no_filter_results)} results")
    if no_filter_results:
        print(f"  Best score: {no_filter_results[0].score}")
        print(f"  Best match case: {no_filter_results[0].payload.get('case_name')}")
    
    # Step 4: Test search WITH filter
    print("\n4. SEARCH WITH CASE FILTER:")
    print("-"*40)
    
    case_filter = Filter(
        must=[
            FieldCondition(
                key="case_name",
                match=MatchValue(value=test_case)
            )
        ]
    )
    
    print(f"Filter: case_name = '{test_case}'")
    
    filtered_results = client.search(
        collection_name=collection_name,
        query_vector=test_vector,
        query_filter=case_filter,
        limit=5,
        score_threshold=0.0  # Accept any score
    )
    
    print(f"Found {len(filtered_results)} results")
    if filtered_results:
        print(f"  Best score: {filtered_results[0].score}")
        for i, result in enumerate(filtered_results[:3]):
            print(f"\n  Result {i+1}:")
            print(f"    Score: {result.score}")
            print(f"    Case: {result.payload.get('case_name')}")
            print(f"    Doc: {result.payload.get('document_name')}")
    
    # Step 5: Test with a real query
    print("\n5. TEST WITH REAL QUERY:")
    print("-"*40)
    
    query = "What are the allegations in the complaint?"
    print(f"Query: '{query}'")
    
    embed_gen = EmbeddingGenerator()
    query_vector, tokens = embed_gen.generate_embedding(query)
    print(f"✓ Generated query embedding (tokens: {tokens})")
    
    query_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=case_filter,
        limit=5,
        score_threshold=0.0
    )
    
    print(f"Found {len(query_results)} results")
    if query_results:
        print(f"  Best score: {query_results[0].score}")
        print(f"  Best match content: {query_results[0].payload.get('content', '')[:150]}...")
    
    # Step 6: Check indexes
    print("\n6. INDEX CHECK:")
    print("-"*40)
    
    try:
        # Try creating index to see if it exists
        client.create_payload_index(
            collection_name=collection_name,
            field_name="case_name",
            field_schema="keyword"
        )
        print("✗ case_name index was MISSING (just created it)")
        print("  This was likely the cause of zero results!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("✓ case_name index exists")
        else:
            print(f"✗ Index check error: {str(e)}")
    
    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    verbose_search_debug()