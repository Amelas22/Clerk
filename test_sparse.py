"""
Test script to verify sparse vector generation is working correctly.
Run this to ensure sparse vectors have multiple indices and values.
"""

import json
from src.vector_storage.sparse_encoder import SparseVectorEncoder


def test_sparse_vector_generation():
    """Test that sparse vectors are generated correctly"""
    
    # Initialize encoder
    encoder = SparseVectorEncoder()
    
    # Test text with various legal content
    test_text = """
    The plaintiff filed a motion to dismiss under Rule 12(b)(6) of the Federal Rules 
    of Civil Procedure. The court found that the defendant violated 42 U.S.C. § 1983 
    and awarded damages of $150,000. This decision was made on January 15, 2024.
    
    The case cites Smith v. Jones, 123 F.3d 456 (9th Cir. 2023) as precedent for 
    the liability determination. The settlement amount included attorney fees of 
    $50,000 and costs of $5,000.
    """
    
    print("Testing sparse vector generation...")
    print("=" * 80)
    print(f"Test text: {test_text[:200]}...")
    print("=" * 80)
    
    # Generate sparse vectors
    keyword_sparse, citation_sparse = encoder.encode_for_hybrid_search(test_text)
    
    print("\n1. KEYWORD SPARSE VECTOR:")
    print(f"   - Number of non-zero elements: {len(keyword_sparse)}")
    print(f"   - Sample indices: {list(keyword_sparse.keys())[:10]}")
    print(f"   - Sample values: {list(keyword_sparse.values())[:10]}")
    print(f"   - Index range: {min(keyword_sparse.keys()) if keyword_sparse else 'N/A'} to {max(keyword_sparse.keys()) if keyword_sparse else 'N/A'}")
    
    print("\n2. CITATION SPARSE VECTOR:")
    print(f"   - Number of non-zero elements: {len(citation_sparse)}")
    print(f"   - Indices: {list(citation_sparse.keys())}")
    print(f"   - Values: {list(citation_sparse.values())}")
    
    # Extract entities to verify
    entities = encoder.extract_legal_entities(test_text)
    print("\n3. EXTRACTED ENTITIES:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"   - {entity_type}: {entity_list}")
    
    # Test tokenization
    tokens = encoder.tokenize_legal_text(test_text)
    print(f"\n4. TOKENIZATION:")
    print(f"   - Number of tokens: {len(tokens)}")
    print(f"   - Sample tokens: {tokens[:20]}")
    
    # Verify the format for Qdrant
    print("\n5. QDRANT FORMAT:")
    if keyword_sparse:
        indices_list = list(keyword_sparse.keys())
        values_list = list(keyword_sparse.values())
        qdrant_format = {
            "indices": indices_list,
            "values": values_list
        }
        print(f"   - Keyword sparse vector for Qdrant: {json.dumps(qdrant_format, indent=2)}")
    
    # Test edge cases
    print("\n6. EDGE CASE TESTS:")
    
    # Empty text
    empty_kw, empty_cite = encoder.encode_for_hybrid_search("")
    print(f"   - Empty text: keywords={len(empty_kw)}, citations={len(empty_cite)}")
    
    # Only citations
    citation_only = "See 42 U.S.C. § 1983 and Rule 12(b)(6)"
    cite_kw, cite_cite = encoder.encode_for_hybrid_search(citation_only)
    print(f"   - Citation only text: keywords={len(cite_kw)}, citations={len(cite_cite)}")
    
    # Regular text without legal terms
    regular_text = "The weather today is sunny and warm."
    reg_kw, reg_cite = encoder.encode_for_hybrid_search(regular_text)
    print(f"   - Regular text: keywords={len(reg_kw)}, citations={len(reg_cite)}")
    
    print("\n7. VALIDATION:")
    if len(keyword_sparse) > 1:
        print("   ✓ Keyword sparse vector has multiple elements")
    else:
        print("   ✗ Keyword sparse vector has too few elements")
    
    if all(isinstance(idx, int) for idx in keyword_sparse.keys()):
        print("   ✓ All indices are integers")
    else:
        print("   ✗ Some indices are not integers")
    
    if all(isinstance(val, float) for val in keyword_sparse.values()):
        print("   ✓ All values are floats")
    else:
        print("   ✗ Some values are not floats")
    
    print("\n" + "=" * 80)
    print("Test complete!")


def test_hash_consistency():
    """Test that hash-based indexing is consistent"""
    encoder = SparseVectorEncoder()
    
    print("\n\nTesting hash consistency...")
    print("=" * 80)
    
    # Test that same token always maps to same index
    test_tokens = ["plaintiff", "motion", "dismiss", "damages", "settlement"]
    
    print("Testing same token produces same index:")
    for token in test_tokens:
        indices = []
        for i in range(3):
            idx = encoder._hash_token_to_index(token)
            indices.append(idx)
        
        if len(set(indices)) == 1:
            print(f"   ✓ '{token}' -> {indices[0]} (consistent)")
        else:
            print(f"   ✗ '{token}' -> {indices} (INCONSISTENT!)")
    
    # Test distribution of indices
    print("\nTesting index distribution:")
    num_tokens = 1000
    indices = []
    for i in range(num_tokens):
        token = f"token_{i}"
        idx = encoder._hash_token_to_index(token)
        indices.append(idx)
    
    unique_indices = len(set(indices))
    print(f"   - {num_tokens} unique tokens -> {unique_indices} unique indices")
    print(f"   - Collision rate: {(num_tokens - unique_indices) / num_tokens * 100:.2f}%")
    

if __name__ == "__main__":
    test_sparse_vector_generation()
    test_hash_consistency()