#!/usr/bin/env python
"""
Quick fix: Create case_name index on legal_documents collection
Run this to immediately fix the zero results issue
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# Create the missing index
client = QdrantClient()

try:
    client.create_payload_index(
        collection_name="legal_documents",
        field_name="case_name",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("✓ Created case_name index successfully!")
    
    # Test it works
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    count = client.count(
        collection_name="legal_documents",
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="case_name",
                    match=MatchValue(value="Cerrtio v Test")
                )
            ]
        )
    )
    print(f"✓ Index working! Found {count.count} chunks for 'Cerrtio v Test'")
    
except Exception as e:
    if "already exists" in str(e):
        print("• Index already exists")
    else:
        print(f"✗ Error: {str(e)}")