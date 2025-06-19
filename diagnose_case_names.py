from src.vector_storage.qdrant_store import QdrantVectorStore
store = QdrantVectorStore()
# Get a sample chunk directly
sample = store.client.retrieve(
    collection_name="legal_documents",
    ids=["any-chunk-id"],  # or use scroll to get any ID
    with_payload=True
)
print(sample[0].payload)