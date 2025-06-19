#!/usr/bin/env python
"""
Initial setup script for Qdrant in the Clerk legal AI system.
This script ensures all collections and indexes are properly configured.

Usage: python scripts/setup_qdrant.py
"""

import logging
import sys
import os
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.vector_storage.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def check_environment() -> bool:
    """Check if all required environment variables are set"""
    print("Checking environment variables...")
    
    required_vars = {
        "Box API": ["BOX_CLIENT_ID", "BOX_CLIENT_SECRET"],
        "OpenAI": ["OPENAI_API_KEY"],
        "Qdrant": ["QDRANT_HOST"]
    }
    
    all_good = True
    
    for service, vars in required_vars.items():
        print(f"\n{service}:")
        for var in vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if "KEY" in var or "SECRET" in var or "PASSWORD" in var:
                    masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                    print(f"  ✓ {var}: {masked}")
                else:
                    print(f"  ✓ {var}: {value}")
            else:
                print(f"  ✗ {var}: NOT SET")
                all_good = False
    
    return all_good


def setup_qdrant_collections() -> bool:
    """Set up Qdrant collections with optimal configuration"""
    print("\n" + "="*60)
    print("Setting up Qdrant collections...")
    print("="*60)
    
    try:
        # Initialize Qdrant store (this will create collections if needed)
        vector_store = QdrantVectorStore()
        
        # Initialize document deduplicator (this will create registry collection)
        from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator
        deduplicator = QdrantDocumentDeduplicator()
        
        # Check collections
        collections = vector_store.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        print(f"\nExisting collections: {collection_names}")
        
        # Verify standard collection
        if vector_store.collection_name in collection_names:
            info = vector_store.client.get_collection(vector_store.collection_name)
            print(f"\n✓ Standard collection '{vector_store.collection_name}':")
            print(f"  - Vectors: {info.vectors_count}")
            print(f"  - Points: {info.points_count}")
            print(f"  - Status: {info.status}")
        else:
            print(f"\n✗ Standard collection '{vector_store.collection_name}' not found!")
            return False
        
        # Verify hybrid collection
        if settings.legal["enable_hybrid_search"]:
            if vector_store.hybrid_collection_name in collection_names:
                info = vector_store.client.get_collection(vector_store.hybrid_collection_name)
                print(f"\n✓ Hybrid collection '{vector_store.hybrid_collection_name}':")
                print(f"  - Vectors: {info.vectors_count}")
                print(f"  - Points: {info.points_count}")
                print(f"  - Status: {info.status}")
            else:
                print(f"\n✗ Hybrid collection '{vector_store.hybrid_collection_name}' not found!")
                return False
        
        # Verify document registry collection
        if deduplicator.collection_name in collection_names:
            info = deduplicator.client.get_collection(deduplicator.collection_name)
            print(f"\n✓ Document registry collection '{deduplicator.collection_name}':")
            print(f"  - Documents: {info.points_count}")
            print(f"  - Status: {info.status}")
            
            # Get deduplication statistics
            stats = deduplicator.get_statistics()
            print(f"  - Unique documents: {stats['total_unique_documents']}")
            print(f"  - Total cases: {stats['total_cases']}")
        else:
            print(f"\n✗ Document registry collection '{deduplicator.collection_name}' not found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error setting up Qdrant: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_connectivity() -> Dict[str, bool]:
    """Verify connectivity to all required services"""
    print("\n" + "="*60)
    print("Verifying service connectivity...")
    print("="*60)
    
    results = {}
    
    # Test Qdrant
    print("\n1. Testing Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            prefer_grpc=settings.qdrant.prefer_grpc
        )
        print(f"QDRANT_HOST is: '{os.getenv('QDRANT_HOST')}'")
        collections = client.get_collections()
        print(f"  ✓ Qdrant is reachable. Collections: {collections}")
        results["qdrant"] = True
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {str(e)}")
        results["qdrant"] = False
    
    # Test OpenAI
    print("\n2. Testing OpenAI connection...")
    try:
        from src.vector_storage.embeddings import EmbeddingGenerator
        embed_gen = EmbeddingGenerator()
        test_embedding, _ = embed_gen.generate_embedding("test")
        print(f"  ✓ OpenAI connection successful")
        print(f"  - Model: {settings.openai.embedding_model}")
        print(f"  - Dimensions: {len(test_embedding)}")
        results["openai"] = True
    except Exception as e:
        print(f"  ✗ OpenAI connection failed: {str(e)}")
        results["openai"] = False
    
    # Test Box
    print("\n3. Testing Box connection...")
    try:
        from src.document_processing.box_client import BoxClient
        box_client = BoxClient()
        if box_client.check_connection():
            print(f"  ✓ Box connection successful")
            results["box"] = True
        else:
            print(f"  ✗ Box connection failed")
            results["box"] = False
    except Exception as e:
        print(f"  ✗ Box connection failed: {str(e)}")
        results["box"] = False
    
    return results


def create_test_data() -> bool:
    """Create and verify test data"""
    print("\n" + "="*60)
    print("Creating test data...")
    print("="*60)
    
    try:
        from src.vector_storage.qdrant_store import QdrantVectorStore
        from src.vector_storage.embeddings import EmbeddingGenerator
        from src.document_processing.qdrant_deduplicator import QdrantDocumentDeduplicator
        
        vector_store = QdrantVectorStore()
        embed_gen = EmbeddingGenerator()
        deduplicator = QdrantDocumentDeduplicator()
        
        # Create test chunks
        test_chunks = [
            {
                "content": "This is a test legal document. Motion to dismiss pursuant to Rule 12(b)(6).",
                "embedding": embed_gen.generate_embedding("Test legal document motion to dismiss")[0],
                "search_text": "test legal document motion dismiss rule 12b6",
                "metadata": {
                    "document_type": "test",
                    "created_by": "setup_script"
                }
            }
        ]
        
        # Store in test case
        chunk_ids = vector_store.store_document_chunks(
            case_name="TEST_CASE_SETUP",
            document_id="test_doc_001",
            chunks=test_chunks,
            use_hybrid=True
        )
        
        print(f"  ✓ Created {len(chunk_ids)} test chunks")
        
        # Verify retrieval
        results = vector_store.search_case_documents(
            case_name="TEST_CASE_SETUP",
            query_embedding=test_chunks[0]["embedding"],
            limit=5
        )
        
        if results:
            print(f"  ✓ Successfully retrieved test data")
            
            # Clean up vector data
            deleted = vector_store.delete_document_vectors(
                case_name="TEST_CASE_SETUP",
                document_id="test_doc_001"
            )
            print(f"  ✓ Cleaned up {deleted} test vectors")
            
            # Clean up deduplication data
            deduplicator.cleanup_case("TEST_CASE_SETUP")
            print(f"  ✓ Cleaned up test deduplication records")
            
            return True
        else:
            print(f"  ✗ Failed to retrieve test data")
            return False
            
    except Exception as e:
        print(f"  ✗ Error creating test data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def display_configuration() -> None:
    """Display current configuration"""
    print("\n" + "="*60)
    print("Current Configuration")
    print("="*60)
    
    print(f"\nQdrant Settings:")
    print(f"  URL: {settings.qdrant.url}")
    print(f"  gRPC enabled: {settings.qdrant.prefer_grpc}")
    print(f"  Collection: {settings.qdrant.collection_name}")
    print(f"  Batch size: {settings.qdrant.batch_size}")
    
    print(f"\nVector Settings:")
    print(f"  Embedding model: {settings.openai.embedding_model}")
    print(f"  Dimensions: {settings.vector.embedding_dimensions}")
    print(f"  Distance metric: {settings.vector.distance_metric}")
    print(f"  HNSW M: {settings.vector.hnsw_m}")
    
    # Handle quantization setting gracefully
    quantization_enabled = False
    if hasattr(settings.vector, 'quantization'):
        quantization_enabled = settings.vector.quantization
    elif hasattr(settings.vector, 'quantization_enabled'):
        quantization_enabled = settings.vector.quantization_enabled
    print(f"  Quantization: {quantization_enabled}")
    
    print(f"\nDocument Processing:")
    print(f"  Chunk size: {settings.chunking.target_chunk_size}")
    print(f"  Overlap: {settings.chunking.overlap_size}")
    print(f"  Context model: {settings.openai.context_model}")
    
    print(f"\nLegal AI Features:")
    print(f"  Case isolation: {settings.legal['enable_case_isolation']}")
    print(f"  Citation tracking: {settings.legal['enable_citation_tracking']}")
    print(f"  Hybrid search: {settings.legal['enable_hybrid_search']}")
    if settings.legal['enable_hybrid_search']:
        weights = settings.legal['hybrid_search_weights']
        print(f"  - Vector weight: {weights['vector']}")
        print(f"  - Keyword weight: {weights['keyword']}")
        print(f"  - Citation weight: {weights['citation']}")


def main():
    """Main setup function"""
    print("\n" + "="*80)
    print("CLERK LEGAL AI - QDRANT SETUP")
    print("="*80)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check environment
    if not check_environment():
        print("\n✗ Missing required environment variables!")
        print("Please check your .env file and try again.")
        sys.exit(1)
    
    # Display configuration
    display_configuration()
    
    # Verify connectivity
    connectivity = verify_connectivity()
    
    if not all(connectivity.values()):
        print("\n✗ Some services are not accessible!")
        failed = [k for k, v in connectivity.items() if not v]
        print(f"Failed services: {', '.join(failed)}")
        
        if not connectivity.get("qdrant"):
            print("\nQdrant is required. Please ensure:")
            print("1. Docker container is running: docker ps")
            print("2. Ports are correctly mapped")
            print("3. QDRANT_HOST is set correctly")
            sys.exit(1)
    
    # Setup collections
    if not setup_qdrant_collections():
        print("\n✗ Failed to setup Qdrant collections!")
        sys.exit(1)
    
    # Create and verify test data
    if not create_test_data():
        print("\n✗ Test data verification failed!")
        print("The system may not be functioning correctly.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE")
    print("="*80)
    print("\nYour Clerk legal AI system is ready to use with Qdrant!")
    print("\nNext steps:")
    print("1. Process your first case: python -m src.document_injector --folder-id YOUR_FOLDER_ID")
    print("2. Test hybrid search: python examples/qdrant_usage.py")
    print("3. Monitor performance: docker logs qdrant")
    print("\nFor more information, see README_QDRANT.md")


if __name__ == "__main__":
    main()