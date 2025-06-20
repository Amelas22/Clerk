#!/usr/bin/env python
"""
Comprehensive example demonstrating all features of the Clerk document injector:
- Document processing with cost tracking
- Hybrid search (vector + full-text)
- Case isolation verification
- Cost reporting
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_injector import DocumentInjector
from src.vector_storage.qdrant_store import QdrantVectorStore
from src.ai_agents.legal_document_agent import LegalDocumentAgent
from src.document_processing import DocumentDeduplicator


def demonstrate_document_processing():
    """Demonstrate document processing with cost tracking"""
    print("\n" + "="*80)
    print("DOCUMENT PROCESSING WITH COST TRACKING")
    print("="*80)
    
    # Initialize injector with cost tracking enabled
    injector = DocumentInjector(enable_cost_tracking=True)
    
    # Initialize components
    vector_store = QdrantVectorStore()
    legal_agent = LegalDocumentAgent(allowed_case_name="general")
    
    # Process a sample folder (replace with your actual Box folder ID)
    folder_id = "325242457476"
    folder_name = "Cerrito v Test"  # Folder name becomes collection name
    
    print(f"\nProcessing folder: {folder_id}")
    print(f"Using collection: {folder_name}")
    
    # Simulate processing documents
    for i in range(3):
        doc = {
            "content": f"Sample legal document {i}",
            "metadata": {"source": "Box"},
            "document_type": "pleading"
        }
        legal_agent.index_document(doc, folder_name)
        print(f"Indexed document {i}")
    
    print("\nIndexing complete")

    # Return the injector for cost tracking
    return injector


async def demonstrate_hybrid_search(case_name: str):
    """Demonstrate hybrid search capabilities"""
    print("\n" + "="*80)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*80)
    
    # Initialize search components
    search_manager = QdrantVectorStore()
    legal_agent = LegalDocumentAgent(allowed_case_name="Cerrito v Test")
    
    # Example: Legal document search
    print("\n1. Legal Document Search")
    print("-" * 40)
    
    query = "What counts were filed in this complaint?"
    print(f"Query: {query}")
    
    # Generate embedding
    query_embedding, tokens = legal_agent.generate_embedding(query)
    print(f"Embedding tokens used: {tokens}")
    
    # Perform hybrid search
    results = await search_manager.hybrid_search(
        folder_name=case_name,
        query=query,
        query_embedding=query_embedding,
        limit=5
    )
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Combined Score: {result.combined_score:.3f}")
        print(f"   Vector: {result.vector_similarity:.3f}, Text: {result.text_rank:.3f}")
        print(f"   Document: {result.metadata.get('document_name', 'Unknown')}")
        print(f"   Preview: {result.content[:200]}...")
    
    # Example 2
    print("\n\n2. Hybrid Search")
    print("-" * 40)
    
    query = "What did we say about the toxicology?"
    print(f"Query: {query}")
    
    query_embedding, tokens = legal_agent.generate_embedding(query)
    
    results = await search_manager.hybrid_search(
        folder_name=case_name,
        query=query,
        query_embedding=query_embedding,
        limit=5
    )
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Combined Score: {result.combined_score:.3f}")
        print(f"   Document: {result.metadata.get('document_name', 'Unknown')}")
        print(f"   Preview: {result.content[:200]}...")
    
    # Example: Financial search
    print("\n\n3. Financial/Monetary Search")
    print("-" * 40)
    
    query = "What was the point of the fee petition?"
    print(f"Query: {query}")
    
    query_embedding, tokens = legal_agent.generate_embedding(query)
    
    results = await search_manager.hybrid_search(
        folder_name=case_name,
        query=query,
        query_embedding=query_embedding,
        limit=5
    )
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Combined Score: {result.combined_score:.3f}")
        print(f"   Document: {result.metadata.get('document_name', 'Unknown')}")
        print(f"   Preview: {result.content[:200]}...")


def verify_case_isolation():
    """Verify that case isolation is working properly"""
    print("\n" + "="*80)
    print("CASE ISOLATION VERIFICATION")
    print("="*80)
    
    vector_store = QdrantVectorStore()
    
    # Get list of cases
    cases = ["Cerrito v Test"]
    
    print("\nVerifying case isolation for each case:")
    for case in cases:
        is_isolated = vector_store.verify_case_isolation(case, sample_size=20)
        status = "✅ PASS" if is_isolated else "❌ FAIL"
        print(f"{case}: {status}")
        
        # Get statistics
        stats = vector_store.get_case_statistics(case)
        print(f"  - Documents: {stats['unique_documents']}")
        print(f"  - Chunks: {stats['total_chunks']}")


def check_deduplication_stats():
    """Display deduplication statistics"""
    print("\n" + "="*80)
    print("DEDUPLICATION STATISTICS")
    print("="*80)
    
    deduplicator = DocumentDeduplicator()
    stats = deduplicator.get_statistics()
    
    print(f"\nTotal unique documents: {stats['total_unique_documents']}")
    print(f"Total duplicate instances: {stats['total_duplicate_instances']}")
    print(f"Total cases in system: {stats['total_cases']}")
    print(f"Average duplicates per document: {stats['average_duplicates_per_document']:.2f}")
    
    # Show some specific cases
    print("\nSample case document counts:")
    sample_cases = ["Cerrito v Test"]
    for case in sample_cases:
        case_docs = deduplicator.get_case_documents(case)
        print(f"- {case}: {len(case_docs)} unique documents")


def generate_comprehensive_report(injector: DocumentInjector, output_dir: str = "reports"):
    """Generate a comprehensive processing report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cost report
    if injector.enable_cost_tracking:
        cost_path = injector.cost_tracker.save_report(
            os.path.join(output_dir, f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        )
        print(f"Cost report saved to: {cost_path}")
    
    # Generate summary report
    summary_path = os.path.join(output_dir, f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(summary_path, 'w') as f:
        f.write("CLERK DOCUMENT PROCESSING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Processing stats
        f.write("PROCESSING STATISTICS:\n")
        f.write(f"Total documents processed: {injector.stats['total_processed']}\n")
        f.write(f"Successful: {injector.stats['successful']}\n")
        f.write(f"Duplicates: {injector.stats['duplicates']}\n")
        f.write(f"Failed: {injector.stats['failed']}\n\n")
        
        # Cost summary
        if injector.enable_cost_tracking:
            report = injector.cost_tracker.get_session_report()
            f.write("COST SUMMARY:\n")
            f.write(f"Total API calls: {report['summary']['total_api_calls']}\n")
            f.write(f"Total tokens: {report['tokens']['total_tokens']:,}\n")
            f.write(f"Total cost: ${report['costs']['total_cost']:.4f}\n")
            f.write(f"Average per document: ${report['costs']['average_per_document']:.4f}\n\n")
            
            f.write("COSTS BY CASE:\n")
            for case, data in report['costs_by_case'].items():
                f.write(f"- {case}: ${data['cost']:.4f} ({data['documents']} docs)\n")
    
    print(f"Summary report saved to: {summary_path}")


async def main():
    """Main demonstration function"""
    print("\nCLERK DOCUMENT INJECTOR - COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    
    # 1. Process documents with cost tracking
    injector = demonstrate_document_processing()
    
    # 2. Demonstrate hybrid search (use a case from the results)
    await demonstrate_hybrid_search("Smith_v_Jones")
    
    # # 3. Verify case isolation
    # verify_case_isolation()
    
    # # 4. Check deduplication statistics
    # check_deduplication_stats()
    
    # # 5. Generate comprehensive report
    generate_comprehensive_report(injector)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/comprehensive_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    import asyncio
    asyncio.run(main())