## New Features Added

### 1. Hybrid Search (Vector + Full-Text)

The system now supports hybrid search combining semantic similarity (vector search) with keyword matching (full-text search):

- **Automatic text preprocessing** for optimal keyword search
- **Weighted scoring** to balance semantic and keyword matching
- **Query analysis** to detect search intent and important terms
- **Support for legal citations, dates, monetary amounts, and medical terminology**

Key benefits:
- Better results for specific legal citations (e.g., "§ 1983")
- Improved accuracy for date and monetary amount searches
- Flexible weighting based on query type

### 2. Comprehensive Cost Tracking

Track API usage and costs in real-time during document processing:

- **Per-document cost breakdown** showing tokens and costs for each operation
- **Case-level cost aggregation** to understand costs by legal matter
- **Detailed token usage** for embeddings and context generation
- **Multiple report formats**: JSON and Excel
- **Session-based tracking** for comparing costs across processing runs

Cost reports include:
- Total API calls and tokens used
- Cost breakdown by operation type (embedding vs. context)
- Top expensive documents
- Average cost per document
- Pricing reference for all models used

### 3. Excel Reporting

Generate detailed Excel workbooks with multiple sheets:
- Summary overview
- Document-level details
- Case-level aggregation
- API usage breakdown
- Pricing reference

Perfect for:
- Budget planning and cost allocation
- Identifying expensive documents
- Optimizing processing strategies# Clerk Document Injector

The document injector component of the Clerk legal AI system. This module processes PDF documents from Box cloud storage, chunks them with contextual summaries, and stores them in a vector database with strict case isolation. It now includes hybrid search capabilities (vector + full-text) and comprehensive API cost tracking.

## Key Features

- **Box Integration**: Automatically traverses Box folders to find and process PDFs
- **Duplicate Detection**: Hash-based deduplication prevents redundant processing
- **Smart Chunking**: Creates ~1100 character chunks with 200 character overlap
- **Contextual Enhancement**: Uses LLM to add contextual summaries to each chunk
- **Hybrid Search**: Combines vector similarity with full-text search for better results
- **Vector Storage**: Stores embeddings in Qdrant
- **Case Isolation**: Strict metadata filtering ensures case data never mixes
- **Cost Tracking**: Detailed tracking of API usage and costs per document

## What's New

### Hybrid Search (Vector + Full-Text)
- Combines semantic understanding with keyword precision
- Automatically preprocesses text for optimal search
- Supports legal citations, dates, monetary amounts
- Configurable weighting between vector and text matching

### API Cost Tracking
- Real-time tracking of OpenAI API usage
- Per-document cost breakdown
- Case-level cost aggregation
- Excel report generation
- Session-based comparison

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Box API Configuration
BOX_CLIENT_ID=your_box_client_id
BOX_CLIENT_SECRET=your_box_client_secret
BOX_ENTERPRISE_ID=your_enterprise_id
BOX_JWT_KEY_ID=your_jwt_key_id
BOX_PRIVATE_KEY="-----BEGIN ENCRYPTED PRIVATE KEY-----\n...\n-----END ENCRYPTED PRIVATE KEY-----"
BOX_PASSPHRASE=your_private_key_passphrase

# Qdrant Configuration
QDRANT_HOST=your_qdrant_host
QDRANT_PORT=your_qdrant_port
QDRANT_GRPC_PORT=your_qdrant_grpc_port
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_HTTPS=your_qdrant_https
QDRANT_TIMEOUT=your_qdrant_timeout
QDRANT_PREFER_GRPC=your_qdrant_prefer_grpc

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
CONTEXT_LLM_MODEL=gpt-3.5-turbo  # or gpt-4 for better contexts

# Optional: Override default settings
# CHUNK_SIZE=1100
# CHUNK_OVERLAP=200
```

### 2. Database Setup

Run the migration script in your Qdrant SQL editor:

```sql
-- See migrations/001_hybrid_search_setup.sql for the complete setup
```

This will create:
- Document registry table for deduplication
- Case documents table with vector and full-text search capabilities
- Hybrid search functions combining vector and text search
- Necessary indexes for performance
- Helper views for statistics

Key tables created:
- `document_registry`: Tracks unique documents and duplicates
- `case_documents`: Stores chunks with embeddings and search text
- `case_statistics`: View showing case-level statistics

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.document_injector import DocumentInjector

# Initialize the injector with cost tracking
injector = DocumentInjector(enable_cost_tracking=True)

# Process a single case folder
folder_id = "123456789"  # Your Box folder ID
results = injector.process_case_folder(folder_id)

# Check results
for result in results:
    print(f"{result.file_name}: {result.status} ({result.chunks_created} chunks)")

# Get cost report
cost_report = injector.get_cost_report()
print(f"Total cost: ${cost_report['costs']['total_cost']:.4f}")
```

### Process Multiple Cases

```python
# Process multiple case folders
folder_ids = ["123456789", "987654321", "456789123"]
all_results = injector.process_multiple_cases(folder_ids)

# Check statistics
stats = injector.deduplicator.get_statistics()
print(f"Total unique documents: {stats['total_unique_documents']}")
print(f"Total duplicates found: {stats['total_duplicate_instances']}")
```

### Hybrid Search

```python
from src.vector_storage import FullTextSearchManager, EmbeddingGenerator

# Initialize search components
search_manager = FullTextSearchManager()
embedding_gen = EmbeddingGenerator()

# Perform hybrid search
query = "patient diagnosis on January 15, 2023"
query_embedding = embedding_gen.generate_embedding(query)

results = search_manager.hybrid_search(
    case_name="Smith v. Jones",
    query_text=query,
    query_embedding=query_embedding,
    limit=20,
    vector_weight=0.7,  # 70% weight to semantic similarity
    text_weight=0.3     # 30% weight to keyword matching
)

# Display results
for result in results:
    print(f"Score: {result.combined_score:.3f}")
    print(f"Content: {result.content[:200]}...")
    print("---")
```

### Cost Tracking

```python
# After processing documents, view detailed cost report
cost_report = injector.cost_tracker.get_session_report()

# Print summary
injector.cost_tracker.print_summary()

# Save detailed report
report_path = injector.cost_tracker.save_report()

# Access specific information
print(f"Total tokens used: {cost_report['tokens']['total_tokens']:,}")
print(f"Embedding costs: ${cost_report['costs']['embedding_cost']:.4f}")
print(f"Context generation costs: ${cost_report['costs']['context_cost']:.4f}")

# View costs by case
for case, data in cost_report['costs_by_case'].items():
    print(f"{case}: ${data['cost']:.4f}")
```

### Test Connections

```python
from src.document_injector import test_connection

# Test all connections
test_connection()
```

### Command Line Usage

```bash
# Process a single folder
python -m src.document_injector --folder-id 123456789

# Process with limit (for testing)
python -m src.document_injector --folder-id 123456789 --max-documents 10
```

## Architecture

### Processing Pipeline

1. **Box Traversal**: Recursively finds all PDFs in folder and subfolders
2. **Duplicate Check**: Calculates SHA-256 hash and checks registry
3. **Text Extraction**: Uses multiple PDF libraries (pdfplumber, PyPDF2, pdfminer)
4. **Chunking**: Splits into ~1100 char chunks with semantic boundaries
5. **Context Generation**: LLM adds contextual summary to each chunk
6. **Embedding**: OpenAI text-embedding-3-small creates vectors
7. **Storage**: Vectors stored in Qdrant with case metadata

### Case Isolation

**Critical**: The system enforces strict case isolation through:

- Case name stored in every record
- Metadata filtering on all queries
- Verification checks after storage
- Logging of any isolation breaches

### Error Handling

- Retries for API calls (3 attempts with exponential backoff)
- Fallback extraction methods for PDFs
- Comprehensive error logging
- Graceful handling of failures (continues processing other documents)

## Monitoring

### Logging

The system uses Python's logging module with detailed logs:

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

# Or configure file logging
logging.basicConfig(
    level=logging.INFO,
    filename='clerk_injector.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Statistics

Get processing statistics:

```python
# During processing
injector.stats  # Real-time statistics

# Overall deduplication stats
dedup_stats = injector.deduplicator.get_statistics()

# Case-specific stats
case_stats = injector.vector_store.get_case_statistics("Smith v. Jones")
```

## Troubleshooting

### Common Issues

1. **Box Authentication Failed**
   - Verify JWT credentials in .env
   - Check Box app configuration
   - Ensure service account has folder access

2. **Qdrant Connection Error**
   - Verify Qdrant URL and keys
   - Check if tables are created
   - Ensure pgvector extension is enabled

3. **OpenAI Rate Limits**
   - Reduce batch sizes
   - Add delays between batches
   - Upgrade OpenAI tier if needed

4. **Memory Issues with Large PDFs**
   - Process fewer documents at once
   - Increase system memory
   - Enable document size limits

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Initial Import**: ~4-6 hours for 4,000-6,000 documents
- **Chunk Processing**: ~10-15 seconds per document (depends on size)
- **Vector Search**: <100ms for most queries
- **Storage**: ~2GB for 240,000 vectors

## Security Notes

- Never commit .env file
- Use service role key only for admin operations
- Rotate API keys regularly
- Monitor for case isolation breaches
- Log access patterns for audit trail

## Next Steps

After document injection:
1. Test vector search functionality
2. Implement motion drafting agents
3. Add deadline tracking
4. Create user interface with Open WebUI