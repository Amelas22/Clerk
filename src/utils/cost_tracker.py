"""
Cost tracking module for monitoring API usage and expenses.
Tracks tokens used and costs for OpenAI API calls during document processing.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Represents token usage for a single API call"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    operation: str  # "embedding", "context_generation", etc.
    
    def calculate_cost(self, pricing: Dict[str, float]) -> float:
        """Calculate cost based on model pricing"""
        if self.model not in pricing:
            logger.warning(f"No pricing found for model {self.model}")
            return 0.0
        
        model_pricing = pricing[self.model]
        
        if self.operation == "embedding":
            # Embeddings are priced per token
            return (self.total_tokens / 1000) * model_pricing.get("per_1k_tokens", 0)
        else:
            # Chat models have separate input/output pricing
            input_cost = (self.prompt_tokens / 1000) * model_pricing.get("input_per_1k", 0)
            output_cost = (self.completion_tokens / 1000) * model_pricing.get("output_per_1k", 0)
            return input_cost + output_cost

@dataclass
class DocumentCost:
    """Tracks costs for processing a single document"""
    document_name: str
    document_id: str
    case_name: str
    chunks_processed: int = 0
    embedding_calls: List[TokenUsage] = field(default_factory=list)
    context_calls: List[TokenUsage] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    processing_time: Optional[float] = None
    
    def add_usage(self, usage: TokenUsage, pricing: Dict[str, float]):
        """Add token usage and update costs"""
        if usage.operation == "embedding":
            self.embedding_calls.append(usage)
        elif usage.operation == "context_generation":
            self.context_calls.append(usage)
        
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.calculate_cost(pricing)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of document processing costs"""
        return {
            "document_name": self.document_name,
            "document_id": self.document_id,
            "case_name": self.case_name,
            "chunks_processed": self.chunks_processed,
            "api_calls": {
                "embeddings": len(self.embedding_calls),
                "context_generation": len(self.context_calls)
            },
            "tokens": {
                "embedding_tokens": sum(u.total_tokens for u in self.embedding_calls),
                "context_tokens": sum(u.total_tokens for u in self.context_calls),
                "total": self.total_tokens
            },
            "cost": {
                "embedding_cost": sum(u.calculate_cost(OPENAI_PRICING) for u in self.embedding_calls),
                "context_cost": sum(u.calculate_cost(OPENAI_PRICING) for u in self.context_calls),
                "total": self.total_cost
            },
            "processing_time_seconds": self.processing_time
        }

# OpenAI Pricing (as of June 2025)
OPENAI_PRICING = {
    "text-embedding-3-small": {
        "per_1k_tokens": 0.00002  # $0.00002 per 1K tokens
    },
    "text-embedding-3-large": {
        "per_1k_tokens": 0.00013  # $0.00013 per 1K tokens
    },
    "gpt-4.1": {
        "input_per_1k": 0.002,   # $0.0005 per 1K input tokens
        "output_per_1k": 0.008   # $0.0015 per 1K output tokens
    },
    "gpt-4.1-mini": {
        "input_per_1k": 0.0004,     # $0.03 per 1K input tokens
        "output_per_1k": 0.0016     # $0.06 per 1K output tokens
    },
    "gpt-4.1-nano": {
        "input_per_1k": 0.0001,     # $0.01 per 1K input tokens
        "output_per_1k": 0.0004     # $0.03 per 1K output tokens
    },
    "gpt-4o": {
        "input_per_1k": 0.0025,     # $0.01 per 1K input tokens
        "output_per_1k": 0.01     # $0.03 per 1K output tokens
    },
    "o3": {
        "input_per_1k": 0.002,     # $0.01 per 1K input tokens
        "output_per_1k": 0.008     # $0.03 per 1K output tokens
    },
    "o4-mini": {
        "input_per_1k": 0.0011,     # $0.01 per 1K input tokens
        "output_per_1k": 0.0044     # $0.03 per 1K output tokens
    }
}

class CostTracker:
    """Tracks API costs across document processing operations"""
    
    def __init__(self, pricing: Dict[str, float] = None):
        """Initialize cost tracker
        
        Args:
            pricing: Custom pricing dictionary (uses OPENAI_PRICING by default)
        """
        self.pricing = pricing or OPENAI_PRICING
        self.documents: Dict[str, DocumentCost] = {}
        self.current_session_start = datetime.utcnow()
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
    def start_document(self, document_name: str, document_id: str, case_name: str) -> DocumentCost:
        """Start tracking a new document
        
        Args:
            document_name: Name of the document
            document_id: Unique document identifier
            case_name: Case this document belongs to
            
        Returns:
            DocumentCost object for tracking
        """
        doc_cost = DocumentCost(
            document_name=document_name,
            document_id=document_id,
            case_name=case_name
        )
        self.documents[document_id] = doc_cost
        return doc_cost
    
    def track_embedding_usage(self, document_id: str, tokens: int, model: str = "text-embedding-3-small"):
        """Track embedding API usage
        
        Args:
            document_id: Document being processed
            tokens: Number of tokens used
            model: Model used for embedding
        """
        if document_id not in self.documents:
            logger.warning(f"Document {document_id} not being tracked")
            return
        
        usage = TokenUsage(
            prompt_tokens=tokens,
            completion_tokens=0,
            total_tokens=tokens,
            model=model,
            operation="embedding"
        )
        
        self.documents[document_id].add_usage(usage, self.pricing)
    
    def track_context_usage(self, document_id: str, prompt_tokens: int, 
                          completion_tokens: int, model: str = "gpt-3.5-turbo"):
        """Track context generation API usage
        
        Args:
            document_id: Document being processed
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens generated
            model: Model used for context generation
        """
        if document_id not in self.documents:
            logger.warning(f"Document {document_id} not being tracked")
            return
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            operation="context_generation"
        )
        
        self.documents[document_id].add_usage(usage, self.pricing)
    
    def finish_document(self, document_id: str, chunks_processed: int, processing_time: float):
        """Mark document processing as complete
        
        Args:
            document_id: Document that finished processing
            chunks_processed: Number of chunks created
            processing_time: Time taken in seconds
        """
        if document_id not in self.documents:
            return
        
        self.documents[document_id].chunks_processed = chunks_processed
        self.documents[document_id].processing_time = processing_time
    
    def get_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost report for the session
        
        Returns:
            Detailed cost report
        """
        total_documents = len(self.documents)
        successful_documents = sum(1 for d in self.documents.values() if d.chunks_processed > 0)
        
        # Calculate totals by operation
        total_embedding_tokens = sum(
            sum(u.total_tokens for u in d.embedding_calls) 
            for d in self.documents.values()
        )
        total_context_tokens = sum(
            sum(u.total_tokens for u in d.context_calls) 
            for d in self.documents.values()
        )
        
        total_embedding_cost = sum(
            sum(u.calculate_cost(self.pricing) for u in d.embedding_calls) 
            for d in self.documents.values()
        )
        total_context_cost = sum(
            sum(u.calculate_cost(self.pricing) for u in d.context_calls) 
            for d in self.documents.values()
        )
        
        # Group by case
        costs_by_case = defaultdict(lambda: {"documents": 0, "tokens": 0, "cost": 0.0})
        for doc in self.documents.values():
            costs_by_case[doc.case_name]["documents"] += 1
            costs_by_case[doc.case_name]["tokens"] += doc.total_tokens
            costs_by_case[doc.case_name]["cost"] += doc.total_cost
        
        # Document details
        document_details = [doc.get_summary() for doc in self.documents.values()]
        
        # Sort documents by cost (highest first)
        document_details.sort(key=lambda x: x["cost"]["total"], reverse=True)
        
        report = {
            "session_id": self.session_id,
            "session_start": self.current_session_start.isoformat(),
            "session_duration": (datetime.utcnow() - self.current_session_start).total_seconds(),
            "summary": {
                "total_documents": total_documents,
                "successful_documents": successful_documents,
                "total_chunks": sum(d.chunks_processed for d in self.documents.values()),
                "total_api_calls": sum(
                    len(d.embedding_calls) + len(d.context_calls) 
                    for d in self.documents.values()
                )
            },
            "tokens": {
                "embedding_tokens": total_embedding_tokens,
                "context_tokens": total_context_tokens,
                "total_tokens": total_embedding_tokens + total_context_tokens
            },
            "costs": {
                "embedding_cost": round(total_embedding_cost, 4),
                "context_cost": round(total_context_cost, 4),
                "total_cost": round(total_embedding_cost + total_context_cost, 4),
                "average_per_document": round(
                    (total_embedding_cost + total_context_cost) / max(total_documents, 1), 4
                )
            },
            "costs_by_case": dict(costs_by_case),
            "top_5_expensive_documents": document_details[:5],
            "all_documents": document_details,
            "pricing_used": self.pricing
        }
        
        return report
    
    def save_report(self, filepath: str = None, excel: bool = True):
        """Save cost report to file
        
        Args:
            filepath: Path to save report (defaults to logs/cost_report_TIMESTAMP.json)
            excel: Also generate Excel report
        """
        if not filepath:
            filepath = f"logs/cost_report_{self.session_id}.json"
        
        report = self.get_session_report()
        
        # Save JSON report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cost report saved to {filepath}")
        
        # Generate Excel report if requested
        if excel:
            try:
                from utils.cost_report_excel import ExcelCostReporter
                excel_reporter = ExcelCostReporter(report)
                excel_path = excel_reporter.generate_excel_report()
                logger.info(f"Excel report saved to {excel_path}")
            except ImportError:
                logger.warning("Excel reporting requires pandas and openpyxl")
            except Exception as e:
                logger.error(f"Failed to generate Excel report: {str(e)}")
        
        return filepath
    
    def print_summary(self):
        """Print a human-readable cost summary"""
        report = self.get_session_report()
        
        print("\n" + "="*60)
        print("API COST TRACKING REPORT")
        print("="*60)
        print(f"Session ID: {report['session_id']}")
        print(f"Duration: {report['session_duration']:.2f} seconds")
        print(f"\nDocuments Processed: {report['summary']['total_documents']}")
        print(f"Total Chunks Created: {report['summary']['total_chunks']}")
        print(f"Total API Calls: {report['summary']['total_api_calls']}")
        
        print(f"\nToken Usage:")
        print(f"  Embeddings: {report['tokens']['embedding_tokens']:,} tokens")
        print(f"  Context Generation: {report['tokens']['context_tokens']:,} tokens")
        print(f"  Total: {report['tokens']['total_tokens']:,} tokens")
        
        print(f"\nCosts:")
        print(f"  Embeddings: ${report['costs']['embedding_cost']:.4f}")
        print(f"  Context Generation: ${report['costs']['context_cost']:.4f}")
        print(f"  TOTAL: ${report['costs']['total_cost']:.4f}")
        print(f"  Average per document: ${report['costs']['average_per_document']:.4f}")
        
        if report['costs_by_case']:
            print(f"\nCosts by Case:")
            for case, data in report['costs_by_case'].items():
                print(f"  {case}: ${data['cost']:.4f} ({data['documents']} docs, {data['tokens']:,} tokens)")
        
        if report['top_5_expensive_documents']:
            print(f"\nTop 5 Most Expensive Documents:")
            for i, doc in enumerate(report['top_5_expensive_documents'], 1):
                print(f"  {i}. {doc['document_name']}: ${doc['cost']['total']:.4f} ({doc['tokens']['total']:,} tokens)")
        
        print("="*60 + "\n")