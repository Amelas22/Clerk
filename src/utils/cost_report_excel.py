"""
Excel report generator for cost tracking data.
Creates detailed spreadsheets for cost analysis.
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class ExcelCostReporter:
    """Generates Excel reports from cost tracking data"""
    
    def __init__(self, cost_report: Dict[str, Any]):
        """Initialize with cost report data
        
        Args:
            cost_report: Cost report dictionary from CostTracker
        """
        self.report = cost_report
        
    def generate_excel_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive Excel report with multiple sheets
        
        Args:
            output_path: Path for Excel file (auto-generated if not provided)
            
        Returns:
            Path to generated Excel file
        """
        if not output_path:
            os.makedirs("reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/cost_report_{timestamp}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Summary sheet
            self._create_summary_sheet(writer)
            
            # 2. Document details sheet
            self._create_documents_sheet(writer)
            
            # 3. Case summary sheet
            self._create_case_summary_sheet(writer)
            
            # 4. API usage breakdown
            self._create_api_usage_sheet(writer)
            
            # 5. Pricing information
            self._create_pricing_sheet(writer)
        
        logger.info(f"Excel report generated: {output_path}")
        return output_path
    
    def _create_summary_sheet(self, writer):
        """Create summary overview sheet"""
        summary_data = {
            'Metric': [
                'Session ID',
                'Session Start',
                'Session Duration (seconds)',
                'Total Documents Processed',
                'Successful Documents',
                'Total Chunks Created',
                'Total API Calls',
                'Total Tokens Used',
                'Total Embedding Tokens',
                'Total Context Tokens',
                'Total Cost ($)',
                'Average Cost per Document ($)',
                'Embedding Cost ($)',
                'Context Generation Cost ($)'
            ],
            'Value': [
                self.report['session_id'],
                self.report['session_start'],
                f"{self.report['session_duration']:.2f}",
                self.report['summary']['total_documents'],
                self.report['summary']['successful_documents'],
                self.report['summary']['total_chunks'],
                self.report['summary']['total_api_calls'],
                f"{self.report['tokens']['total_tokens']:,}",
                f"{self.report['tokens']['embedding_tokens']:,}",
                f"{self.report['tokens']['context_tokens']:,}",
                f"{self.report['costs']['total_cost']:.4f}",
                f"{self.report['costs']['average_per_document']:.4f}",
                f"{self.report['costs']['embedding_cost']:.4f}",
                f"{self.report['costs']['context_cost']:.4f}"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the sheet
        worksheet = writer.sheets['Summary']
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 40
    
    def _create_documents_sheet(self, writer):
        """Create detailed document breakdown sheet"""
        documents = self.report['all_documents']
        
        # Prepare data for DataFrame
        doc_data = []
        for doc in documents:
            doc_data.append({
                'Document Name': doc['document_name'],
                'Case Name': doc['case_name'],
                'Chunks Processed': doc['chunks_processed'],
                'Embedding Calls': doc['api_calls']['embeddings'],
                'Context Calls': doc['api_calls']['context_generation'],
                'Total Tokens': doc['tokens']['total'],
                'Embedding Tokens': doc['tokens']['embedding_tokens'],
                'Context Tokens': doc['tokens']['context_tokens'],
                'Total Cost ($)': doc['cost']['total'],
                'Embedding Cost ($)': doc['cost']['embedding_cost'],
                'Context Cost ($)': doc['cost']['context_cost'],
                'Processing Time (s)': doc.get('processing_time_seconds', 0)
            })
        
        df = pd.DataFrame(doc_data)
        
        # Sort by cost descending
        df = df.sort_values('Total Cost ($)', ascending=False)
        
        df.to_excel(writer, sheet_name='Document Details', index=False)
        
        # Format columns
        worksheet = writer.sheets['Document Details']
        for col in ['A', 'B']:
            worksheet.column_dimensions[col].width = 25
        for col in ['C', 'D', 'E', 'F', 'G', 'H']:
            worksheet.column_dimensions[col].width = 15
        for col in ['I', 'J', 'K', 'L']:
            worksheet.column_dimensions[col].width = 18
    
    def _create_case_summary_sheet(self, writer):
        """Create case-level summary sheet"""
        case_data = []
        
        for case_name, data in self.report['costs_by_case'].items():
            case_data.append({
                'Case Name': case_name,
                'Documents Processed': data['documents'],
                'Total Tokens': f"{data['tokens']:,}",
                'Total Cost ($)': data['cost'],
                'Average Cost per Doc ($)': data['cost'] / max(data['documents'], 1)
            })
        
        df = pd.DataFrame(case_data)
        
        # Sort by cost descending
        df = df.sort_values('Total Cost ($)', ascending=False)
        
        df.to_excel(writer, sheet_name='Case Summary', index=False)
        
        # Format columns
        worksheet = writer.sheets['Case Summary']
        worksheet.column_dimensions['A'].width = 30
        for col in ['B', 'C', 'D', 'E']:
            worksheet.column_dimensions[col].width = 20
    
    def _create_api_usage_sheet(self, writer):
        """Create API usage breakdown sheet"""
        # Aggregate API usage by model
        model_usage = {}
        
        for doc in self.report['all_documents']:
            # Embedding model (assuming consistent)
            embed_model = 'text-embedding-3-small'
            if embed_model not in model_usage:
                model_usage[embed_model] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost': 0.0
                }
            model_usage[embed_model]['calls'] += doc['api_calls']['embeddings']
            model_usage[embed_model]['tokens'] += doc['tokens']['embedding_tokens']
            model_usage[embed_model]['cost'] += doc['cost']['embedding_cost']
            
            # Context model (could vary)
            # Assuming gpt-3.5-turbo for now
            context_model = 'gpt-3.5-turbo'
            if context_model not in model_usage:
                model_usage[context_model] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost': 0.0
                }
            model_usage[context_model]['calls'] += doc['api_calls']['context_generation']
            model_usage[context_model]['tokens'] += doc['tokens']['context_tokens']
            model_usage[context_model]['cost'] += doc['cost']['context_cost']
        
        # Convert to DataFrame
        usage_data = []
        for model, data in model_usage.items():
            usage_data.append({
                'Model': model,
                'API Calls': data['calls'],
                'Total Tokens': f"{data['tokens']:,}",
                'Total Cost ($)': f"{data['cost']:.4f}",
                'Average Tokens per Call': f"{data['tokens'] / max(data['calls'], 1):.0f}"
            })
        
        df = pd.DataFrame(usage_data)
        df.to_excel(writer, sheet_name='API Usage', index=False)
        
        # Format columns
        worksheet = writer.sheets['API Usage']
        worksheet.column_dimensions['A'].width = 25
        for col in ['B', 'C', 'D', 'E']:
            worksheet.column_dimensions[col].width = 20
    
    def _create_pricing_sheet(self, writer):
        """Create pricing reference sheet"""
        pricing_data = []
        
        for model, pricing in self.report['pricing_used'].items():
            if 'per_1k_tokens' in pricing:
                pricing_data.append({
                    'Model': model,
                    'Type': 'Embedding',
                    'Price per 1K Tokens ($)': pricing['per_1k_tokens'],
                    'Price per 1M Tokens ($)': pricing['per_1k_tokens'] * 1000
                })
            else:
                pricing_data.append({
                    'Model': model,
                    'Type': 'Chat - Input',
                    'Price per 1K Tokens ($)': pricing.get('input_per_1k', 0),
                    'Price per 1M Tokens ($)': pricing.get('input_per_1k', 0) * 1000
                })
                pricing_data.append({
                    'Model': model,
                    'Type': 'Chat - Output',
                    'Price per 1K Tokens ($)': pricing.get('output_per_1k', 0),
                    'Price per 1M Tokens ($)': pricing.get('output_per_1k', 0) * 1000
                })
        
        df = pd.DataFrame(pricing_data)
        df.to_excel(writer, sheet_name='Pricing Reference', index=False)
        
        # Format columns
        worksheet = writer.sheets['Pricing Reference']
        worksheet.column_dimensions['A'].width = 25
        for col in ['B', 'C', 'D']:
            worksheet.column_dimensions[col].width = 20


def generate_cost_comparison_report(reports: Dict[str, Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """Generate comparison report across multiple processing sessions
    
    Args:
        reports: Dictionary mapping session names to cost reports
        output_path: Output file path
        
    Returns:
        Path to generated Excel file
    """
    if not output_path:
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/cost_comparison_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Comparison summary
        comparison_data = []
        
        for session_name, report in reports.items():
            comparison_data.append({
                'Session': session_name,
                'Date': report['session_start'],
                'Documents': report['summary']['total_documents'],
                'Total Tokens': report['tokens']['total_tokens'],
                'Total Cost ($)': report['costs']['total_cost'],
                'Avg Cost/Doc ($)': report['costs']['average_per_document']
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_excel(writer, sheet_name='Session Comparison', index=False)
    
    return output_path