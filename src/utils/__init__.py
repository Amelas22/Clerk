"""
Utility modules for Clerk legal AI system.
"""

from .cost_tracker import CostTracker, TokenUsage, DocumentCost

__all__ = [
    "CostTracker",
    "TokenUsage", 
    "DocumentCost"
]

# Optional Excel reporting (requires pandas)
try:
    from .cost_report_excel import ExcelCostReporter
    __all__.append("ExcelCostReporter")
except ImportError:
    pass