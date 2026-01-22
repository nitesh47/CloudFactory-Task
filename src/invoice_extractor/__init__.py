"""
Invoice field extraction system for German invoices.
"""

from .core.pipeline import ExtractionPipeline
from .models.result import ExtractionResult, FieldResult

__version__ = "0.2.0"
__all__ = ["ExtractionPipeline", "ExtractionResult", "FieldResult"]