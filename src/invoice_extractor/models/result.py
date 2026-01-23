"""
Data models for extraction results.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HITLDecision(Enum):
    AUTO_ACCEPT = "auto_accept"
    NEEDS_REVIEW = "needs_review"
    REJECT = "reject"


@dataclass
class LineItem:
    """single line item from invoice"""

    description: Optional[str]
    quantity: Optional[str]
    unit_price: Optional[str]
    total: Optional[str]
    confidence: float = 0.0
    needs_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total": self.total,
            "confidence": round(self.confidence, 3),
            "needs_review": self.needs_review,
        }


@dataclass
class FieldResult:
    """single extracted field with confidence info"""

    field_name: str
    value: Optional[str]
    confidence: float
    extraction_method: str = "pattern"
    needs_review: bool = False
    review_reason: Optional[str] = None
    validated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "method": self.extraction_method,
            "validated": self.validated,
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
        }


@dataclass
class ExtractionResult:
    """complete extraction result for one invoice"""

    document_id: str
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    line_items: List[LineItem] = field(default_factory=list)
    overall_confidence: float = 0.0
    hitl_decision: HITLDecision = HITLDecision.NEEDS_REVIEW
    hitl_reasons: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    raw_ocr_text: Optional[str] = None

    def add_field(self, result: FieldResult):
        self.fields[result.field_name] = result

    def get_value(self, field_name: str) -> Optional[str]:
        if field_name in self.fields:
            return self.fields[field_name].value
        return None

    def compute_overall_confidence(self):
        """weighted average of field confidences"""
        if not self.fields:
            self.overall_confidence = 0.0
            return

        critical = {"Summe", "IBAN", "Rechnungsnummer"}
        total_weight = 0.0
        weighted_sum = 0.0

        for name, fld in self.fields.items():
            if fld.value is None:
                continue
            weight = 1.5 if name in critical else 1.0
            weighted_sum += fld.confidence * weight
            total_weight += weight

        self.overall_confidence = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "line_items": [item.to_dict() for item in self.line_items],
            "overall_confidence": round(self.overall_confidence, 3),
            "hitl_decision": self.hitl_decision.value,
            "hitl_reasons": self.hitl_reasons,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_simple_dict(self) -> Dict[str, Optional[str]]:
        """just the field values, no metadata"""
        return {k: v.value for k, v in self.fields.items()}
