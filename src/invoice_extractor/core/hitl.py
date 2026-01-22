"""
Human-in-the-loop routing logic.
"""

import re

from ..config import HITLConfig
from ..models.result import ExtractionResult, HITLDecision


class HITLRouter:
    """
    Routes extractions to human review based on:
    - field confidence scores
    - validation failures
    - missing critical fields
    - extraction method (ocr vs llm)
    """

    def __init__(self, config: HITLConfig = None):
        self.config = config or HITLConfig()

    def evaluate(self, result: ExtractionResult) -> ExtractionResult:
        """
        evaluate extraction result and set hitl decision

        scoring:
        validated high-confidence fields are trusted
        unvalidated or low-confidence fields need review
        missing critical fields trigger review
        too many issues -> reject for manual entry
        """
        issues = []
        severe_issues = 0

        for name, field in result.fields.items():
            if field.value is None:
                if name in self.config.critical_fields:
                    issues.append(f"missing: {name}")
                    severe_issues += 1
                continue

            # check confidence
            if field.confidence < self.config.confidence_threshold:
                issues.append(f"low conf ({field.confidence:.2f}): {name}")
                field.needs_review = True
                field.review_reason = "low confidence"

                if name in self.config.critical_fields:
                    severe_issues += 1

            # check validation status
            if hasattr(field, 'validated') and not field.validated:
                issues.append(f"validation failed: {name}")
                field.needs_review = True
                field.review_reason = "validation failed"

                if name in self.config.critical_fields:
                    severe_issues += 1

            # additional checks for high confidence but potentially wrong
            format_error = self._check_format(name, field.value)
            if format_error:
                issues.append(format_error)
                field.needs_review = True
                field.review_reason = format_error

        # check all critical fields present
        for name in self.config.critical_fields:
            if name not in result.fields or result.fields[name].value is None:
                msg = f"missing critical: {name}"
                if msg not in issues and f"missing: {name}" not in issues:
                    issues.append(msg)
                    severe_issues += 1

        # compute overall confidence
        result.compute_overall_confidence()

        # make decision
        result.hitl_reasons = issues

        if severe_issues >= 3 or result.overall_confidence < 0.25:
            result.hitl_decision = HITLDecision.REJECT
        elif issues:
            result.hitl_decision = HITLDecision.NEEDS_REVIEW
        else:
            result.hitl_decision = HITLDecision.AUTO_ACCEPT

        return result

    def _check_format(self, field_name: str, value: str) -> str:
        """additional format validation for edge cases"""
        if not value:
            return ""

        if field_name == "IBAN":
            cleaned = value.replace(" ", "").upper()
            if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$", cleaned):
                return "iban format"
            if cleaned.startswith("DE") and len(cleaned) != 22:
                return "iban length"

        elif field_name in ("Rechnungsdatum", "Falligkeitsdatum"):
            if not re.match(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}$", value):
                return f"date format: {field_name}"

        elif field_name == "Summe":
            digits = re.sub(r"\D", "", value)
            if len(digits) < 1:
                return "amount format"

        elif field_name == "Telefonnummer":
            digits = re.sub(r"\D", "", value)
            if len(digits) < 6:
                return "phone too short"

        elif field_name == "Rechnungsnummer":
            wrong_words = ["datum", "summe", "bank", "telefon", "iban"]
            if value.lower() in wrong_words:
                return "wrong field content"

        return ""


class HITLQueue:
    """
    manages queue of documents needing human review
    """

    def __init__(self):
        self.pending = []
        self.completed = []

    def add(self, result: ExtractionResult):
        """add document to review queue"""
        self.pending.append({
            "document_id": result.document_id,
            "decision": result.hitl_decision.value,
            "reasons": result.hitl_reasons,
            "fields": {n: f.value for n, f in result.fields.items()},
            "confidence": result.overall_confidence
        })

    def get_next(self):
        """get next document for review"""
        if self.pending:
            return self.pending.pop(0)
        return None

    def submit_correction(self, document_id: str, corrections: dict):
        """submit human corrections"""
        self.completed.append({
            "document_id": document_id,
            "corrections": corrections
        })

    def get_stats(self) -> dict:
        """get queue statistics"""
        return {
            "pending": len(self.pending),
            "completed": len(self.completed)
        }