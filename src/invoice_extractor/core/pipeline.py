"""
Invoice Extraction Pipeline

Smart extraction that minimizes LLM usage while maximizing accuracy:
stage 1: Pattern Extraction
stage 2: LLM for Low-Confidence Fields
stage 3: Full LLM Extraction (RARE)
"""

import io
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from ..config import Config
from ..models.result import ExtractionResult, FieldResult, HITLDecision
from .doc_aligner import DocAligner
from .doctr_ocr import DocTROCR
from .extractor import InvoiceExtractor
from .llm_extractor import LLMExtractor
from .validators import InvoiceValidator

# All possible invoice fields
ALL_FIELDS = [
    "Summe",
    "Rechnungsdatum",
    "Rechnungsnummer",
    "Der Name der Firma",
    "Die Adresse der Firma",
    "IBAN",
    "Falligkeitsdatum",
    "Telefonnummer",
    "Der Name der Bank",
]

# Confidence thresholds
FIELD_HIGH_CONFIDENCE = 0.70  # Above this = good extraction
FIELD_LOW_CONFIDENCE = 0.50  # Below this = needs LLM
DOCUMENT_AUTO_ACCEPT = 0.75  # Overall confidence for auto-accept
LLM_TRIGGER_RATIO = 0.3  # If >30% fields are bad, use full LLM


class ExtractionPipeline:
    """
    Tiered invoice extraction pipeline.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._ocr = None
        self._extractor = None
        self._validator = None
        self._llm = None
        self._aligner = None

    @property
    def ocr(self):
        if self._ocr is None:
            cache = self.config.cache_dir if self.config.enable_cache else None
            self._ocr = DocTROCR(
                det_arch=self.config.ocr.det_arch,
                rec_arch=self.config.ocr.rec_arch,
                use_gpu=self.config.ocr.use_gpu,
                cache_dir=cache,
            )
        return self._ocr

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = InvoiceExtractor()
        return self._extractor

    @property
    def validator(self):
        if self._validator is None:
            self._validator = InvoiceValidator()
        return self._validator

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMExtractor(
                api_key=self.config.llm.api_key, model=self.config.llm.model
            )
        return self._llm

    @property
    def aligner(self):
        if self._aligner is None and self.config.aligner_path:
            try:
                self._aligner = DocAligner(self.config.aligner_path)
            except Exception:
                pass
        return self._aligner

    def process(
        self,
        image,
        document_id: str = None,
        include_text: bool = False,
        use_llm_fallback: bool = True,
    ) -> ExtractionResult:
        """
        Extract fields from invoice image
        """
        start = time.time()

        # Generate document ID
        if document_id is None:
            if isinstance(image, (str, Path)):
                document_id = Path(image).stem
            else:
                document_id = f"doc_{int(time.time() * 1000)}"

        result = ExtractionResult(document_id=document_id)

        try:
            # Load and align image
            img = self._load_image(image)

            if self.aligner:
                try:
                    aligned, conf = self.aligner.align(img)
                    if conf > 0.3:
                        img = aligned
                except Exception:
                    pass

            # OCR extraction
            text, ocr_conf = self.ocr.extract_text(img)
            if include_text:
                result.raw_ocr_text = text

            if not text.strip():
                result.hitl_reasons.append("OCR returned empty text")
                result.hitl_decision = HITLDecision.REJECT
                result.processing_time_ms = (time.time() - start) * 1000
                return result

            # Pattern extraction
            pattern_fields, low_conf_fields = self._stage1_pattern_extraction(
                text, ocr_conf
            )

            # Add pattern results to result
            for name, field in pattern_fields.items():
                result.add_field(field)

            # Extract line items
            result.line_items = self.extractor.extract_line_items(text)

            # Decide if LLM is needed
            fields_with_values = [f for f in pattern_fields.values() if f.value]
            num_low_conf = len(low_conf_fields)
            num_extracted = len(fields_with_values)

            need_llm = False
            llm_mode = None

            if use_llm_fallback and self.llm.is_available():
                if num_extracted == 0:
                    # No fields extracted
                    need_llm = True
                    llm_mode = "full"
                elif num_low_conf > 0:
                    # Some fields need help
                    ratio = num_low_conf / max(num_extracted, 1)
                    if ratio > LLM_TRIGGER_RATIO:
                        need_llm = True
                        llm_mode = "full"
                    else:
                        need_llm = True
                        llm_mode = "targeted"

            # LLM extraction if needed
            if need_llm:
                if llm_mode == "full":
                    # Full LLM extraction
                    self._stage3_full_llm(result, text)
                else:
                    # Targeted LLM for specific fields
                    self._stage2_targeted_llm(result, text, low_conf_fields)

            # Final validation and HITL routing
            self._finalize_result(result)

        except Exception as e:
            result.hitl_reasons.append(f"Processing error: {str(e)}")
            result.hitl_decision = HITLDecision.REJECT

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    def _stage1_pattern_extraction(
        self, text: str, ocr_conf: float
    ) -> Tuple[Dict[str, FieldResult], List[str]]:
        """
        stage 1: Fast pattern-based extraction.
        """
        extracted = self.extractor.extract_all(text)
        fields = {}
        low_conf_fields = []

        for name, field in extracted.items():
            if field.value is None:
                fields[name] = field
                continue

            # Validate and calculate confidence
            validation = self.validator.validate_field(name, field.value)
            base_conf = field.confidence * validation.confidence_adjustment
            final_conf = base_conf * 0.7 + ocr_conf * 0.3

            if validation.is_valid and validation.corrected_value:
                field.value = validation.corrected_value

            field.confidence = final_conf
            field.validated = validation.is_valid

            # Track low-confidence fields
            if final_conf < FIELD_LOW_CONFIDENCE or not validation.is_valid:
                low_conf_fields.append(name)
                field.needs_review = True
                field.review_reason = f"Low confidence ({final_conf:.0%})"

            fields[name] = field

        return fields, low_conf_fields

    def _stage2_targeted_llm(
        self, result: ExtractionResult, text: str, target_fields: List[str]
    ) -> None:
        """
        Stage 2: LLM extraction for specific low-confidence fields only.
        """
        try:
            existing = {
                name: field.value
                for name, field in result.fields.items()
                if field.value and name not in target_fields  # Exclude target fields!
            }

            llm_result = self.llm.extract(
                text, target_fields=target_fields, existing_values=existing
            )

            if not llm_result.success:
                return

            for name, llm_field in llm_result.fields.items():
                if not llm_field.value:
                    continue

                existing_field = result.fields.get(name)
                existing_conf = (
                    existing_field.confidence
                    if existing_field and existing_field.value
                    else 0
                )

                if llm_field.confidence > existing_conf:
                    result.fields[name] = FieldResult(
                        field_name=name,
                        value=llm_field.value,
                        confidence=llm_field.confidence,
                        extraction_method="llm_targeted",
                        validated=llm_field.validated,
                        needs_review=llm_field.confidence < FIELD_LOW_CONFIDENCE,
                    )

        except Exception:
            pass

    def _stage3_full_llm(self, result: ExtractionResult, text: str) -> None:
        """
        Stage 3: Full LLM extraction when pattern extraction fails badly.
        """
        try:
            llm_result = self.llm.extract(text)

            if not llm_result.success:
                return

            # Replace all fields with LLM results
            for name, llm_field in llm_result.fields.items():
                if not llm_field.value:
                    continue

                existing_field = result.fields.get(name)
                existing_conf = (
                    existing_field.confidence
                    if existing_field and existing_field.value
                    else 0
                )

                # Use LLM if it found something and is reasonably confident
                if llm_field.confidence > existing_conf or llm_field.confidence > 0.5:
                    result.fields[name] = FieldResult(
                        field_name=name,
                        value=llm_field.value,
                        confidence=llm_field.confidence,
                        extraction_method="llm_full",
                        validated=llm_field.validated,
                        needs_review=llm_field.confidence < FIELD_LOW_CONFIDENCE,
                    )

        except Exception:
            pass

    def _finalize_result(self, result: ExtractionResult) -> None:
        """
        Calculate overall confidence and determine HITL decision.
        """
        issues = []
        fields_with_values = []
        fields_needing_review = []

        for name, field in result.fields.items():
            if field.value:
                fields_with_values.append(field)
                if field.needs_review:
                    fields_needing_review.append(name)
                    issues.append(f"{name}: {field.review_reason or 'needs review'}")

        # Calculate overall confidence
        if fields_with_values:
            total_conf = sum(f.confidence for f in fields_with_values)
            result.overall_confidence = total_conf / len(fields_with_values)
        else:
            result.overall_confidence = 0.0
            issues.append("No fields extracted")

        result.hitl_reasons = issues

        # HITL decision
        num_fields = len(fields_with_values)
        num_review = len(fields_needing_review)

        if num_fields == 0:
            result.hitl_decision = HITLDecision.REJECT
        elif num_review == 0 and result.overall_confidence >= DOCUMENT_AUTO_ACCEPT:
            result.hitl_decision = HITLDecision.AUTO_ACCEPT
        elif num_review > num_fields / 2:
            result.hitl_decision = HITLDecision.REJECT
        else:
            result.hitl_decision = HITLDecision.NEEDS_REVIEW

    def _load_image(self, image) -> Image.Image:
        """Load image from various input types"""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        else:
            return Image.open(image)

    def process_batch(
        self, images, doc_ids: List[str] = None, use_llm_fallback: bool = True
    ) -> List[ExtractionResult]:
        """Process multiple images"""
        if doc_ids is None:
            doc_ids = [None] * len(images)
        return [
            self.process(img, did, use_llm_fallback=use_llm_fallback)
            for img, did in zip(images, doc_ids)
        ]
