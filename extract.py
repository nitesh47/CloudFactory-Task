"""
Invoice Field Extraction
"""

import argparse
import json
import sys
from pathlib import Path

from src.invoice_extractor.core.pipeline import ExtractionPipeline


def extract_invoice(image_path: str, include_ocr_text: bool = False) -> dict:
    """
    Extract fields from an invoice image.
    """
    pipeline = ExtractionPipeline()
    result = pipeline.process(
        image_path,
        document_id=Path(image_path).stem,
        include_text=include_ocr_text,
        use_llm_fallback=True
    )

    # Build clean output structure
    output = {
        "document_id": result.document_id,
        "processing_time_ms": round(result.processing_time_ms, 2),

        # Overall document status
        "overall_confidence": round(result.overall_confidence, 3),
        "status": result.hitl_decision.value,
        "needs_human_review": result.hitl_decision.value != "auto_accept",

        # Extracted fields
        "extracted_fields": {},

        # Line items
        "line_items": [],

        # Fields flagged for human review
        "fields_for_review": [],

        # Summary of issues
        "review_reasons": result.hitl_reasons,
    }

    # Process each field
    for name, field in result.fields.items():
        if field.value is not None:
            # Field was extracted
            field_data = {
                "value": field.value,
                "confidence": round(field.confidence, 3),
                "needs_review": field.needs_review,
            }
            output["extracted_fields"][name] = field_data

            # Track fields needing review
            if field.needs_review:
                output["fields_for_review"].append({
                    "field": name,
                    "value": field.value,
                    "confidence": round(field.confidence, 3),
                    "reason": field.review_reason or "low confidence"
                })

    # Process line items
    for item in result.line_items:
        item_data = {
            "description": item.description,
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "total": item.total,
            "confidence": round(item.confidence, 3),
            "needs_review": item.needs_review
        }
        output["line_items"].append(item_data)

    # Add OCR text
    if include_ocr_text and result.raw_ocr_text:
        output["raw_ocr_text"] = result.raw_ocr_text

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract fields from invoice documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output includes:
- extracted_fields: Fields found with values and confidence scores
- overall_confidence: Average confidence across all extracted fields
- needs_human_review: Whether the document should be reviewed
- fields_for_review: Specific fields flagged for review
- line_items: Individual line items (if detected)

Examples:
    python extract.py invoice.png
    python extract.py invoice.png --pretty
    python extract.py invoice.png --output result.json
        """
    )
    parser.add_argument("image", help="Path to invoice image (PNG, JPG, PDF)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print JSON")
    parser.add_argument("--include-ocr", action="store_true", help="Include raw OCR text")

    args = parser.parse_args()

    # Validate input
    if not Path(args.image).exists():
        print(f"Error: File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Extract
    result = extract_invoice(args.image, include_ocr_text=args.include_ocr)

    # Output
    indent = 2 if args.pretty else None
    json_output = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(json_output)
        print(f"Output written to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()