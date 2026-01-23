"""
Quick test to verify the system works.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.invoice_extractor.core.pipeline import ExtractionPipeline
from src.invoice_extractor.utils.data_loader import DataLoader

def test_extraction():
    """Test extraction on a sample invoice"""
    print("Running quick system test...")

    # Check data exists
    data_path = "data/train-00000-of-00001-07d07b95f758bb43.parquet"
    if not Path(data_path).exists():
        print(f"Can't find data at {data_path}")
        return False

    # Load dependencies
    print("\nChecking dependencies...")
    try:
        print("looks good")
    except ImportError as e:
        print(f"missing: {e}")
        print("try: poetry install")
        return False

    # Load sample
    print("\nLoading a sample invoice...")
    try:
        loader = DataLoader(data_path)
        sample = loader[0]
        print(f"got {len(loader)} samples total")
    except Exception as e:
        print(f"failed: {e}")
        return False

    # Run extraction
    print("\nRunning extraction...")
    try:
        pipeline = ExtractionPipeline()
        result = pipeline.process(
            sample.image_bytes, document_id="test_sample", use_llm_fallback=False
        )
        print(f"done in {result.processing_time_ms:.0f}ms")
    except Exception as e:
        print(f"something went wrong: {e}")
        return False

    # Check output
    print("\nBuilding output...")
    output = {
        "document_id": result.document_id,
        "overall_confidence": round(result.overall_confidence, 3),
        "hitl_decision": result.hitl_decision.value,
        "needs_human_review": result.hitl_decision.value != "auto_accept",
        "fields": {
            name: {
                "value": f.value,
                "confidence": round(f.confidence, 3),
                "needs_review": f.needs_review or f.confidence < 0.6,
            }
            for name, f in result.fields.items()
        },
        "line_items": [
            {
                "description": item.description,
                "quantity": item.quantity,
                "total": item.total,
                "confidence": round(item.confidence, 3),
            }
            for item in result.line_items
        ],
    }

    # Print result
    print("\n--- Extraction output ---")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Summary
    print("\n--- Quick summary ---")
    print(f"doc: {result.document_id}")
    print(f"confidence: {result.overall_confidence:.1%}")
    print(f"decision: {result.hitl_decision.value}")
    extracted_count = sum(1 for f in result.fields.values() if f.value)
    print(f"extracted {extracted_count} of {len(result.fields)} fields")
    print(f"found {len(result.line_items)} line items")
    low_conf = sum(1 for f in result.fields.values() if f.value and f.confidence < 0.6)
    if low_conf:
        print(f"{low_conf} fields need review")

    print("\nCompleted!")
    return True


if __name__ == "__main__":
    success = test_extraction()
    exit(0 if success else 1)
