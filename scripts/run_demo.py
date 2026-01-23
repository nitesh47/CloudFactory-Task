"""
Demo script for invoice extraction.
"""

import os
import sys
from pathlib import Path
from src.invoice_extractor.core.pipeline import ExtractionPipeline

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.invoice_extractor.utils.data_loader import DataLoader
from src.invoice_extractor.evaluation.metrics import Evaluator

def single_extraction(loader, idx=0, use_llm=False):
    """show detailed extraction for one invoice"""
    print(f"\n--- Extracting sample {idx} ---")

    sample = loader[idx]
    print(f"Ground truth has: {list(sample.fields.keys())}")

    pipeline = ExtractionPipeline()
    result = pipeline.process(
        sample.image_bytes, document_id=f"sample_{idx}", use_llm_fallback=use_llm
    )

    print(f"\nGot result for {result.document_id}:")
    conf = result.overall_confidence
    decision = result.hitl_decision.value
    print(f"confidence {conf:.2f}, decision: {decision}")
    print(f"took {result.processing_time_ms:.0f}ms")

    if result.hitl_reasons:
        print(f"  flagged issues: {result.hitl_reasons[:5]}")

    print(f"\nExtracted fields:")
    for name, field in result.fields.items():
        expected = sample.fields.get(name, "-")

        # check if values match
        match = "  "
        if field.value and expected != "-":
            got_clean = field.value.lower().replace(" ", "").replace("-", "")
            exp_clean = expected.lower().replace(" ", "").replace("-", "")
            if got_clean in exp_clean or exp_clean in got_clean:
                match = "ok"

        method = field.extraction_method[:6] if field.extraction_method else "?"
        validated = "V" if getattr(field, "validated", True) else "X"

        print(f"[{match}] {name}:")
        print(f"got: {field.value}")
        print(f"exp: {expected}")
        print(f"conf: {field.confidence:.2f} | {method} | {validated}")


def batch_evaluation(loader, limit=20, use_llm=False):
    """evaluate on multiple samples with proper metrics"""
    llm_status = "with LLM" if use_llm else "OCR only"
    print(f"\n--- Running batch eval on {limit} samples ({llm_status}) ---")

    pipeline = ExtractionPipeline()
    evaluator = Evaluator()

    results = []
    ground_truths = []

    print()
    for i, sample in enumerate(loader.iter_samples(limit=limit)):
        result = pipeline.process(
            sample.image_bytes, document_id=f"doc_{i}", use_llm_fallback=use_llm
        )
        results.append(result)
        ground_truths.append(sample.fields)
        if (i + 1) % 5 == 0:
            print(f"processed {i + 1} of {limit}")

    report = evaluator.evaluate(results, ground_truths)

    print("\n" + report.summary())

    # extraction method breakdown
    method_counts = {"pattern": 0, "llm": 0, "ocr_low_conf": 0, "other": 0}
    for r in results:
        for f in r.fields.values():
            method = f.extraction_method if f.value else None
            if method in method_counts:
                method_counts[method] += 1
            elif method:
                method_counts["other"] += 1

    print("\nMethods used:")
    for method, count in method_counts.items():
        if count > 0:
            print(f"  {method} - {count}")

    # threshold analysis
    print("\nThreshold check:")
    analysis = evaluator.analyze_thresholds(results, ground_truths)
    for thresh in sorted(analysis.keys()):
        stats = analysis[thresh]
        accept_rate = stats["auto_accept_rate"]
        acc = stats["accuracy"]
        print(f"at {thresh:.1f} -> accept {accept_rate:.1%}, accuracy {acc:.1%}")


def main():
    data_path = "data/train-00000-of-00001-07d07b95f758bb43.parquet"

    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        return

    loader = DataLoader(data_path)
    print(f"Got {len(loader)} samples from dataset")

    # check if LLM is available
    has_llm = bool(os.environ.get("OPENAI_API_KEY"))
    if has_llm:
        print("LLM is available (found OPENAI_API_KEY)")
    else:
        print("No LLM - set OPENAI_API_KEY if you want that")

    # single extraction demo
    single_extraction(loader, idx=2, use_llm=has_llm)

    # batch evaluation
    batch_evaluation(loader, limit=20, use_llm=has_llm)


if __name__ == "__main__":
    main()
