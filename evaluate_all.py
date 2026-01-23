"""
Full Dataset Evaluation
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from src.invoice_extractor.core.pipeline import ExtractionPipeline
from src.invoice_extractor.utils.data_loader import DataLoader
from src.invoice_extractor.evaluation.metrics import Evaluator


def run_evaluation(data_path: str, limit: int = None, output_dir: str = None):
    """
    Run evaluation on dataset.
    """
    # Load data
    loader = DataLoader(data_path)
    total_samples = len(loader)
    num_samples = min(limit, total_samples) if limit else total_samples

    print("\nStarting invoice extraction evaluation")
    print(f"Using dataset: {data_path}")
    print(f"Found {total_samples} samples, will process {num_samples}\n")

    # Initialize
    pipeline = ExtractionPipeline()
    evaluator = Evaluator()

    results = []
    ground_truths = []
    all_extractions = []

    # Process all samples
    start_time = time.time()
    print("Running extraction on invoices...")

    for i, sample in enumerate(loader.iter_samples(limit=num_samples)):
        result = pipeline.process(
            sample.image_bytes, document_id=f"doc_{i:04d}", use_llm_fallback=False
        )
        results.append(result)
        ground_truths.append(sample.fields)

        # Store detailed extraction for output
        extraction = {
            "document_id": result.document_id,
            "index": i,
            "overall_confidence": round(result.overall_confidence, 3),
            "hitl_decision": result.hitl_decision.value,
            "hitl_reasons": result.hitl_reasons,
            "processing_time_ms": round(result.processing_time_ms, 2),
            "fields": {
                name: {
                    "extracted": field.value,
                    "expected": sample.fields.get(name),
                    "confidence": round(field.confidence, 3),
                    "method": field.extraction_method,
                    "validated": field.validated,
                }
                for name, field in result.fields.items()
            },
            "line_items": [
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total": item.total,
                    "confidence": round(item.confidence, 3),
                }
                for item in result.line_items
            ],
        }
        all_extractions.append(extraction)

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"done {i + 1} of {num_samples} - {rate:.1f} per sec")

    total_time = time.time() - start_time

    # Evaluate
    print("\nNow computing metrics...")
    report = evaluator.evaluate(results, ground_truths)

    # HITL statistics
    hitl_stats = {
        "auto_accept": sum(
            1 for r in results if r.hitl_decision.value == "auto_accept"
        ),
        "needs_review": sum(
            1 for r in results if r.hitl_decision.value == "needs_review"
        ),
        "reject": sum(1 for r in results if r.hitl_decision.value == "reject"),
    }

    # Print results
    print("\n--- Results ---\n")

    avg_ms = (total_time / num_samples) * 1000
    print(
        f"Processed {num_samples} docs in {total_time:.1f}s (avg {avg_ms:.0f}ms each)"
    )

    print(f"\nRouting breakdown:")
    auto_pct = hitl_stats["auto_accept"] / num_samples
    review_pct = hitl_stats["needs_review"] / num_samples
    reject_pct = hitl_stats["reject"] / num_samples
    print(f"auto-accept: {hitl_stats['auto_accept']} ({auto_pct:.1%})")
    print(f"needs review: {hitl_stats['needs_review']} ({review_pct:.1%})")
    print(f"rejected: {hitl_stats['reject']} ({reject_pct:.1%})")

    print("\n" + report.summary())

    # Threshold analysis
    print("\nThreshold analysis:")
    analysis = evaluator.analyze_thresholds(results, ground_truths)
    for thresh in sorted(analysis.keys()):
        stats = analysis[thresh]
        accept_rate = stats["auto_accept_rate"]
        acc = stats["accuracy"]
        print(f"{thresh:.2f} -> accept {accept_rate:.1%}, accuracy {acc:.1%}")

    # Save results if output dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_path / f"extractions_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(all_extractions, f, indent=2, ensure_ascii=False)
        print(f"\nWrote detailed results to {results_file}")

        # Save summary
        summary = {
            "timestamp": timestamp,
            "dataset": data_path,
            "num_samples": num_samples,
            "total_time_seconds": round(total_time, 2),
            "avg_time_ms": round((total_time / num_samples) * 1000, 2),
            "hitl_stats": hitl_stats,
            "metrics": {
                "overall": {
                    "precision": report.overall_precision,
                    "recall": report.overall_recall,
                    "f1": report.overall_f1,
                },
                "per_field": {
                    name: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "exact_match": m.exact_matches,
                        "cer": m.avg_cer,
                    }
                    for name, m in report.field_metrics.items()
                },
            },
        }
        summary_file = output_path / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary at {summary_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on full invoice dataset"
    )
    parser.add_argument(
        "--data",
        "-d",
        default="data/train-00000-of-00001-07d07b95f758bb43.parquet",
        help="Path to parquet data file",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of samples (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results",
        help="Output directory for results (default: results/)",
    )

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Can't find data file: {args.data}")
        return

    run_evaluation(args.data, limit=args.limit, output_dir=args.output)


if __name__ == "__main__":
    main()
