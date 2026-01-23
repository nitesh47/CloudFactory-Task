"""
Evaluation metrics for invoice field extraction.

metrics used:
- Precision: of extracted fields, how many are correct
- Recall: of expected fields, how many were extracted
- F1: harmonic mean of precision and recall
- CER: character error rate
- Exact Match: percentage of perfect matches
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..models.result import HITLDecision


def levenshtein_distance(s1: str, s2: str) -> int:
    """compute edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(extracted: str, expected: str) -> float:
    """CER = edit_distance / len(expected)"""
    if not expected:
        return 0.0 if not extracted else 1.0
    dist = levenshtein_distance(extracted, expected)
    return dist / len(expected)


def normalize_text(text: str) -> str:
    """normalize for comparison"""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class FieldMetrics:
    """metrics for one field type"""

    name: str
    true_positives: int = 0  # extracted and correct
    false_positives: int = 0  # extracted but wrong
    false_negatives: int = 0  # not extracted but should be
    exact_matches: int = 0
    total_cer: float = 0.0
    sample_count: int = 0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.sample_count if self.sample_count > 0 else 0.0

    @property
    def avg_cer(self) -> float:
        extracted = self.true_positives + self.false_positives
        return self.total_cer / extracted if extracted > 0 else 0.0


@dataclass
class EvaluationReport:
    """complete evaluation results"""

    total_docs: int = 0
    auto_accepted: int = 0
    needs_review: int = 0
    rejected: int = 0
    avg_time_ms: float = 0.0
    field_metrics: Dict[str, FieldMetrics] = field(default_factory=dict)

    @property
    def overall_precision(self) -> float:
        tp = sum(m.true_positives for m in self.field_metrics.values())
        fp = sum(m.false_positives for m in self.field_metrics.values())
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def overall_recall(self) -> float:
        tp = sum(m.true_positives for m in self.field_metrics.values())
        fn = sum(m.false_negatives for m in self.field_metrics.values())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def overall_f1(self) -> float:
        p, r = self.overall_precision, self.overall_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def hitl_rate(self) -> float:
        return (
            (self.needs_review + self.rejected) / self.total_docs
            if self.total_docs > 0
            else 0.0
        )

    def summary(self) -> str:
        lines = [
            f"Documents: {self.total_docs}",
            f"Overall - P: {self.overall_precision:.1%}, "
            f"R: {self.overall_recall:.1%}, F1: {self.overall_f1:.1%}",
            f"HITL: {self.auto_accepted} auto / {self.needs_review} review / "
            f"{self.rejected} reject",
            f"Avg time: {self.avg_time_ms:.0f}ms",
            "",
            "Per-field metrics:",
            f"{'Field':<25} {'P':>8} {'R':>8} {'F1':>8} {'Exact':>8} {'CER':>8}",
            "-" * 70,
        ]
        for name in sorted(self.field_metrics.keys()):
            m = self.field_metrics[name]
            lines.append(
                f"{name:<25} {m.precision:>7.1%} {m.recall:>7.1%} {m.f1:>7.1%} "
                f"{m.exact_accuracy:>7.1%} {m.avg_cer:>7.2f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_documents": self.total_docs,
            "overall": {
                "precision": self.overall_precision,
                "recall": self.overall_recall,
                "f1": self.overall_f1,
            },
            "hitl": {
                "auto_accepted": self.auto_accepted,
                "needs_review": self.needs_review,
                "rejected": self.rejected,
                "hitl_rate": self.hitl_rate,
            },
            "avg_processing_time_ms": self.avg_time_ms,
            "fields": {
                name: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "exact_accuracy": m.exact_accuracy,
                    "cer": m.avg_cer,
                }
                for name, m in self.field_metrics.items()
            },
        }


class Evaluator:
    """evaluates extraction results against ground truth"""

    def __init__(self, match_threshold=0.8):
        self.match_threshold = match_threshold

    def evaluate(self, results, ground_truths) -> EvaluationReport:
        """
        compare extraction results to ground truth labels
        """

        report = EvaluationReport()
        report.total_docs = len(results)

        # collect all field names
        all_fields = set()
        for gt in ground_truths:
            all_fields.update(gt.keys())

        for fname in all_fields:
            report.field_metrics[fname] = FieldMetrics(name=fname)

        total_time = 0.0

        for result, gt in zip(results, ground_truths):
            total_time += result.processing_time_ms

            # hitl stats
            if result.hitl_decision == HITLDecision.AUTO_ACCEPT:
                report.auto_accepted += 1
            elif result.hitl_decision == HITLDecision.NEEDS_REVIEW:
                report.needs_review += 1
            else:
                report.rejected += 1

            # evaluate each field
            for field_name, expected in gt.items():
                metrics = report.field_metrics.get(field_name)
                if not metrics:
                    continue

                metrics.sample_count += 1
                extracted = result.get_value(field_name)

                if extracted is None:
                    # field not extracted
                    metrics.false_negatives += 1
                    continue

                # compare extracted vs expected
                ext_norm = normalize_text(extracted)
                exp_norm = normalize_text(expected)

                cer = character_error_rate(ext_norm, exp_norm)
                metrics.total_cer += cer

                if ext_norm == exp_norm:
                    # exact match
                    metrics.true_positives += 1
                    metrics.exact_matches += 1
                elif cer <= (1 - self.match_threshold):
                    # close enough
                    metrics.true_positives += 1
                else:
                    # extracted but wrong
                    metrics.false_positives += 1

        report.avg_time_ms = total_time / len(results) if results else 0.0
        return report

    def analyze_thresholds(self, results, ground_truths, thresholds=None):
        """
        test different hitl confidence thresholds
        """
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        analysis = {}
        for thresh in thresholds:
            accepted = [
                (r, g)
                for r, g in zip(results, ground_truths)
                if r.overall_confidence >= thresh
            ]

            if not accepted:
                analysis[thresh] = {
                    "threshold": thresh,
                    "auto_accept_rate": 0,
                    "auto_accept_count": 0,
                    "accuracy": 0,
                }
                continue

            # check accuracy of auto-accepted
            correct = 0
            for result, gt in accepted:
                all_ok = True
                for fname in ["Summe", "IBAN", "Rechnungsnummer"]:
                    if fname not in gt:
                        continue
                    extracted = result.get_value(fname)
                    if not extracted:
                        all_ok = False
                        break
                    cer = character_error_rate(
                        normalize_text(extracted), normalize_text(gt[fname])
                    )
                    if cer > 0.2:
                        all_ok = False
                        break
                if all_ok:
                    correct += 1

            analysis[thresh] = {
                "threshold": thresh,
                "auto_accept_rate": len(accepted) / len(results),
                "auto_accept_count": len(accepted),
                "accuracy": correct / len(accepted) if accepted else 0,
            }

        return analysis
