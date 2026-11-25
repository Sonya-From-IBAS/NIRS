"""Pinned F1/recall/FPR definitions and explicit handling of missing decisions."""

from collections import Counter
import math

VERSION = "evaluation-metrics-v1"
STATUSES = {"ok", "api_error", "parse_error", "not_run"}


def _divide(first, second):
    return first / second if second else None


def classification_report(y_true, y_pred, *, label_space="binary", statuses=None, sample_weight=None):
    if label_space not in ("binary", "legacy_five_class"):
        raise ValueError("Unknown reference label space")
    count = len(y_true)
    if len(y_pred) != count:
        raise ValueError("Prediction/reference lengths differ")
    statuses = list(statuses) if statuses is not None else ["ok"] * count
    weights = list(sample_weight) if sample_weight is not None else [1] * count
    if len(statuses) != count or len(weights) != count or not set(statuses) <= STATUSES:
        raise ValueError("Invalid status or weight alignment")
    labels = list(range(2 if label_space == "binary" else 5))
    matrix = [[0 for _ in labels] for _ in labels]
    successful = 0
    for truth, predicted, status, weight in zip(y_true, y_pred, statuses, weights):
        if truth not in labels or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid reference label or sample weight")
        if status != "ok":
            if predicted is not None:
                raise ValueError("A failed decision cannot contain a fabricated prediction")
            continue
        if predicted not in labels:
            raise ValueError("A successful decision needs a label in the declared space")
        successful += 1
        matrix[truth][predicted] += weight
    tp = sum(matrix[i][j] for i in labels if i != 0 for j in labels if j != 0)
    fp = sum(matrix[0][j] for j in labels if j != 0)
    fn = sum(matrix[i][0] for i in labels if i != 0)
    tn = matrix[0][0]
    per_class = {}
    for label in labels:
        class_tp = matrix[label][label]
        support = sum(matrix[label])
        predicted_support = sum(row[label] for row in matrix)
        per_class[str(label)] = {"support": support, "predicted_support": predicted_support,
            "precision": _divide(class_tp, predicted_support), "recall": _divide(class_tp, support),
            "f1": _divide(2 * class_tp, support + predicted_support)}
    if label_space == "binary":
        score = _divide(2 * tp, 2 * tp + fp + fn)
    else:
        # Fixed legacy class space; absent-class F1 follows zero_division=0.
        score = sum(value["f1"] or 0 for value in per_class.values()) / 5 if successful else None
    complete = count > 0 and successful == count
    return {"metric_contract": VERSION, "label_space": label_space, "observations": count,
            "status_counts": dict(Counter(statuses)), "decision_coverage": _divide(successful, count),
            "successful_decisions": successful, "primary_f1": score if complete else None,
            "eligible_for_primary_ranking": complete and score is not None,
            "successful_decision_metrics": {"f1": score, "attack_recall": _divide(tp, tp + fn),
                "false_positive_rate": _divide(fp, fp + tn), "attack_precision": _divide(tp, tp + fp),
                "confusion_matrix": matrix, "collapsed_binary_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "per_class": per_class}, "sample_weighted": sample_weight is not None}
