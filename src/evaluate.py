"""
Общие утилиты для оценки детекторов.
Используется и в baseline.py, и в detector_llm.py.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_NAMES = {0: "legit", 1: "sqli", 2: "obfuscated", 3: "privilege", 4: "exfiltration"}


def metrics_report(y_true, y_pred, method_name: str, elapsed_s: float, n_samples: int) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [LABEL_NAMES[l] for l in labels]

    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    print(f"Latency: {elapsed_s / n_samples * 1000:.2f} ms/sample  |  total: {elapsed_s:.1f}s")

    return {
        "method": method_name,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "latency_ms_per_sample": elapsed_s / n_samples * 1000,
        **{
            f"f1_{LABEL_NAMES[l]}": f1_score(y_true, y_pred, labels=[l], average="micro", zero_division=0)
            for l in labels
        },
    }


def save_results(results: list[dict], path) -> None:
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"\n[+] Results saved -> {path}")
