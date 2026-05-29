"""
Baseline-методы детектирования вредоносного SQL.

Методы:
    1. RegexDetector   — ручные regex-правила по классам угроз
    2. MLDetector      — TF-IDF (char n-gram) + классификатор sklearn

Запуск:
    python src/baseline.py
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from evaluate import metrics_report, save_results
from normalize import SQLNormalizer

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── 1. Regex Detector ──────────────────────────────────────────────────────

# Паттерны упорядочены от более специфичных к общим.
# Первый совпавший класс побеждает.
_RULES: list[tuple[int, re.Pattern]] = [
    # --- класс 3: privilege escalation ---
    (3, re.compile(
        r"GRANT\s+\w|REVOKE\s+\w|xp_cmdshell|xp_dirtree|UTL_FILE|UTL_HTTP"
        r"|CREATE\s+USER|ALTER\s+USER|CREATE\s+ROLE|sys\.all_tables"
        r"|information_schema\s*\.\s*user|pg_shadow|mysql\.user",
        re.IGNORECASE,
    )),
    # --- класс 4: exfiltration ---
    (4, re.compile(
        r"INTO\s+OUTFILE|INTO\s+DUMPFILE|COPY\s+TO|LOAD_FILE\s*\("
        r"|UTL_HTTP\.request|xp_dirtree\s*\(",
        re.IGNORECASE,
    )),
    # --- класс 2: obfuscation ---
    (2, re.compile(
        r"0x[0-9a-fA-F]{2,}|CHAR\s*\(\s*\d|/\*.*?\*/|"
        r"--\s*\w|\|\|\s*'|\bCONCAT\s*\(|%[0-9a-fA-F]{2}",
        re.IGNORECASE,
    )),
    # --- класс 1: sqli ---
    (1, re.compile(
        r"\bOR\b\s+[\'\d].*=.*[\'\d]|\bAND\b\s+[\'\d].*=.*[\'\d]"
        r"|UNION\s+(ALL\s+)?SELECT|\bSLEEP\s*\(|\bWAITFOR\b"
        r"|BENCHMARK\s*\(|DROP\s+TABLE|TRUNCATE\s+TABLE"
        r"|;\s*(SELECT|INSERT|UPDATE|DELETE|DROP)"
        r"|'\s*OR\s*'|1\s*=\s*1|' --",
        re.IGNORECASE,
    )),
]


class RegexDetector:
    def predict(self, queries: list[str]) -> np.ndarray:
        results = []
        for q in queries:
            label = 0
            for cls, pattern in _RULES:
                if pattern.search(q):
                    label = cls
                    break
            results.append(label)
        return np.array(results)

    def predict_timed(self, queries: list[str]) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        preds = self.predict(queries)
        return preds, time.perf_counter() - t0


# ── 2. ML Detector ─────────────────────────────────────────────────────────

ML_MODELS = {
    "TF-IDF + LogReg": Pipeline([
        ("norm", SQLNormalizer()),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
    ]),
    "TF-IDF + LinearSVC": Pipeline([
        ("norm", SQLNormalizer()),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50_000)),
        ("clf", LinearSVC(max_iter=2000, class_weight="balanced", C=1.0)),
    ]),
    "TF-IDF + RandomForest": Pipeline([
        ("norm", SQLNormalizer()),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=30_000)),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)),
    ]),
}


class MLDetector:
    def __init__(self, model_name: str = "TF-IDF + LogReg"):
        self.model_name = model_name
        self.pipeline = ML_MODELS[model_name]

    def fit(self, X_train: list[str], y_train: np.ndarray) -> None:
        print(f"[*] Training {self.model_name}...")
        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        print(f"    done in {time.perf_counter() - t0:.1f}s")

    def predict_timed(self, X_test: list[str]) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        preds = self.pipeline.predict(X_test)
        return preds, time.perf_counter() - t0


# ── k-Fold Cross-Validation ────────────────────────────────────────────────

def cross_val_ml(X: list[str], y: np.ndarray, n_splits: int = 5) -> list[dict]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    # Regex: не обучается — считаем вручную по фолдам
    print(f"\n[*] Cross-val RegexDetector ({n_splits} folds)...")
    regex_det = RegexDetector()
    f1_scores = []
    for fold, (_, test_idx) in enumerate(cv.split(X, y)):
        X_fold = [X[i] for i in test_idx]
        y_fold = y[test_idx]
        preds = regex_det.predict(X_fold)
        from sklearn.metrics import f1_score
        f1_scores.append(f1_score(y_fold, preds, average="macro", zero_division=0))
    cv_results.append({
        "method": "Regex",
        "cv_f1_macro_mean": float(np.mean(f1_scores)),
        "cv_f1_macro_std":  float(np.std(f1_scores)),
    })
    print(f"    F1-macro: {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}")

    # ML pipelines
    for name, pipeline in ML_MODELS.items():
        print(f"\n[*] Cross-val {name} ({n_splits} folds)...")
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring={"f1_macro": "f1_macro", "f1_weighted": "f1_weighted"},
            n_jobs=1,  # Pipeline с TF-IDF нельзя параллелить с n_jobs=-1 внутри cv
            verbose=0,
        )
        mean_f1 = scores["test_f1_macro"].mean()
        std_f1  = scores["test_f1_macro"].std()
        cv_results.append({
            "method": name,
            "cv_f1_macro_mean": float(mean_f1),
            "cv_f1_macro_std":  float(std_f1),
            "cv_f1_weighted_mean": float(scores["test_f1_weighted"].mean()),
            "cv_f1_weighted_std":  float(scores["test_f1_weighted"].std()),
        })
        print(f"    F1-macro: {mean_f1:.3f} +/- {std_f1:.3f}")

    return cv_results


# ── Main ───────────────────────────────────────────────────────────────────

def main(dataset_file: str = "dataset.csv", results_suffix: str = "") -> list[dict]:
    df = pd.read_csv(DATA_PROCESSED / dataset_file, encoding="utf-8")
    df = df.dropna(subset=["query", "label"])
    df["query"] = df["query"].astype(str)

    print(f"Dataset: {len(df)} samples, {df['label'].nunique()} classes")
    print(df["label_name"].value_counts().to_string())

    X = df["query"].tolist()
    y = df["label"].to_numpy()

    # ── 5-Fold CV ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  5-FOLD CROSS-VALIDATION")
    print("="*60)
    cv_results = cross_val_ml(X, y)
    cv_out = RESULTS_DIR / f"cv_results{results_suffix}.csv"
    pd.DataFrame(cv_results).to_csv(cv_out, index=False, encoding="utf-8")
    print(f"\n[+] CV results saved -> {cv_out}")

    # ── Holdout test (80/20) ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HOLDOUT TEST (80/20 split)")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    results = []

    print("\n[*] Running RegexDetector...")
    regex_det = RegexDetector()
    preds, elapsed = regex_det.predict_timed(X_test)
    results.append(metrics_report(y_test, preds, "Regex", elapsed, len(X_test)))

    for name in ML_MODELS:
        ml = MLDetector(name)
        ml.fit(X_train, y_train)
        preds, elapsed = ml.predict_timed(X_test)
        results.append(metrics_report(y_test, preds, name, elapsed, len(X_test)))

    out = RESULTS_DIR / f"baseline_results{results_suffix}.csv"
    save_results(results, out)
    return results


if __name__ == "__main__":
    main()
