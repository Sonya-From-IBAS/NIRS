"""Stage 1: preserve the historical five-class protocol and record its evidence.

Run from the project root; no network or LLM calls. See STAGE1_REPRODUCTION.md.
This is an historical control, not the corrected evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

from baseline import ML_MODELS, RegexDetector
from dataset import LABEL_NAMES

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
LABELS = list(LABEL_NAMES)
SLUGS = {
    "Regex": "regex", "TF-IDF + LogReg": "logreg",
    "TF-IDF + LinearSVC": "linearsvc", "TF-IDF + RandomForest": "randomforest",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8", newline="\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_manifest(path: Path) -> dict:
    raw = path.read_bytes()
    item = {
        "path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
        "sha256_bytes": sha256(raw),
        # A byte-level newline convention, not semantic SQL normalization.
        "sha256_crlf_to_lf": sha256(raw.replace(b"\r\n", b"\n")),
    }
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV rows: {path}")
        item.update(columns=fields, rows=len(rows))
        for field in ("label", "Label", "source", "label_name"):
            if field in fields:
                item[field + "_counts"] = dict(Counter(r[field] for r in rows))
        if "query" in fields:
            texts = [r["query"] for r in rows]
            item.update(
                duplicate_excess_exact=len(texts) - len(set(texts)),
                duplicate_excess_strip=len(texts) - len(set(q.strip() for q in texts)),
                blank_after_strip=sum(not q.strip() for q in texts),
            )
    return item


def tracked_inputs() -> list[Path]:
    paths = list((ROOT / "src").glob("*.py"))
    paths += list((ROOT / "data").rglob("*.csv"))
    paths += list((ROOT / "data" / "raw").glob("*.txt"))
    paths += list((ROOT / "results").glob("*.csv"))
    paths += list((ROOT / "results").glob("*.md"))
    paths += list((ROOT / "review").rglob("*.md"))
    paths += list((ROOT / "tests").glob("*.py"))
    paths += [ROOT / name for name in (
        "requirements.txt", "requirements-lock.txt", "PROJECT_CORRECTION_PLAN.md",
        "STAGE1_REPRODUCTION.md",
    )]
    return sorted(path for path in paths if path.is_file())


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    return result.stdout.strip()


def dataset(root: Path = ROOT) -> pd.DataFrame:
    frame = pd.read_csv(root / "data/processed/dataset.csv", encoding="utf-8")
    # Stable zero-based CSV record index, before the historical dropna step.
    frame.insert(0, "row_id", np.arange(len(frame)))
    frame = frame.dropna(subset=["query", "label"]).copy()
    frame["query"] = frame["query"].astype(str)
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Unexpected target labels")
    if not (frame["label_name"] == frame["label"].map(LABEL_NAMES)).all():
        raise ValueError("label and label_name disagree")
    return frame.reset_index(drop=True)


def splits(frame: pd.DataFrame) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    indices = np.arange(len(frame))
    y = frame["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    result = [("cv", fold, train, test)
              for fold, (train, test) in enumerate(cv.split(indices, y))]
    train, test = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    result.append(("holdout", 0, train, test))
    return result


def new_model(name: str):
    if name == "Regex":
        return RegexDetector()
    pipeline = clone(ML_MODELS[name])
    changes = {key: SEED for key in pipeline.get_params() if key.endswith("random_state")}
    # One worker and one BLAS/OpenMP thread; identical algorithm/hyperparameters.
    changes.update({key: 1 for key in pipeline.get_params() if key.endswith("n_jobs")})
    return pipeline.set_params(**changes)


def scores(y_true, y_pred) -> dict:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        **{f"f1_{LABEL_NAMES[label]}": float(f1_score(
            y_true, y_pred, labels=[label], average="micro", zero_division=0,
        )) for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def derived_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    for (protocol, fold, method), group in predictions.groupby(["protocol", "fold", "method"], sort=False):
        metric = scores(group["y_true"], group["y_pred"])
        metric["confusion_matrix"] = json.dumps(metric["confusion_matrix"])
        records.append({"protocol": protocol, "fold": int(fold), "method": method,
                        "n_samples": len(group), **metric})
    folds = pd.DataFrame(records)
    holdout = folds[folds["protocol"] == "holdout"].drop(columns=["protocol", "fold"]).reset_index(drop=True)
    cv = []
    for method, group in folds[folds["protocol"] == "cv"].groupby("method", sort=False):
        row = {"method": method}
        for metric in ("f1_macro", "f1_weighted"):
            row[f"cv_{metric}_mean"] = float(group[metric].mean())
            row[f"cv_{metric}_std"] = float(group[metric].std(ddof=0))
        cv.append(row)
    return holdout, pd.DataFrame(cv), folds


def compare_historical(holdout: pd.DataFrame, cv: pd.DataFrame, archive: Path) -> pd.DataFrame:
    rows = []
    for protocol, current, filename in (
        ("holdout", holdout, "baseline_results.csv"), ("cv", cv, "cv_results.csv"),
    ):
        previous = pd.read_csv(archive / "results" / filename).set_index("method")
        for row in current.to_dict("records"):
            method = row["method"]
            for metric, value in row.items():
                if metric in previous.columns and metric != "method" and pd.notna(previous.loc[method, metric]):
                    old = float(previous.loc[method, metric])
                    rows.append({"protocol": protocol, "method": method, "metric": metric,
                                 "historical": old, "current": value, "delta": float(value) - old})
    return pd.DataFrame(rows)


def csv_save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def verify(run: Path) -> dict:
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        if sha256((run / relative).read_bytes()) != expected:
            raise ValueError(f"Artifact changed: {relative}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("Incomplete run")
    frame = dataset(run / "snapshot")
    for item in manifest["files"]:
        path = run / "snapshot" / item["path"]
        if sha256(path.read_bytes()) != item["sha256_bytes"]:
            raise ValueError(f"Snapshot changed: {item['path']}")
    # Joblib resolves custom transformers by module name, so check their source.
    normalizer = next(item for item in manifest["files"] if item["path"] == "src/normalize.py")
    if sha256((ROOT / "src/normalize.py").read_bytes()) != normalizer["sha256_bytes"]:
        raise ValueError("Normalizer changed; run verification using snapshot/src/reproduce_baseline.py")
    predictions = pd.read_csv(run / "predictions.csv")
    if predictions.duplicated(["protocol", "fold", "method", "row_id"]).any():
        raise ValueError("Duplicate predictions")
    expected_splits = splits(frame)
    for protocol, fold, train, test in expected_splits:
        recorded = pd.read_csv(run / "splits" / f"{protocol}-{fold}.csv")
        expected = pd.DataFrame({
            "row_id": np.concatenate([frame.iloc[train]["row_id"], frame.iloc[test]["row_id"]]),
            "part": ["train"] * len(train) + ["test"] * len(test),
        })
        pd.testing.assert_frame_equal(recorded, expected)
        for method in SLUGS:
            rows = predictions[(predictions["protocol"] == protocol)
                               & (predictions["fold"] == fold) & (predictions["method"] == method)]
            if rows["row_id"].tolist() != frame.iloc[test]["row_id"].tolist():
                raise ValueError(f"Prediction IDs/order differ: {protocol}/{fold}/{method}")
            if rows["y_true"].tolist() != frame.iloc[test]["label"].tolist():
                raise ValueError("Prediction targets disagree with dataset")
            if not rows["y_pred"].isin(LABELS).all():
                raise ValueError("Invalid predicted label")
    if len(predictions) != len(frame) * 4 + len(expected_splits[-1][3]) * 4:
        raise ValueError("Unexpected prediction count")
    for table, filename in zip(derived_tables(predictions),
                               ("baseline_results.csv", "cv_results.csv", "fold_metrics.csv")):
        pd.testing.assert_frame_equal(pd.read_csv(run / filename), table, check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
    test_indices = expected_splits[-1][3]
    for method, slug in SLUGS.items():
        if method == "Regex":
            continue
        # Only load locally generated and checksum-verified trusted artifacts.
        pipeline = joblib.load(run / "models" / f"{slug}.joblib")
        actual = pipeline.predict(frame.iloc[test_indices]["query"].tolist())
        expected = predictions[(predictions["protocol"] == "holdout")
                               & (predictions["method"] == method)]["y_pred"].to_numpy()
        np.testing.assert_array_equal(actual, expected)
    return {"status": "passed", "artifacts_checked": len(hashes),
            "predictions_checked": len(predictions), "loaded_models_checked": 3}


def run_experiment(output: Path, repeat_of: Path | None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("snapshot", "splits", "models"):
        (output / directory).mkdir()
    write_json(output / "status.json", {"status": "running", "started_utc": utc_now()})
    inputs = tracked_inputs()
    manifest = {
        "created_utc": utc_now(), "git_head": git_output("rev-parse", "HEAD"),
        "git_status_before": git_output("status", "--short"),
        "files": [file_manifest(path) for path in inputs],
        "newline_hash_rule": "Replace CRLF bytes with LF only; no trimming or SQL normalization",
        "excluded": [".env", ".git contents", "unrelated files outside project"],
    }
    write_json(output / "manifest.json", manifest)
    for path in inputs:
        target = output / "snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    environment = {
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "threadpools": threadpool_info(),
        "environment_allowlist": {key: os.environ.get(key) for key in (
            "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }
    write_json(output / "environment.json", environment)
    random.seed(SEED)
    np.random.seed(SEED)
    frame = dataset()
    configurations = {}
    for name in ML_MODELS:
        configurations[name] = {key: value if isinstance(value, (str, int, float, bool, type(None)))
                                else repr(value) for key, value in new_model(name).get_params().items()}
    write_json(output / "config.json", {
        "purpose": "Historical five-class control; row-stratified CV and holdout, not a transfer estimate",
        "seed": SEED, "cv": {"folds": 5, "shuffle": True, "random_state": SEED},
        "holdout": {"test_size": 0.2, "stratify": "label", "random_state": SEED},
        "label_order": LABELS, "label_names": LABEL_NAMES,
        "csv_records_before_dropna": next(x["rows"] for x in manifest["files"]
                                          if x["path"] == "data/processed/dataset.csv"),
        "rows_after_historical_dropna": len(frame),
        "changes_from_original": ["Explicit random_state=42", "Independent cloned pipelines",
                                  "One worker and one numerical thread", "Persist predictions and holdout models"],
        "model_parameters": configurations,
        "repeat_of": str(repeat_of) if repeat_of else None,
    })
    prediction_blocks, timings, captured_warnings = [], [], []
    for protocol, fold, train, test in splits(frame):
        csv_save(pd.DataFrame({
            "row_id": np.concatenate([frame.iloc[train]["row_id"], frame.iloc[test]["row_id"]]),
            "part": ["train"] * len(train) + ["test"] * len(test),
        }), output / "splits" / f"{protocol}-{fold}.csv")
        train_x, test_x = frame.iloc[train]["query"].tolist(), frame.iloc[test]["query"].tolist()
        train_y = frame.iloc[train]["label"].to_numpy()
        for method, slug in SLUGS.items():
            print(f"[{protocol} {fold + 1}] {method}: train={len(train)} test={len(test)}", flush=True)
            model = new_model(method)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if method != "Regex":
                    model.fit(train_x, train_y)
                fit_seconds = time.perf_counter() - started
                started_predict = time.perf_counter()
                predicted = model.predict(test_x)
                predict_seconds = time.perf_counter() - started_predict
            captured_warnings.extend({"protocol": protocol, "fold": fold, "method": method,
                                      "category": w.category.__name__, "message": str(w.message)} for w in caught)
            if protocol == "holdout" and method != "Regex":
                target = output / "models" / f"{slug}.joblib"
                joblib.dump(model, target, compress=3)
                np.testing.assert_array_equal(joblib.load(target).predict(test_x), predicted)
            prediction_blocks.append(pd.DataFrame({
                "protocol": protocol, "fold": fold, "method": method,
                "row_id": frame.iloc[test]["row_id"].to_numpy(),
                "y_true": frame.iloc[test]["label"].to_numpy(), "y_pred": predicted,
            }))
            timings.append({"protocol": protocol, "fold": fold, "method": method,
                            "train_seconds": fit_seconds, "predict_seconds": predict_seconds,
                            "n_test": len(test), "predict_ms_per_sample": predict_seconds * 1000 / len(test)})
            print(f"  F1-macro={scores(frame.iloc[test]['label'], predicted)['f1_macro']:.9f}; "
                  f"fit={fit_seconds:.1f}s predict={predict_seconds:.2f}s", flush=True)
            # Save completed work even if a later model fails; status stays running.
            csv_save(pd.concat(prediction_blocks, ignore_index=True), output / "predictions.csv")
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    holdout, cv, folds = derived_tables(predictions)
    for table, filename in ((holdout, "baseline_results.csv"), (cv, "cv_results.csv"),
                            (folds, "fold_metrics.csv"), (pd.DataFrame(timings), "timings.csv")):
        csv_save(table, output / filename)
    csv_save(compare_historical(holdout, cv, output / "snapshot"), output / "historical_deltas.csv")
    write_json(output / "warnings.json", captured_warnings)
    for item in manifest["files"]:
        if sha256((ROOT / item["path"]).read_bytes()) != item["sha256_bytes"]:
            raise ValueError(f"Input/code changed during run: {item['path']}")
    if repeat_of:
        verify(repeat_of)
        previous_manifest = json.loads((repeat_of / "manifest.json").read_text(encoding="utf-8"))
        if manifest["files"] != previous_manifest["files"]:
            raise ValueError("Repeat used different input/code files")
        previous_environment = json.loads((repeat_of / "environment.json").read_text(encoding="utf-8"))
        if environment != previous_environment:
            raise ValueError("Repeat used a different environment")
        for filename in ("predictions.csv", "baseline_results.csv", "cv_results.csv", "fold_metrics.csv"):
            if (output / filename).read_bytes() != (repeat_of / filename).read_bytes():
                raise ValueError(f"Repeat differs: {filename}")
        write_json(output / "reproducibility.json", {
            "status": "passed", "reference": str(repeat_of),
            "identical": ["input manifests", "environment", "splits (deterministic reconstruction)",
                          "predictions.csv", "baseline_results.csv", "cv_results.csv", "fold_metrics.csv"],
            "excluded_from_equality": ["wall times", "timestamps", "joblib byte representation"],
        })
    write_json(output / "status.json", {"status": "complete", "finished_utc": utc_now()})
    write_json(output / "artifact_hashes.json", {
        path.relative_to(output).as_posix(): sha256(path.read_bytes())
        for path in sorted(output.rglob("*")) if path.is_file()
    })
    print(json.dumps(verify(output), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True, help="New, non-existing directory")
    run.add_argument("--repeat-of", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "verify":
            print(json.dumps(verify(args.run_dir.resolve()), ensure_ascii=False))
        else:
            run_experiment(args.output.resolve(), args.repeat_of.resolve() if args.repeat_of else None)


if __name__ == "__main__":
    main()
