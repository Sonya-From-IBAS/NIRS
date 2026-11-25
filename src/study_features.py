"""Compare frozen feature candidates on inner validation and archive sparse caches."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
from psycopg.types.json import Jsonb
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from evaluation_metrics import classification_report
from sql_features import CONFIGURATIONS, SQLFeatureEncoder, VERSION, prepare_input, project_input
from sql_views import KEYWORDS, lex
from storage import ROOT, Storage, canonical, digest, file_hash

OUTPUT = ROOT / "results/stage7"
LOCAL = ROOT / ".local/stage7"
SCOPES = {"legacy": "legacy-inner-0", "sqlstateguard": "transfer-web_app-to-wordpress-inner", "controlled": "controlled-inner-0"}
SQL_CONFIGS = ("legacy_text", "raw_text", "masked_text", "masked_parts", "structure_only", "text_structure")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_scope(store, entry, legacy=False):
    train_ids = store.split_ids(entry["split_id"], 0, "train")
    validation_ids = store.split_ids(entry["split_id"], 0, "validation")
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Invalid development split")
    with store.connect() as connection:
        connection.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        rows = connection.execute("""SELECT o.observation_id,o.sql_text,o.context_json,a.target_label,c.source_label
            FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
            JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
            JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE d.dataset_version_id=%s AND o.observation_id=ANY(%s)""",
            (entry["dataset_version_id"], train_ids + validation_ids)).fetchall()
    indexed = {row["observation_id"]: row for row in rows}
    def part(ids):
        inputs, labels = [], []
        for oid in ids:
            row = indexed[oid]
            # Read the import-validated context through the same predecision
            # projection used for new inputs. Retrospective fields are absent.
            inputs.append(prepare_input({"sql": {"text": row["sql_text"]}, "context": row["context_json"]}))
            labels.append(int(row["source_label"]["label"]) if legacy else row["target_label"])
        return inputs, np.array(labels, dtype=np.int64)
    return train_ids, validation_ids, *part(train_ids), *part(validation_ids)


def renamed_input(document):
    projected = deepcopy(document.projected)
    symbols = document.sql["symbols"]
    names = sorted({value for (kind, value) in symbols.names if kind == "plain" and re.fullmatch(r"[a-z_][a-z_0-9]*", value)
                    and value.upper() not in KEYWORDS}, key=lambda x: (-len(x), x))
    # Account names may be SQL string literals rather than word identifiers.
    for field in ("active_role",):
        entry = projected["context"].get(field, {})
        if entry.get("value"):
            part = str(entry["value"]).split("@", 1)[0]
            if re.fullmatch(r"[a-z_][a-z_0-9]*", part):
                names.append(part)
    def recipients(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "recipient" and isinstance(item, str):
                    names.append(item.split("@", 1)[0])
                else:
                    recipients(item)
        elif isinstance(value, list):
            for item in value:
                recipients(item)
    for field in ("permissions", "policy"):
        for rule in (projected["context"].get(field, {}).get("value") or {}).get("rules", []):
            try:
                recipients(json.loads(rule.get("condition") or "null"))
            except json.JSONDecodeError:
                pass
    mapping = {name: f"object_renamed_{index}" for index, name in enumerate(dict.fromkeys(names))}
    if not mapping:
        return prepare_input(projected)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(key) for key in mapping) + r")\b", re.I)
    def replace(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: mapping[m.group().lower()], value)
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value
    return prepare_input(replace(projected))


def reformatted_input(document):
    parts = []
    for token in lex(document.projected["sql"]):
        if token.kind == "space":
            parts.append("  \n ")
        elif token.kind == "word" and token.text.upper() in KEYWORDS:
            parts.append(token.text.lower())
        else:
            parts.append(token.text)
    projected = deepcopy(document.projected)
    projected["sql"] = "".join(parts)
    return prepare_input(projected)


def cache_representation(store, entry, name, config, train_ids, validation_ids, X_train, X_validation, y_train, y_validation, cohort, mutations):
    code = {filename: digest((ROOT / "src" / filename).read_bytes().replace(b"\r\n", b"\n"))
            for filename in ("sql_views.py", "sql_features.py", "normalize.py", "study_features.py")}
    configuration = {"version": VERSION, "candidate": name, "parameters": config,
        "split_id": entry["split_id"], "fold": 0, "fit_part": "train", "evaluation_part": "validation",
        "fit_ids_sha256": digest(canonical(train_ids).encode()), "validation_ids_sha256": digest(canonical(validation_ids).encode()),
        "code_sha256_lf": code, "probe_classifier": {"type": "LogisticRegression", "solver": "liblinear", "C": 1.0,
            "class_weight": "balanced", "max_iter": 2000, "random_state": 20261125},
        "label_space": "legacy_five_class" if cohort == "legacy" else "binary", "role": "inner_feature_development"}
    rid = "repr_" + digest(canonical({"dataset": entry["dataset_version_id"], "configuration": configuration}).encode())
    folder = LOCAL / rid
    folder.mkdir(parents=True, exist_ok=True)
    with store.connect() as connection:
        cached = connection.execute("SELECT configuration FROM nirs.representations WHERE representation_id=%s", (rid,)).fetchone()
        cached_roles = connection.execute("SELECT role,artifact_id FROM nirs.representation_artifacts WHERE representation_id=%s", (rid,)).fetchall() if cached else []
    if cached:
        if cached["configuration"] != configuration:
            raise ValueError("Cached representation configuration differs")
        artifacts = {row["role"]: row["artifact_id"] for row in cached_roles}
        if len(artifacts) != 13:
            raise ValueError("Incomplete representation cache")
        paths = {role: store.get_artifact(artifact) for role, artifact in artifacts.items()}
        if json.loads(paths["rows_train.json"].read_text()) != train_ids or json.loads(paths["rows_validation.json"].read_text()) != validation_ids:
            raise ValueError("Cached matrix row order differs")
        pipeline = joblib.load(paths["pipeline.joblib"])
        expected = sparse.load_npz(paths["validation.npz"])
        actual = pipeline.named_steps["features"].transform(X_validation)
        predictions = json.loads(paths["validation_predictions.json"].read_text())
        if expected.shape != actual.shape or (expected != actual).nnz or not np.array_equal(pipeline.predict(X_validation), predictions["prediction"]):
            raise ValueError("Cached transformation or prediction differs")
        report = json.loads(paths["metadata.json"].read_text())
        report["artifacts"] = artifacts
        return report
    started = time.perf_counter()
    encoder = SQLFeatureEncoder(**config)
    matrix_train = encoder.fit_transform(X_train, y_train)
    matrix_validation = encoder.transform(X_validation)
    feature_seconds = time.perf_counter() - started
    model = LogisticRegression(**{key: value for key, value in configuration["probe_classifier"].items() if key != "type"})
    model.fit(matrix_train, y_train)
    predicted = model.predict(matrix_validation)
    pipeline = Pipeline([("features", encoder), ("classifier", model)])
    save_json(folder / "rows_train.json", train_ids)
    save_json(folder / "rows_validation.json", validation_ids)
    save_json(folder / "feature_names.json", encoder.get_feature_names_out().tolist())
    joblib.dump(encoder, folder / "encoder.joblib", compress=3)
    joblib.dump(pipeline, folder / "pipeline.joblib", compress=3)
    for part, matrix in (("train", matrix_train), ("validation", matrix_validation)):
        matrix = matrix.tocsr().astype(np.float32)
        matrix.eliminate_zeros(); matrix.sort_indices()
        sparse.save_npz(folder / (part + ".npz"), matrix, compressed=True)
        restored = sparse.load_npz(folder / (part + ".npz"))
        if (restored.shape != matrix.shape or not np.array_equal(restored.indptr, matrix.indptr)
                or not np.array_equal(restored.indices, matrix.indices) or not np.array_equal(restored.data, matrix.data)):
            raise ValueError("Sparse matrix roundtrip differs")
    restored = joblib.load(folder / "pipeline.joblib")
    after = restored.named_steps["features"].transform(X_validation)
    if matrix_validation.shape != after.shape or (matrix_validation != after).nnz:
        raise ValueError("Loaded encoder columns or values differ")
    if not np.array_equal(restored.predict(X_validation), predicted):
        raise ValueError("Loaded pipeline predictions differ")
    metrics = classification_report(y_validation.tolist(), predicted.tolist(), label_space=configuration["label_space"])
    save_json(folder / "validation_predictions.json", {"observation_ids": validation_ids, "truth": y_validation.tolist(), "prediction": predicted.tolist()})
    stability = {}
    for kind, documents in mutations.items():
        baseline = predicted[:len(documents)]
        changed = restored.predict(documents)
        stability[kind] = {"rows": len(documents), "prediction_changes": int(np.count_nonzero(changed != baseline)),
                           "agreement": float(np.mean(changed == baseline))}
    widths = dict(Counter(encoder.feature_groups_.tolist()))
    coefficient_norms = {branch: float(np.linalg.norm(model.coef_[:, encoder.feature_groups_ == branch])) for branch in widths}
    report = {"representation_id": rid, "cohort": cohort, "candidate": name, "configuration": configuration,
        "train_rows": len(train_ids), "validation_rows": len(validation_ids), "fit_rows_in_encoder": encoder.fit_rows_,
        "train_matrix": {"shape": list(matrix_train.shape), "format": "csr", "dtype": str(matrix_train.dtype), "nnz": matrix_train.nnz},
        "validation_matrix": {"shape": list(matrix_validation.shape), "format": "csr", "dtype": str(matrix_validation.dtype), "nnz": matrix_validation.nnz},
        "feature_group_widths": widths, "classifier_coefficient_norms": coefficient_norms,
        "metrics": metrics, "stability": stability, "feature_seconds": round(feature_seconds, 4),
        "matrix_and_pipeline_reload": "passed"}
    save_json(folder / "metadata.json", report)
    roles = ["encoder.joblib", "pipeline.joblib", "train.npz", "validation.npz", "rows_train.json", "rows_validation.json",
             "feature_names.json", "validation_predictions.json", "metadata.json"]
    artifacts = {role: store.register_artifact(folder / role) for role in roles}
    for name, expected in code.items():
        path = ROOT / "src" / name
        if digest(path.read_bytes().replace(b"\r\n", b"\n")) != expected:
            raise ValueError("Feature source changed during the run")
        artifacts["code/" + name] = store.register_artifact(path)
    with store.write() as connection:
        existing = connection.execute("SELECT configuration FROM nirs.representations WHERE representation_id=%s", (rid,)).fetchone()
        if existing and existing["configuration"] != configuration:
            raise ValueError("Representation configuration changed")
        if not existing:
            connection.execute("INSERT INTO nirs.representations VALUES (%s,%s,%s)", (rid, entry["dataset_version_id"], Jsonb(configuration)))
        for role, artifact in artifacts.items():
            metadata = {"part": "train" if role.startswith("train") or role == "rows_train.json" else "validation" if role.startswith("validation") or role == "rows_validation.json" else "shared",
                        "fit_split_id": entry["split_id"], "fit_fold": 0}
            if role in ("train.npz", "validation.npz"):
                metadata.update(report["train_matrix" if role == "train.npz" else "validation_matrix"])
                metadata.update(row_ids_artifact=artifacts["rows_train.json" if role == "train.npz" else "rows_validation.json"],
                                encoder_artifact=artifacts["encoder.joblib"], feature_names_artifact=artifacts["feature_names.json"])
            connection.execute("INSERT INTO nirs.representation_artifacts VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                               (rid, artifact, role, Jsonb(metadata)))
    report["artifacts"] = artifacts
    return report


def main():
    store = Storage()
    registry = json.loads((ROOT / "results/stage6/split_registry.json").read_text(encoding="utf-8"))["protocols"]
    reports, invariants = [], {}
    with threadpool_limits(limits=1):
        for cohort, scope in SCOPES.items():
            entry = registry[scope]
            train_ids, validation_ids, X_train, y_train, X_validation, y_validation = load_scope(store, entry, legacy=cohort == "legacy")
            probe = X_validation[:min(120, len(X_validation))]
            mutations = {"formatting": [reformatted_input(item) for item in probe]}
            if cohort == "controlled":
                mutations["renaming_with_context"] = [renamed_input(item) for item in probe]
                invariants[cohort] = {"renamed_rows": len(probe),
                    "same_masked_SQL": sum(first.sql["masked"] == second.sql["masked"] for first, second in zip(probe, mutations["renaming_with_context"])),
                    "same_context_tokens": sum(first.context_text == second.context_text for first, second in zip(probe, mutations["renaming_with_context"]))}
            names = list(CONFIGURATIONS) if cohort == "controlled" else list(SQL_CONFIGS) + (["text_structure_selected", "text_structure_unscaled"] if cohort == "sqlstateguard" else [])
            for name in names:
                report = cache_representation(store, entry, name, CONFIGURATIONS[name], train_ids, validation_ids,
                    X_train, X_validation, y_train, y_validation, cohort, mutations)
                reports.append(report)
                print(f"{cohort}/{name}: F1={report['metrics']['primary_f1']:.4f}, columns={report['train_matrix']['shape'][1]}", flush=True)
    save_json(OUTPUT / "representation_study.json", {"version": VERSION, "scope": "fixed-classifier inner-validation feature development",
        "reports": reports, "invariance_checks": invariants, "candidate_selection": "all candidates remain available for per-outer-fold selection at stage 8"})
    print(json.dumps({"representations": len(reports), "invariance_checks": invariants}, indent=2))


if __name__ == "__main__":
    main()
