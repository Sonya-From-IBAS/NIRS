"""Import the saved stage-1 control as historical five-class experiments."""

import csv
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from reproduce_baseline import verify
from storage import Storage, canonical, file_hash


def import_stage1(store: Storage, archive: Path, dataset_id: str, group_name: str) -> dict:
    verify(archive)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((archive / "config.json").read_text(encoding="utf-8"))
    with store.connect() as connection:
        members = connection.execute("""SELECT d.ordinal,d.observation_id,c.source_version_id
            FROM nirs.dataset_members d JOIN nirs.occurrences c
            ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE d.dataset_version_id=%s ORDER BY d.ordinal""", (dataset_id,)).fetchall()
        if not members:
            raise ValueError("Unknown dataset")
        source = connection.execute("SELECT sha256_bytes FROM nirs.source_versions WHERE source_version_id=%s",
                                    (members[0]["source_version_id"],)).fetchone()
    dataset_hash = next(item["sha256_bytes"] for item in manifest["files"]
                        if item["path"] == "data/processed/dataset.csv")
    if source["sha256_bytes"] != dataset_hash:
        raise ValueError("Stage-1 dataset bytes differ from the registered dataset")
    mapping = {row["ordinal"]: row["observation_id"] for row in members}
    split_files = sorted((archive / "splits").glob("*.csv"))
    split_hashes = {path.name: file_hash(path) for path in split_files}
    split_ids = {protocol: dataset_id + ":legacy:" + protocol for protocol in ("cv", "holdout")}
    for protocol, split_id in split_ids.items():
        specification = {"protocol": protocol, "type": "historical_row_stratified", "files": {
            name: checksum for name, checksum in split_hashes.items() if name.startswith(protocol + "-")}}
        with store.write() as connection:
            existing = connection.execute("SELECT specification FROM nirs.splits WHERE split_id=%s", (split_id,)).fetchone()
            if existing:
                if existing["specification"] != specification:
                    raise ValueError("Saved split configuration differs")
                continue
            connection.execute("INSERT INTO nirs.splits VALUES (%s,%s,%s)", (split_id, dataset_id, Jsonb(specification)))
            for path in split_files:
                if not path.name.startswith(protocol + "-"):
                    continue
                fold = int(path.stem.rsplit("-", 1)[1])
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = [(split_id, dataset_id, mapping[int(row["row_id"])], fold, row["part"])
                            for row in csv.DictReader(stream)]
                store._copy(connection, "split_members", ["split_id", "dataset_version_id", "observation_id", "fold", "part"], rows)

    paths = [path for path in archive.iterdir() if path.is_file()]
    paths += list((archive / "models").glob("*.joblib"))
    paths += list((archive / "snapshot/src").glob("*.py"))
    paths += [path for path in (archive / "snapshot").glob("requirements*.txt")]
    paths += split_files
    artifacts = {path.relative_to(archive).as_posix(): store.register_artifact(path) for path in sorted(paths)}
    predictions = {}
    with (archive / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            predictions.setdefault(row["method"], []).append(row)
    with (archive / "fold_metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    results = []
    model_roles = {"Regex": None, "TF-IDF + LogReg": "models/logreg.joblib",
                   "TF-IDF + LinearSVC": "models/linearsvc.joblib",
                   "TF-IDF + RandomForest": "models/randomforest.joblib"}
    for method, rows in predictions.items():
        run_id = group_name + ":" + method
        configuration = {"stage1": config, "original_git_head": manifest["git_head"],
            "predictions_sha256": artifacts["predictions.csv"], "reference": "original unverified five-class source labels",
            "model_artifact_role": model_roles[method]}
        with store.write() as connection:
            existing = connection.execute("SELECT configuration FROM nirs.experiment_runs WHERE run_id=%s", (run_id,)).fetchone()
            if existing:
                if existing["configuration"] != configuration:
                    raise ValueError("Existing run has different configuration")
                results.append({"run_id": run_id, "predictions": len(rows), "reused": True})
                continue
            connection.execute("INSERT INTO nirs.experiment_runs VALUES (%s,%s,%s,'legacy_five_class','prepared',%s)",
                               (run_id, dataset_id, method, Jsonb(configuration)))
            values = [(run_id, dataset_id, split_ids[row["protocol"]], mapping[int(row["row_id"])],
                row["protocol"], int(row["fold"]), int(row["y_true"]), int(row["y_pred"]), "ok", Jsonb({})) for row in rows]
            store._copy(connection, "predictions", ["run_id", "dataset_version_id", "split_id", "observation_id",
                "protocol", "fold", "true_label", "predicted_label", "status", "metadata"], values)
            metric_values = []
            for row in metrics:
                if row["method"] != method:
                    continue
                for key, value in row.items():
                    if key in ("method", "protocol", "fold"):
                        continue
                    parsed = json.loads(value) if key == "confusion_matrix" else (int(value) if key == "n_samples" else float(value))
                    metric_values.append((run_id, row["protocol"], int(row["fold"]), key, Jsonb(parsed)))
            store._copy(connection, "metrics", ["run_id", "protocol", "fold", "metric_name", "value"], metric_values)
            for role, artifact_id in artifacts.items():
                if role.startswith("models/") and role != model_roles[method]:
                    continue
                connection.execute("INSERT INTO nirs.run_artifacts VALUES (%s,%s,%s,%s)",
                                   (run_id, artifact_id, role, Jsonb({"archive": group_name})))
            connection.execute("UPDATE nirs.experiment_runs SET status='complete' WHERE run_id=%s", (run_id,))
        print(f"Registered {run_id}: {len(rows)} predictions", flush=True)
        results.append({"run_id": run_id, "predictions": len(rows), "reused": False})
    return {"archive": group_name, "runs": results, "registered_archive_files": len(artifacts)}
