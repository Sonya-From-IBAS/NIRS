"""Ablate explicit SQL/context relationships on the frozen inner development fold."""

from collections import Counter
import json
import time

import joblib
import numpy as np
from psycopg.types.json import Jsonb
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from evaluation_metrics import classification_report
from relational_features import CONFIGURATIONS, RelationalFeatureEncoder, VERSION
from storage import ROOT, Storage, canonical, digest
from study_features import LOCAL, OUTPUT, load_scope, save_json


def main():
    store = Storage()
    study = json.loads((OUTPUT / "representation_study.json").read_text(encoding="utf-8"))
    entry = json.loads((ROOT / "results/stage6/split_registry.json").read_text())["protocols"]["controlled-inner-0"]
    train_ids, validation_ids, train, y_train, validation, y_validation = load_scope(store, entry)
    reports = []
    with threadpool_limits(limits=1):
        for name, parameters in CONFIGURATIONS.items():
            code = {filename: digest((ROOT / "src" / filename).read_bytes().replace(b"\r\n", b"\n")) for filename in
                    ("sql_views.py", "sql_features.py", "normalize.py", "study_features.py", "relational_features.py", "study_relations.py")}
            configuration = {"version": VERSION, "candidate": name, "parameters": parameters, "split_id": entry["split_id"],
                "fold": 0, "fit_part": "train", "evaluation_part": "validation", "role": "inner_feature_development",
                "fit_ids_sha256": digest(canonical(train_ids).encode()), "validation_ids_sha256": digest(canonical(validation_ids).encode()),
                "code_sha256_lf": code, "probe_classifier": {"type": "LogisticRegression", "solver": "liblinear", "C": 1.0,
                    "class_weight": "balanced", "max_iter": 2000, "random_state": 20261125}, "label_space": "binary"}
            rid = "repr_" + digest(canonical({"dataset": entry["dataset_version_id"], "configuration": configuration}).encode())
            with store.connect() as connection:
                known = connection.execute("SELECT configuration FROM nirs.representations WHERE representation_id=%s", (rid,)).fetchone()
                roles = connection.execute("SELECT role,artifact_id FROM nirs.representation_artifacts WHERE representation_id=%s", (rid,)).fetchall() if known else []
            if known:
                artifacts = {row["role"]: row["artifact_id"] for row in roles}
                if len(artifacts) != 15 or known["configuration"] != configuration:
                    raise ValueError("Incomplete relational representation")
                paths = {role: store.get_artifact(value) for role, value in artifacts.items()}
                pipeline = joblib.load(paths["pipeline.joblib"])
                matrix = sparse.load_npz(paths["validation.npz"])
                actual = pipeline.named_steps["features"].transform(validation)
                predictions = json.loads(paths["validation_predictions.json"].read_text())
                if (matrix != actual).nnz or not np.array_equal(pipeline.predict(validation), predictions["prediction"]):
                    raise ValueError("Relational cache reload differs")
                report = json.loads(paths["metadata.json"].read_text())
                report["artifacts"] = artifacts
                reports.append(report)
                continue
            folder = LOCAL / rid
            folder.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            encoder = RelationalFeatureEncoder(**parameters)
            matrix_train = encoder.fit_transform(train, y_train)
            matrix_validation = encoder.transform(validation)
            model = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", max_iter=2000, random_state=20261125)
            model.fit(matrix_train, y_train)
            pipeline = Pipeline([("features", encoder), ("classifier", model)])
            predicted = pipeline.predict(validation)
            for part, matrix, ids in (("train", matrix_train, train_ids), ("validation", matrix_validation, validation_ids)):
                sparse.save_npz(folder / (part + ".npz"), matrix, compressed=True)
                save_json(folder / ("rows_" + part + ".json"), ids)
                if (sparse.load_npz(folder / (part + ".npz")) != matrix).nnz:
                    raise ValueError("Relational sparse matrix roundtrip differs")
            save_json(folder / "feature_names.json", encoder.get_feature_names_out().tolist())
            save_json(folder / "validation_predictions.json", {"observation_ids": validation_ids, "truth": y_validation.tolist(), "prediction": predicted.tolist()})
            joblib.dump(encoder, folder / "encoder.joblib", compress=3)
            joblib.dump(pipeline, folder / "pipeline.joblib", compress=3)
            restored = joblib.load(folder / "pipeline.joblib")
            if (restored.named_steps["features"].transform(validation) != matrix_validation).nnz or not np.array_equal(restored.predict(validation), predicted):
                raise ValueError("Relational pipeline reload differs")
            widths = dict(Counter(encoder.feature_groups_.tolist()))
            report = {"representation_id": rid, "cohort": "controlled", "candidate": name, "configuration": configuration,
                "train_rows": len(train_ids), "validation_rows": len(validation_ids), "fit_rows_in_encoder": encoder.fit_rows_,
                "train_matrix": {"shape": list(matrix_train.shape), "format": "csr", "dtype": str(matrix_train.dtype), "nnz": matrix_train.nnz},
                "validation_matrix": {"shape": list(matrix_validation.shape), "format": "csr", "dtype": str(matrix_validation.dtype), "nnz": matrix_validation.nnz},
                "feature_group_widths": widths, "classifier_coefficient_norms": {key: float(np.linalg.norm(model.coef_[:, encoder.feature_groups_ == key])) for key in widths},
                "metrics": classification_report(y_validation.tolist(), predicted.tolist()), "feature_seconds": round(time.perf_counter() - started, 4),
                "matrix_and_pipeline_reload": "passed", "stability": {}}
            save_json(folder / "metadata.json", report)
            files = ["encoder.joblib", "pipeline.joblib", "train.npz", "validation.npz", "rows_train.json", "rows_validation.json",
                     "feature_names.json", "validation_predictions.json", "metadata.json"]
            artifacts = {role: store.register_artifact(folder / role) for role in files}
            for filename, expected in code.items():
                path = ROOT / "src" / filename
                if digest(path.read_bytes().replace(b"\r\n", b"\n")) != expected:
                    raise ValueError("Relational source changed during study")
                artifacts["code/" + filename] = store.register_artifact(path)
            with store.write() as connection:
                connection.execute("INSERT INTO nirs.representations VALUES (%s,%s,%s)", (rid, entry["dataset_version_id"], Jsonb(configuration)))
                for role, artifact in artifacts.items():
                    metadata = {"fit_split_id": entry["split_id"], "fit_fold": 0}
                    if role in ("train.npz", "validation.npz"):
                        metadata.update(report["train_matrix" if role == "train.npz" else "validation_matrix"])
                        metadata.update(row_ids_artifact=artifacts["rows_train.json" if role == "train.npz" else "rows_validation.json"],
                                        encoder_artifact=artifacts["encoder.joblib"], feature_names_artifact=artifacts["feature_names.json"])
                    connection.execute("INSERT INTO nirs.representation_artifacts VALUES (%s,%s,%s,%s)", (rid, artifact, role, Jsonb(metadata)))
            report["artifacts"] = artifacts
            reports.append(report)
            print(f"controlled/{name}: F1={report['metrics']['primary_f1']:.4f}", flush=True)
    study["reports"] = [report for report in study["reports"] if report["candidate"] not in CONFIGURATIONS] + reports
    study["relational_extension"] = VERSION
    save_json(OUTPUT / "representation_study.json", study)


if __name__ == "__main__":
    main()
