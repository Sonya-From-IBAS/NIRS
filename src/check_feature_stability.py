"""Fixed-label-stratified validation stress tests of archived feature pipelines."""

from collections import Counter, defaultdict
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from sql_features import prepare_input
from storage import ROOT, Storage, digest
from study_features import renamed_input, reformatted_input, save_json


def select_validation(prediction, cohort):
    pairs = list(zip(prediction["observation_ids"], prediction["truth"]))
    if cohort == "controlled":
        return [oid for oid, _ in pairs]
    groups = defaultdict(list)
    for oid, label in pairs:
        groups[label].append(oid)
    quota = 120 // len(groups)
    return [oid for label, ids in sorted(groups.items())
            for oid in sorted(ids, key=lambda value: digest(("20261125:feature-stress:" + value).encode()))[:quota]]


def main():
    store = Storage()
    study = json.loads((ROOT / "results/stage7/representation_study.json").read_text(encoding="utf-8"))
    selected = {}
    with threadpool_limits(limits=1):
        for cohort in ("legacy", "sqlstateguard", "controlled"):
            candidates = [report for report in study["reports"] if report["cohort"] == cohort]
            prediction = json.loads(store.get_artifact(candidates[0]["artifacts"]["validation_predictions.json"]).read_text())
            ids = select_validation(prediction, cohort)
            with store.connect() as connection:
                connection.execute("SET LOCAL max_parallel_workers_per_gather = 0")
                values = connection.execute("""SELECT observation_id,sql_text,context_json FROM nirs.observations
                    WHERE observation_id=ANY(%s)""", (ids,)).fetchall()
            by_id = {value["observation_id"]: value for value in values}
            inputs = [prepare_input({"sql": {"text": by_id[oid]["sql_text"]}, "context": by_id[oid]["context_json"]}) for oid in ids]
            formatted = [reformatted_input(item) for item in inputs]
            renamed = [renamed_input(item) for item in inputs] if cohort == "controlled" else None
            truth = dict(zip(prediction["observation_ids"], prediction["truth"]))
            report = {"observation_ids": ids, "class_counts": dict(Counter(str(truth[oid]) for oid in ids)),
                      "scope": "entire_inner_validation" if cohort == "controlled" else "fixed_hash_label_stratified_inner_validation",
                      "rows": len(ids), "candidates": {}}
            if renamed:
                report["alpha_equivalence"] = {
                    "same_masked_SQL": sum(a.sql["masked"] == b.sql["masked"] for a, b in zip(inputs, renamed)),
                    "same_masked_context": sum(a.context_text == b.context_text for a, b in zip(inputs, renamed))}
            for candidate in candidates:
                for name, expected in candidate["configuration"]["code_sha256_lf"].items():
                    if digest((ROOT / "src" / name).read_bytes().replace(b"\r\n", b"\n")) != expected:
                        raise ValueError("Active feature code differs from the archived representation")
                pipeline = joblib.load(store.get_artifact(candidate["artifacts"]["pipeline.joblib"]))
                original = pipeline.predict(inputs)
                saved = json.loads(store.get_artifact(candidate["artifacts"]["validation_predictions.json"]).read_text())
                expected = dict(zip(saved["observation_ids"], saved["prediction"]))
                if not np.array_equal(original, [expected[oid] for oid in ids]):
                    raise ValueError("Stress-test baseline differs from archived predictions")
                changes = {"formatting": int(np.count_nonzero(pipeline.predict(formatted) != original))}
                if renamed:
                    changes["consistent_renaming"] = int(np.count_nonzero(pipeline.predict(renamed) != original))
                report["candidates"][candidate["candidate"]] = {
                    "representation_id": candidate["representation_id"],
                    "prediction_changes": changes,
                    "agreement": {name: 1 - count / len(ids) for name, count in changes.items()}}
            selected[cohort] = report
    result = {"version": "stage7-stability-v1", "selection": "labels and frozen IDs only, never model scores",
              "reports": selected}
    save_json(ROOT / "results/stage7/stability_audit.json", result)
    print(json.dumps({key: {"rows": value["rows"], "class_counts": value["class_counts"],
                            "alpha_equivalence": value.get("alpha_equivalence")} for key, value in selected.items()}, indent=2))


if __name__ == "__main__":
    main()
