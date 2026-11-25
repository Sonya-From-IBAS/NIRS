"""Freeze group-aware model-development and evaluation assignments in PostgreSQL."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from split_protocol import (SEED, VERSION, canonical, group_folds, group_rows, sha,
                            source_transfer, summary, validate_partition)
from storage import ROOT, Storage, file_hash

OUTPUT = ROOT / "results/stage6"
PART_CODES = {"train": "T", "validation": "V", "test": "E", "excluded": "X"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_compressed(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((canonical(row) + "\n").encode("utf-8"))
    return file_hash(path)


def read_compressed(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream)


def load_rows(store):
    reports = {"legacy": "results/stage4/cleaned_reference.json", "sqlstateguard": "results/stage4/sqlstateguard_import.json",
               "controlled": "results/stage5/import_report.json"}
    datasets = {key: json.loads((ROOT / path).read_text(encoding="utf-8"))["dataset_version_id"] for key, path in reports.items()}
    rows = []
    with store.connect() as connection:
        for cohort, version in datasets.items():
            values = connection.execute("""SELECT d.observation_id AS id,o.sql_text AS query,
                o.grouping_json AS grouping,o.context_json->'dialect'->>'value' AS dialect,
                c.source_name AS source,c.source_label,a.target_label AS target,d.ordinal
                FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
                JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
                JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
                WHERE d.dataset_version_id=%s ORDER BY d.ordinal""", (version,)).fetchall()
            if not values:
                raise ValueError("Missing cohort: " + cohort)
            for row in values:
                row.update(cohort=cohort, dataset_version_id=version,
                           label=int(row["source_label"]["label"]) if cohort == "legacy" else row["target"])
                if row["label"] is None:
                    raise ValueError("Reference labels are required")
                if cohort == "controlled":
                    row["threat"] = "privilege" if row["grouping"]["group_id"].startswith("s5-family:p") else "exfiltration"
                    row["block"] = "obfuscated" if row["grouping"]["parent_observation_ids"] else row["threat"]
            rows.extend(values)
        extra = connection.execute("""SELECT m.observation_id,m.group_id FROM nirs.group_members m
            JOIN nirs.sample_groups g USING(group_id) WHERE g.kind='controlled_component'""").fetchall()
    by_id = {row["id"]: row for row in rows}
    for item in extra:
        if item["observation_id"] in by_id:
            by_id[item["observation_id"]].setdefault("extra_groups", []).append(item["group_id"])
    return datasets, rows


def row_folds(rows, count):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row["id"])
    return {oid: index % count for label, ids in sorted(by_label.items())
            for index, oid in enumerate(sorted(ids, key=lambda value: sha(f"{SEED}:row:{value}")))}


def controlled_stratum(row):
    return row["threat"] + ":" + str(row["label"])


def cross_application_audit(rows):
    references = defaultdict(lambda: defaultdict(set))
    for row in rows:
        scope = row["cohort"] + (":" + row["source"] if row["cohort"] == "sqlstateguard" else "")
        for key in ("exact_sha256", "template_sha256", "group"):
            references[scope][key].add(row[key])
    scopes = sorted(references)
    return [{"first": first, "second": second,
             **{key: len(references[first][key] & references[second][key]) for key in ("exact_sha256", "template_sha256", "group")}}
            for index, first in enumerate(scopes) for second in scopes[index + 1:]]


def persist_groups(store, rows, artifact_hash):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row["id"])
    with store.write() as connection:
        existing = {row["group_id"] for row in connection.execute("SELECT group_id FROM nirs.sample_groups WHERE group_id LIKE 'g6_%'")}
        new = set(grouped) - existing
        store._copy(connection, "sample_groups", ["group_id", "kind", "metadata"],
                    [(key, "stage6_component", Jsonb({"grouping_version": VERSION, "group_map_artifact": artifact_hash})) for key in sorted(new)])
        store._copy(connection, "group_members", ["group_id", "observation_id"],
                    [(key, oid) for key in sorted(new) for oid in grouped[key]])
        actual = connection.execute("SELECT group_id,observation_id FROM nirs.group_members WHERE group_id=ANY(%s)", (list(grouped),)).fetchall()
    if len(actual) != len(rows) or {item["observation_id"]: item["group_id"] for item in actual} != {row["id"]: row["group"] for row in rows}:
        raise ValueError("Persisted global components differ from the frozen group map")


def persist(store, name, rows, assignments, base_spec, group_file_hash, *, role, parent_scope=None, diagnostic=False):
    scope = rows[0]["cohort"]
    specification = {"protocol_version": "stage6-evaluation-v1", "name": name, "cohort": scope,
                     "role": role, "seed": SEED, "grouping_version": VERSION,
                     "group_map_sha256": group_file_hash, "part_codes": PART_CODES,
                     "model_selection_metric": "macro_f1_5class" if scope == "legacy" else "binary_f1_positive_1",
                     "eligible_for_primary_ranking": not diagnostic, **base_spec}
    if parent_scope:
        specification["parent_scope"] = parent_scope
    required = ("train", "validation") if role == "inner_tuning" else ("train", "test")
    audits = {str(fold): validate_partition(rows, members, required=required, enforce_groups=not diagnostic)
              for fold, members in assignments.items()}
    if scope == "controlled":
        for fold, members in assignments.items():
            for part in required:
                if {row["threat"] for row in rows if members[row["id"]] == part} != {"privilege", "exfiltration"}:
                    raise ValueError("A controlled fold must represent both threat directions")
    path = OUTPUT / "memberships" / (name + ".jsonl.gz")
    ordered_folds = sorted(assignments)
    checksum = write_compressed(path, ({"id": row["id"], "roles": "".join(PART_CODES[assignments[fold][row["id"]]] for fold in ordered_folds)} for row in rows))
    specification.update(membership_artifact=path.relative_to(ROOT).as_posix(), membership_artifact_sha256=checksum,
                         fold_order=ordered_folds)
    saved = store.save_split(rows[0]["dataset_version_id"], specification, assignments)
    # Compare every persisted assignment and the public ID manifest, not samples.
    decoded = list(read_compressed(path))
    inverse = {value: key for key, value in PART_CODES.items()}
    for position, fold in enumerate(ordered_folds):
        exported = {row["id"]: inverse[row["roles"][position]] for row in decoded}
        if exported != assignments[fold]:
            raise ValueError("Exported membership differs")
    with store.connect() as connection:
        actual = connection.execute("SELECT fold,observation_id,part FROM nirs.split_members WHERE split_id=%s", (saved["split_id"],)).fetchall()
    if len(actual) != sum(len(x) for x in assignments.values()) or any(assignments[row["fold"]].get(row["observation_id"]) != row["part"] for row in actual):
        raise ValueError("Database membership differs from the validated plan")
    artifact_id = store.register_artifact(path)
    if artifact_id != checksum:
        raise ValueError("Registered membership checksum differs")
    result = {**{key: value for key, value in saved.items() if key != "reused"}, "name": name, "cohort": scope, "role": role, "scope_rows": len(rows),
              "folds": audits, "membership_artifact": path.relative_to(ROOT).as_posix(),
              "membership_artifact_sha256": checksum, "parent_scope": parent_scope,
              "eligible_for_primary_ranking": not diagnostic}
    print(f"Frozen {name}: {len(rows)} observations, {len(assignments)} folds", flush=True)
    return result


def reserve_common_sets(rows, registry, plans):
    by_id = {row["id"]: row for row in rows}
    public_outer = registry["transfer-web_app-to-wordpress"]
    controlled_outer = registry["controlled-outer"]
    selected = []
    for name, task in (("transfer-web_app-to-wordpress", "sqli"), ("controlled-outer", None)):
        for oid, part in plans[name][0].items():
            if part == "test":
                row = by_id[oid]
                selected.append({"observation_id": oid, "dataset_version_id": row["dataset_version_id"],
                                 "split_id": registry[name]["split_id"], "fold": 0,
                                 "task": task or row["threat"], "target": row["label"], "group_id": row["group"]})
    selected.sort(key=lambda item: (item["task"], item["target"], sha(f"{SEED}:common:{item['observation_id']}")))
    ranks = Counter()
    for row in selected:
        key = row["task"] + ":" + str(row["target"])
        row["stratum"] = key
        row["rank_in_stratum"] = ranks[key]
        ranks[key] += 1
    pool_hash = write_compressed(OUTPUT / "common_test_pool.jsonl.gz", selected)
    pilot, shots = [], []
    for name, task in (("transfer-web_app-to-wordpress-inner", "sqli"), ("controlled-inner-0", None)):
        assignment = plans[name][0]
        for actual_task in ([task] if task else ["privilege", "exfiltration"]):
            for target in (0, 1):
                for part, amount, destination in (("validation", 5, pilot), ("train", 1, shots)):
                    candidates = [by_id[oid] for oid, role in assignment.items() if role == part
                                  and by_id[oid]["label"] == target and (task or by_id[oid]["threat"] == actual_task)]
                    candidates.sort(key=lambda row: sha(f"{SEED}:development:{row['id']}"))
                    if len(candidates) < amount:
                        raise ValueError("Not enough development candidates")
                    for row in candidates[:amount]:
                        destination.append({"observation_id": row["id"], "dataset_version_id": row["dataset_version_id"],
                                            "split_id": registry[name]["split_id"], "fold": 0, "part": part,
                                            "task": actual_task, "target": target, "group_id": row["group"]})
    sets = [{row["group_id"] for row in values} for values in (selected, pilot, shots)]
    if any(sets[first] & sets[second] for first, second in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("LLM test, development and few-shot groups overlap")
    development = {"purpose": "development_only_not_final_test", "pilot": pilot, "few_shot_candidates": shots,
                   "pilot_rows": len(pilot), "few_shot_candidate_rows": len(shots),
                   "common_test_group_overlap": 0, "pilot_few_shot_group_overlap": 0}
    save_json(OUTPUT / "llm_development.json", development)
    return {"pool_rows": len(selected), "strata": dict(ranks), "pool_sha256": pool_hash,
            "pool_artifact": "results/stage6/common_test_pool.jsonl.gz", "pilot_rows": len(pilot),
            "few_shot_candidate_rows": len(shots), "main_LLM_sample_size": "set_after_the_cost_pilot_before_main_predictions",
            "selection_rule": "fixed stratum ordering; agreed quotas select prefixes without prediction scores",
            "paired_comparison": "all methods scored on the exact same selected observation IDs and order"}


def main():
    store = Storage()
    store.migrate()
    datasets, rows = load_rows(store)
    print(f"Loaded {len(rows)} frozen observations", flush=True)
    grouping = group_rows(rows)
    print(f"Built {grouping['groups']} global components", flush=True)
    group_map_hash = write_compressed(OUTPUT / "groups.jsonl.gz", ({"id": row["id"], "cohort": row["cohort"],
        "group": row["group"], "source": row["source"], "label": row["label"]} for row in rows))
    if store.register_artifact(OUTPUT / "groups.jsonl.gz") != group_map_hash:
        raise ValueError("Group artifact hash differs")
    persist_groups(store, rows, group_map_hash)
    cohorts = {name: [row for row in rows if row["cohort"] == name] for name in datasets}
    grouping.update(cohorts={name: summary(part) for name, part in cohorts.items()}, cross_cohort_overlaps=cross_application_audit(rows))
    save_json(OUTPUT / "group_audit.json", grouping)
    registry, plans = {}, {}
    common = {"threshold_and_hyperparameter_selection": "inner_validation_F1_only", "tie_breakers": ["lower_FPR", "higher_recall", "simpler_configuration"],
              "reference_policy": "pinned_annotation"}
    for cohort, n_outer, n_inner in (("legacy", 5, 3), ("controlled", 3, 2)):
        values = cohorts[cohort]
        stratum = controlled_stratum if cohort == "controlled" else None
        test_folds = group_folds(values, n_outer, strata=stratum)
        outer = {fold: {row["id"]: "test" if test_folds[row["id"]] == fold else "train" for row in values} for fold in range(n_outer)}
        name = cohort + "-outer"
        spec = {**common, "reference_policy": "original_five_class_source_reference" if cohort == "legacy" else "pinned_annotation",
                "outer_folds": n_outer, "inner_folds": n_inner, "evaluation_role": "known_source_grouped_evaluation"}
        registry[name] = persist(store, name, values, outer, spec, group_map_hash, role="outer_evaluation")
        plans[name] = outer
        for fold in range(n_outer):
            train = [row for row in values if outer[fold][row["id"]] == "train"]
            inner_ids = group_folds(train, n_inner, strata=stratum)
            assignments = {inner: {row["id"]: "validation" if inner_ids[row["id"]] == inner else "train" for row in train} for inner in range(n_inner)}
            inner_name = cohort + "-inner-" + str(fold)
            parent = {"split_id": registry[name]["split_id"], "fold": fold, "part": "train"}
            registry[inner_name] = persist(store, inner_name, train, assignments, spec, group_map_hash,
                                           role="inner_tuning", parent_scope=parent)
            plans[inner_name] = assignments
    legacy = cohorts["legacy"]
    row_ids = row_folds(legacy, 5)
    diagnostic = {fold: {row["id"]: "test" if row_ids[row["id"]] == fold else "train" for row in legacy} for fold in range(5)}
    registry["legacy-row-control"] = persist(store, "legacy-row-control", legacy, diagnostic,
        {"reference_policy": "original_five_class_source_reference", "purpose": "random_vs_grouped_overfit_diagnostic"},
        group_map_hash, role="diagnostic_random_control", diagnostic=True)
    plans["legacy-row-control"] = diagnostic
    public = cohorts["sqlstateguard"]
    for source, target in (("web_app", "wordpress"), ("wordpress", "web_app")):
        name = f"transfer-{source}-to-{target}"
        outer = {0: source_transfer(public, "sqlstateguard:" + target)}
        spec = {**common, "source_application": source, "target_application": target,
                "exclusion_rule": "source-side groups also present in target are excluded; target is preserved in full",
                "frequency_policy": "unique_SQL_per_application_primary; original_multiplicity_secondary",
                "evaluation_role": "known_application_transfer_control"}
        registry[name] = persist(store, name, public, outer, spec, group_map_hash, role="source_transfer")
        plans[name] = outer
        train = [row for row in public if outer[0][row["id"]] == "train"]
        inner_ids = group_folds(train, 3)
        assignments = {fold: {row["id"]: "validation" if inner_ids[row["id"]] == fold else "train" for row in train} for fold in range(3)}
        parent = {"split_id": registry[name]["split_id"], "fold": 0, "part": "train"}
        registry[name + "-inner"] = persist(store, name + "-inner", train, assignments, spec, group_map_hash,
                                            role="inner_tuning", parent_scope=parent)
        plans[name + "-inner"] = assignments
    paired = reserve_common_sets(rows, registry, plans)
    save_json(OUTPUT / "common_comparison.json", paired)
    save_json(OUTPUT / "split_registry.json", {"version": "stage6-evaluation-v1", "datasets": datasets,
        "protocols": registry, "common_comparison": paired, "models_fitted": 0, "LLM_requests": 0})
    # The registry is reproducible: a second run may reuse all rows, while the
    # persisted specification and membership remain unchanged.
    paths = [OUTPUT / "groups.jsonl.gz", OUTPUT / "common_test_pool.jsonl.gz", OUTPUT / "llm_development.json"]
    artifacts = [{"path": path.relative_to(ROOT).as_posix(), "artifact_id": store.register_artifact(path)} for path in paths]
    save_json(OUTPUT / "storage_artifacts.json", artifacts)
    print(json.dumps({"protocols": len(registry), "groups": grouping["groups"], "common_test_pool_rows": paired["pool_rows"],
                      "LLM_development_rows": paired["pilot_rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
