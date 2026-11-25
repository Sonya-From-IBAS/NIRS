"""Import controlled observations as bound data and freeze their dataset versions."""

import argparse
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from generate_scenarios import ROOT, VERSION, audit_corpus, family_links, save_json
from storage import Storage, canonical, digest, file_hash


def load_checked():
    folder = ROOT / "data/scenarios"
    catalog_raw = (folder / "catalog_v1.json").read_bytes()
    catalog = json.loads(catalog_raw)
    records = [json.loads(line) for line in (folder / "corpus_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    details = [json.loads(line) for line in (folder / "static_checks_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = digest(catalog_raw)
    if any(record["source"]["source_file_sha256"] != expected for record in records):
        raise ValueError("Observations do not identify the loaded source catalog")
    audit_corpus(catalog, records, details)
    return catalog, records, details


def ingest(store: Storage) -> dict:
    catalog, records, details = load_checked()
    source_id = "controlled-stage5"
    with store.connect() as connection:
        before = {row["source_version_id"] for row in connection.execute(
            "SELECT source_version_id FROM nirs.source_versions WHERE source_id=%s", (source_id,))}
        existing_sql = {row["sql_sha256"] for row in connection.execute(
            "SELECT sql_sha256 FROM nirs.observations o WHERE NOT EXISTS (SELECT 1 FROM nirs.occurrences c WHERE c.observation_id=o.observation_id AND c.source_name=%s)", (source_id,))}
    metadata = {"kind": "controlled_scenarios", "authoring": "AI-authored templates and deterministic expansion",
                "annotation_protocol": VERSION, "verification_method": "static_scenario_and_policy",
                "role": "contextual_privilege_exfiltration_and_representation_control",
                "dialects": ["mysql", "postgresql", "mssql", "oracle"]}
    with store.write() as connection:
        connection.execute("INSERT INTO nirs.sources VALUES (%s,%s) ON CONFLICT DO NOTHING", (source_id, Jsonb(metadata)))
    version = store.import_records(records, source_id=source_id, version=VERSION)
    dataset = store.create_verified_source_dataset([version], "controlled-stage5-v1",
        {"label_basis": "declared_scenario_and_static_SQL_policy_check", "generation_protocol": VERSION,
         "rows": 1200, "origin": "AI_authored_controlled", "source_groups": "conservative_family_components",
         "purpose": "laboratory_contextual_and_representation_control"})
    by_block = {}
    for block in ("privilege", "exfiltration", "obfuscated"):
        ids = [row["observation_id"] for row in details if row["block"] == block]
        by_block[block] = store.create_subset(dataset, ids, "controlled-stage5-" + block + "-v1",
            {"generation_block": block, "label_basis": "declared_scenario_and_static_SQL_policy_check", "rows": len(ids)})
    components = family_links(records)
    with store.write() as connection:
        for component in components:
            connection.execute("INSERT INTO nirs.sample_groups VALUES (%s,'controlled_component',%s) ON CONFLICT DO NOTHING",
                               (component["group_id"], Jsonb({"families": component["family_group_ids"], "basis": component["basis"]})))
            with connection.cursor() as cursor:
                cursor.executemany("INSERT INTO nirs.group_members VALUES (%s,%s) ON CONFLICT DO NOTHING",
                                   [(component["group_id"], oid) for oid in component["observation_ids"]])
    artifact_paths = ["data/scenarios/catalog_v1.json", "data/scenarios/corpus_v1.jsonl",
                      "data/scenarios/static_checks_v1.jsonl", "data/scenarios/generation_plan_v2.json",
                      "src/generate_scenarios.py", "src/scenario_sql.py", "src/import_scenarios.py",
                      "results/stage5/group_links.json"]
    artifacts = []
    for relative in artifact_paths:
        artifact = store.register_artifact(ROOT / relative)
        with store.write() as connection:
            connection.execute("INSERT INTO nirs.source_artifacts VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                               (version, artifact, relative))
        artifacts.append({"path": relative, "artifact_id": artifact})
    restored = store.dataset_records(dataset)
    if restored != records:
        raise ValueError("Imported observation contract differs on full roundtrip")
    # Re-query counts, rather than reporting only the proposed input size.
    with store.connect() as connection:
        actual = connection.execute("""SELECT count(*) AS occurrences,count(DISTINCT observation_id) AS observations
            FROM nirs.occurrences WHERE source_version_id=%s""", (version,)).fetchone()
        labels = connection.execute("""SELECT a.target_label,count(*) AS rows FROM nirs.dataset_members d
            JOIN nirs.annotations a ON a.annotation_id=d.annotation_id WHERE d.dataset_version_id=%s
            GROUP BY a.target_label ORDER BY a.target_label""", (dataset,)).fetchall()
        historical = connection.execute("""SELECT
            (SELECT count(*) FROM nirs.experiment_runs) AS experiment_runs,
            (SELECT count(*) FROM nirs.predictions) AS predictions""").fetchone()
    if actual != {"occurrences": 1200, "observations": 1200} or labels != [{"target_label": 0, "rows": 600}, {"target_label": 1, "rows": 600}]:
        raise ValueError("Persisted counts differ from the accepted corpus")
    return {"status": "passed", "source_version_id": version, "dataset_version_id": dataset,
            "block_dataset_versions": by_block, "rows": len(records), "counts_in_database": actual,
            "target_counts_in_database": labels, "full_contract_roundtrip": True,
            "reused_source_version": version in before, "source_artifacts": artifacts,
            "linked_components": len(components),
            "exact_SQL_matches_to_previous_sources": sum(digest(row["sql"]["text"].encode()) in existing_sql for row in records),
            "historical_experiments": historical}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/stage5/import_report.json")
    args = parser.parse_args()
    result = ingest(Storage())
    save_json(args.report, result)
    print(json.dumps(result, indent=2))
