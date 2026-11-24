"""Audited SQLStateGuard release adapter, preserving all original occurrences."""

import json
from pathlib import Path
import uuid

from psycopg.types.json import Jsonb

from data_audit import lexical_template, sha
from data_contract import validate_observation
from public_audit import source_rows
from source_fetch import COMMIT, DESTINATION, REPOSITORY
from storage import Storage, canonical, digest, unknown_context, file_hash as hash_file

PROTOCOL = "sqlstateguard-release-labels-v1"
PAPER = "https://doi.org/10.1145/3698038.3698569"


def make_record(row: dict, version_id: str, file_hash: str, conflict: bool) -> dict:
    label = row["label"]
    oid = "obs_sg_" + sha(version_id + ":" + sha(row["query"]) + ":" + str(label))
    annotation_id = oid + "_a1"
    template_id = "lexical-v1:" + sha(lexical_template(row["query"]))
    annotation = {
        "annotation_id": annotation_id, "revision": 1, "supersedes": None,
        "instruction_version": "1.0.0", "status": "needs_review" if conflict else "verified",
        "target_label": None if conflict else label,
        "threat_tags": ["sqli"] if label == 1 and not conflict else [],
        "representation_tags": [], "exfiltration_channels": [],
        "reason": "Exact SQL has conflicting source labels and cannot supply query-only ground truth." if conflict else
                  "Reference follows the author's released normal/attack file category for captured application SQL; no claim of successful exploitation or independent human review.",
        "evidence": [{"kind": "source_documentation", "reference": PAPER + "#section-4.1",
                      "statement": "Author statement-level release; normal files map to 0, attack-family files to 1. Failed attempts remain attacks."}],
        "reviewer": {"kind": "rule", "reviewer_id": PROTOCOL},
    }
    return {"schema_version": "1.0.0", "observation_id": oid,
        "sql": {"text": row["query"], "unit_kind": "full_sql"}, "context": unknown_context(),
        "source": {"source_id": "sqlstateguard:" + row["application"], "source_version": COMMIT,
            "source_record_id": row["source_record_id"], "source_file_sha256": file_hash,
            "origin": "public", "purpose": "research", "source_label": {"label": label, "attack_family": row["attack_family"]},
            "generator": None},
        "annotations": [annotation], "current_annotation_id": annotation_id,
        "grouping": {"group_id": template_id, "scenario_id": None, "template_id": template_id,
                     "campaign_id": None, "parent_observation_ids": []},
        "execution": {"outcome": "unknown", "evidence_ref": None}}


def ingest_release(store: Storage, audit_report: dict) -> dict:
    manifest = json.loads((DESTINATION / "download_manifest.json").read_text(encoding="utf-8"))
    files = {item["path"]: item for item in manifest["files"]}
    if audit_report["source_commit"] != COMMIT or audit_report["download_manifest_sha256"] != hash_file(DESTINATION / "download_manifest.json"):
        raise ValueError("Source audit does not identify this pinned package")
    for relative, item in files.items():
        if hash_file(DESTINATION / relative) != item["sha256_bytes"]:
            raise ValueError("Downloaded source changed after its audit")
    conflicts = {item["sha256"] for item in audit_report["exact_mixed_label_examples"]}
    results = []
    for app in ("web_app", "wordpress"):
        source_id = "sqlstateguard:" + app
        package = [item for key, item in files.items() if key.startswith(app + "/")]
        package_hash = digest(canonical({"commit": COMMIT, "files": package, "protocol": PROTOCOL}).encode())
        version_id = "src_" + sha(source_id + ":" + package_hash)
        total = accepted = rejected = 0
        seen, checked_cases = {}, set()
        def attach_files():
            for item in package + [files["README.md"], files["LICENSE"]]:
                artifact = store.register_artifact(DESTINATION / item["path"])
                with store.write() as linked:
                    linked.execute("INSERT INTO nirs.source_artifacts VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                                   (version_id, artifact, item["path"]))
        with store.connect() as opened:
            existing = opened.execute("SELECT * FROM nirs.source_versions WHERE source_version_id=%s", (version_id,)).fetchone()
        if existing:
            # A previous process may have committed rows before copying files.
            # Re-import must also finish/repair those artifact registrations.
            attach_files()
            results.append({"application": app, "source_version_id": version_id, "raw_rows": existing["row_count"], "reused": True})
            continue
        with store.write() as connection:
            exists = connection.execute("SELECT * FROM nirs.source_versions WHERE source_version_id=%s", (version_id,)).fetchone()
            if exists:
                raise ValueError("Source version was imported concurrently; rerun to reuse it")
            metadata = {"url": "https://github.com/" + REPOSITORY, "commit": COMMIT, "license": "MIT",
                "citation": PAPER, "unit": "author statement-level captured SQL", "label_protocol": PROTOCOL,
                "generation": "laboratory application templates and injection payload sources",
                "trusted_context": "not supplied", "known_development_source": True}
            connection.execute("INSERT INTO nirs.sources VALUES (%s,%s)", (source_id, Jsonb(metadata)))
            connection.execute("""INSERT INTO nirs.source_versions
                (source_version_id,source_id,file_path,sha256_bytes,sha256_lf,columns_json,row_count,accepted_count,rejected_count)
                VALUES (%s,%s,%s,%s,%s,%s,0,0,0)""", (version_id, source_id, "sqlstateguard/" + COMMIT + "/" + app,
                package_hash, package_hash, Jsonb(["one captured SQL per line", "source file category"])))
            new_records, duplicates = [], []

            def flush():
                if new_records:
                    store._store_batch(connection, new_records, version_id)
                    groups = {record["grouping"]["template_id"] for record, _, _, _ in new_records}
                    with connection.cursor() as cursor:
                        cursor.executemany("INSERT INTO nirs.sample_groups VALUES (%s,'lexical_template',%s) ON CONFLICT DO NOTHING",
                            [(key, Jsonb({"method": "lexical-v1", "semantic_equivalence_claimed": False})) for key in sorted(groups)])
                    store._copy(connection, "group_members", ["group_id", "observation_id"],
                        [(record["grouping"]["template_id"], record["observation_id"]) for record, _, _, _ in new_records])
                if duplicates:
                    store._copy(connection, "occurrences", ["source_version_id", "record_id", "observation_id", "source_name",
                        "source_label", "source_json", "raw_record", "ordinal"], duplicates)
                new_records.clear()
                duplicates.clear()

            for row in source_rows():
                if row["application"] != app:
                    continue
                ordinal = total
                total += 1
                query = row["query"]
                if not query.strip() or "\x00" in query:
                    connection.execute("INSERT INTO nirs.rejected_records VALUES (%s,%s,%s,%s)",
                        (version_id, row["source_record_id"], "blank_or_nul", json.dumps(row, ensure_ascii=True)))
                    rejected += 1
                    continue
                fingerprint = (sha(query), row["label"])
                original = {"query": query, "source_file": row["source_file"], "ordinal_in_file": row["ordinal_in_file"],
                            "source_category": row["attack_family"]}
                file_hash = files[row["source_file"]]["sha256_bytes"]
                if fingerprint not in seen:
                    record = make_record(row, version_id, file_hash, fingerprint[0] in conflicts)
                    # The adapter constructs a fixed contract shape. Validate each
                    # label/conflict case; all variable SQL values are bound data.
                    case = (row["label"], fingerprint[0] in conflicts)
                    if case not in checked_cases:
                        validate_observation(record)
                        checked_cases.add(case)
                    seen[fingerprint] = record["observation_id"]
                    new_records.append((record, original, source_id, ordinal))
                else:
                    source = {"source_id": source_id, "source_version": COMMIT, "source_record_id": row["source_record_id"],
                        "source_file_sha256": file_hash, "origin": "public", "purpose": "research",
                        "source_label": {"label": row["label"], "attack_family": row["attack_family"]}, "generator": None}
                    duplicates.append((version_id, row["source_record_id"], seen[fingerprint], source_id,
                        Jsonb(source["source_label"]), Jsonb(source), Jsonb(original), ordinal))
                accepted += 1
                if len(new_records) + len(duplicates) >= 1000:
                    flush()
                if total % 50000 == 0:
                    print(f"Imported {app}: {total} occurrences", flush=True)
            flush()
            for item in package:
                if hash_file(DESTINATION / item["path"]) != item["sha256_bytes"]:
                    raise ValueError("Source file changed during import")
            connection.execute("UPDATE nirs.source_versions SET row_count=%s,accepted_count=%s,rejected_count=%s WHERE source_version_id=%s",
                               (total, accepted, rejected, version_id))
            connection.execute("INSERT INTO nirs.import_jobs(job_id,source_id,source_version_id,status,details,finished_at) VALUES (%s,%s,%s,'complete',%s,now())",
                (uuid.uuid4().hex, source_id, version_id, Jsonb({"adapter": PROTOCOL, "rows": total, "unique_observations": len(seen)})))
        attach_files()
        results.append({"application": app, "source_version_id": version_id, "raw_rows": total,
                        "unique_observations": len(seen), "rejected": rejected, "reused": False})
    dataset = store.create_verified_source_dataset([item["source_version_id"] for item in results],
        "sqlstateguard-sqli-reference-v1", {"label_basis": "author_release_protocol", "source_protocol": PROTOCOL,
            "exact_duplicates": "one observation per application, SQL and source label; all occurrences retained",
            "exact_label_conflicts": "excluded from verified query-only cohort", "new_blind_holdout": False,
            "applications": ["web_app", "wordpress"], "synthetic_lab_data": True})
    return {"sources": results, "dataset_version_id": dataset, "protocol": PROTOCOL}
