"""Compare every imported SQLStateGuard occurrence against the pinned raw release."""

from itertools import zip_longest
import hashlib
import json
from pathlib import Path

from public_audit import source_rows
from source_fetch import DESTINATION
from storage import Storage, file_hash


def verify(store: Storage, folder: Path) -> dict:
    imported = json.loads((folder / "sqlstateguard_import.json").read_text(encoding="utf-8"))
    manifest = json.loads((DESTINATION / "download_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if file_hash(DESTINATION / item["path"]) != item["sha256_bytes"]:
            raise ValueError("Raw package no longer matches its download manifest")
    results = []
    for source in imported["sources"]:
        app, version = source["application"], source["source_version_id"]
        expected = (row for row in source_rows() if row["application"] == app)
        count = 0
        fingerprint = hashlib.sha256()
        with store.connect() as connection:
            with connection.cursor(name="roundtrip_" + app) as cursor:
                cursor.execute("""SELECT c.*,o.sql_text FROM nirs.occurrences c
                    JOIN nirs.observations o USING(observation_id)
                    WHERE c.source_version_id=%s ORDER BY c.ordinal""", (version,))
                for raw, actual in zip_longest(expected, cursor):
                    if raw is None or actual is None:
                        raise ValueError("Occurrence counts differ")
                    original = {"query": raw["query"], "source_file": raw["source_file"],
                                "ordinal_in_file": raw["ordinal_in_file"], "source_category": raw["attack_family"]}
                    expected_label = {"label": raw["label"], "attack_family": raw["attack_family"]}
                    if (actual["sql_text"] != raw["query"] or actual["raw_record"] != original
                            or actual["source_label"] != expected_label or actual["ordinal"] != count
                            or actual["record_id"] != raw["source_record_id"]
                            or actual["source_json"]["source_record_id"] != raw["source_record_id"]):
                        raise ValueError(f"Source roundtrip differs at {app}:{count}")
                    content = raw["query"].encode("utf-8")
                    fingerprint.update(len(content).to_bytes(8, "big"))
                    fingerprint.update(content)
                    count += 1
                if count != source["raw_rows"]:
                    raise ValueError("Import manifest count differs")
        results.append({"application": app, "source_version_id": version,
                        "occurrences_checked": count, "ordered_length_prefixed_sql_sha256": fingerprint.hexdigest()})
    with store.connect() as connection:
        tables = connection.execute("""SELECT tablename FROM pg_tables
            WHERE schemaname='nirs' ORDER BY tablename""").fetchall()
        # Fixed table names; no sampled SQL is executed.
        counts = connection.execute("""SELECT
            (SELECT count(*) FROM nirs.observations) AS observations,
            (SELECT count(*) FROM nirs.occurrences) AS occurrences,
            (SELECT count(*) FROM nirs.experiment_runs) AS experiment_runs,
            (SELECT count(*) FROM nirs.predictions) AS predictions,
            (SELECT count(*) FROM nirs.artifacts WHERE status='ready') AS ready_artifacts""").fetchone()
    files = store.verify_artifacts()
    if files["failures"]:
        raise ValueError("Artifact registry verification failed")
    result = {"status": "passed", "sources": results,
              "occurrences_checked": sum(row["occurrences_checked"] for row in results),
              "sql_text_and_source_labels_preserved": True, "sql_trailing_whitespace_preserved": True,
              "table_count": len(tables), "database_counts": counts, "artifact_verification": files,
              "research_sql_executed": False}
    (folder / "roundtrip_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)
                                                    + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Storage(), Path("results/stage4")), ensure_ascii=False, indent=2))
