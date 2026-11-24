"""Audit a PostgreSQL source-reference dataset and create approved cohorts."""

import argparse
import csv
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from data_audit import AUDIT_VERSION, audit_rows
from dataset import load_source_reference
from storage import Storage, canonical, digest


def write_coverage(store: Storage, cleaned: str, output: Path) -> None:
    """Describe label coverage without inferring rights or target truth from SQL."""
    frame = load_source_reference(cleaned, store)
    rows = []
    for (source, label), count in frame.groupby(["source", "label_name"]).size().items():
        dialect = next((name for name in ("mysql", "postgresql", "mssql", "oracle")
                        if "_" + name + "_" in source), "not_established")
        rows.append({"corpus": "legacy_source_reference", "source": source, "source_label_or_threat": label,
                     "rows": int(count), "count_unit": "reference_records", "dialect_source_hint": dialect,
                     "trusted_context": "unknown", "label_basis": "original_five_class_source_label",
                     "origin": "llm_generated" if source.startswith("generated_") else "public_mixed",
                     "suitable_for_verified_context_comparison": False})
    report_path = output / "sqlstateguard_audit.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for app, counts in report["candidate_unique_reference"].items():
            for label, count in counts.items():
                rows.append({"corpus": "sqlstateguard", "source": app,
                    "source_label_or_threat": "sqli" if label == "attack" else "benign",
                    "rows": count, "count_unit": "unique_sql_within_application",
                    "dialect_source_hint": "mysql_compatible_workload", "trusted_context": "unknown",
                    "label_basis": "author_release_normal_attack_file_category", "origin": "public_laboratory",
                    "suitable_for_verified_context_comparison": False})
    with (output / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_reference(store: Storage, base: str, output: Path) -> dict:
    frame = load_source_reference(base, store)
    if frame.empty:
        raise ValueError("Cannot audit an empty reference")
    rows = frame[["query", "label", "source", "label_name"]].to_dict("records")
    for row in rows:
        row["label"] = str(row["label"])
    summary, details = audit_rows(rows)
    with store.connect() as connection:
        hashes = connection.execute("""SELECT DISTINCT s.sha256_bytes FROM nirs.dataset_members d
            JOIN nirs.source_versions s USING(source_version_id) WHERE d.dataset_version_id=%s""", (base,)).fetchall()
    if len(hashes) == 1:
        summary["input_sha256_bytes"] = hashes[0]["sha256_bytes"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "legacy_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    csv_path = output / "legacy_row_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
        writer.writeheader()
        for row in details:
            writer.writerow({key: ";".join(value) if isinstance(value, list) else value for key, value in row.items()})
    with csv_path.open(encoding="utf-8", newline="") as stream:
        serialized = list(csv.DictReader(stream))
    config = {"method": AUDIT_VERSION, "diagnostic_only": True, "no_semantic_or_parser_validation": True,
              "code_sha256_lf": digest((Path(__file__).parent / "data_audit.py").read_bytes().replace(b"\r\n", b"\n"))}
    audit_id = "audit_" + digest(canonical({"dataset": base, "configuration": config}).encode())
    with store.write() as connection:
        actual = {row["observation_id"]: row["sql_sha256"] for row in connection.execute("""SELECT o.observation_id,o.sql_sha256
            FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id) WHERE d.dataset_version_id=%s""", (base,))}
        for index, row in enumerate(serialized):
            if actual[frame.iloc[index]["observation_id"]] != row["exact_sha256"]:
                raise ValueError("Audit rows do not match the database SQL")
        existing = connection.execute("SELECT summary,configuration FROM nirs.data_audits WHERE audit_id=%s", (audit_id,)).fetchone()
        if existing:
            if existing["summary"] != summary or existing["configuration"] != config:
                raise ValueError("Immutable audit configuration/content differs")
        else:
            connection.execute("INSERT INTO nirs.data_audits VALUES (%s,%s,%s,%s,%s)", (audit_id, base, AUDIT_VERSION, Jsonb(config), Jsonb(summary)))
            store._copy(connection, "observation_audits", ["audit_id", "dataset_version_id", "observation_id", "diagnostic"],
                [(audit_id, base, frame.iloc[index]["observation_id"], Jsonb(row)) for index, row in enumerate(serialized)])
    keep = [frame.iloc[index]["observation_id"] for index, row in enumerate(serialized) if row["unit_hint"] != "blank"]
    cleaned = store.create_subset(base, keep, "legacy-cleaned-source-reference-v1",
        {"label_basis": "original_five_class_source_reference", "audit_id": audit_id,
         "exclusion_rule": "blank_after_strip_only", "excluded_rows": len(frame) - len(keep),
         "whitespace_variants_retained": True, "source_targets_not_reverified": True})
    result = {"base_dataset_version_id": base, "audit_id": audit_id, "dataset_version_id": cleaned,
        "rows": len(keep), "removed_blank_rows": len(frame) - len(keep),
        "original_five_class_source_labels_preserved": True, "new_verified_binary_ground_truth_claimed": False}
    (output / "cleaned_reference.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, help="Base dataset_version_id")
    parser.add_argument("--output", type=Path, default=Path("results/stage4"))
    parser.add_argument("--include-sqlstateguard", action="store_true", help="Use only after agreeing this source selection")
    args = parser.parse_args()
    store = Storage()
    store.migrate()
    result = {"legacy": audit_reference(store, args.legacy, args.output)}
    if args.include_sqlstateguard:
        from source_ingest import ingest_release
        report = json.loads((args.output / "sqlstateguard_audit.json").read_text(encoding="utf-8"))
        result["sqlstateguard"] = ingest_release(store, report)
        (args.output / "sqlstateguard_import.json").write_text(json.dumps(result["sqlstateguard"], ensure_ascii=False, indent=2)
                                                             + "\n", encoding="utf-8", newline="\n")
    from source_catalog import save_registry
    save_registry(store, args.output)
    write_coverage(store, result["legacy"]["dataset_version_id"], args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
