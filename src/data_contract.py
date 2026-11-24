"""Observation contract v1; no SQL execution, relabeling or storage migration."""

from __future__ import annotations

from copy import deepcopy
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
CONTEXT_FIELDS = ("dialect", "active_role", "permissions", "application_operation", "policy")
SCHEMA_PATH = ROOT / "schemas" / "observation-v1.schema.json"
SUBSTANTIVE_EVIDENCE = {
    "source_documentation", "scenario_definition", "policy_check",
    "independent_review", "attack_trace",
}


def validate_observation(record: dict) -> None:
    """Check structure and cross-field invariants; does not establish SQL semantics."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)
    context = record["context"]
    for field in CONTEXT_FIELDS:
        entry = context[field]
        if entry["availability"] == "known":
            if entry["value"] is None or not entry["evidence_ref"] or not entry["available_before_decision"]:
                raise ValueError(f"{field}: known context needs a value and pre-decision evidence")
        elif (entry["value"] is not None or entry["evidence_ref"] is not None
              or entry["available_before_decision"]):
            raise ValueError(f"{field}: unavailable context must not contain a value or evidence")

    annotations = record["annotations"]
    identifiers = [annotation["annotation_id"] for annotation in annotations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate annotation IDs")
    for index, annotation in enumerate(annotations):
        expected_previous = identifiers[index - 1] if index else None
        if annotation["revision"] != index + 1 or annotation["supersedes"] != expected_previous:
            raise ValueError("Annotation revisions must form an ordered, unbroken history")
        verified = annotation["status"] == "verified"
        label = annotation["target_label"]
        if verified != (label is not None):
            raise ValueError("Only verified annotations have a binary target")
        if not verified and (annotation["threat_tags"] or annotation["exfiltration_channels"]):
            raise ValueError("Unverified annotations must not expose tentative tags as ground truth")
        if verified:
            evidence_kinds = {item["kind"] for item in annotation["evidence"]}
            if not evidence_kinds & SUBSTANTIVE_EVIDENCE or annotation["reviewer"]["kind"] == "unreviewed":
                raise ValueError("A source label, generator response or syntax signal alone is not verification")
            if label == 0 and (annotation["threat_tags"] or annotation["exfiltration_channels"]):
                raise ValueError("Benign observations cannot have threat ground-truth tags")
            if label == 1 and not annotation["threat_tags"]:
                raise ValueError("Verified malicious observations need an in-scope threat tag")
        has_exfiltration = "exfiltration" in annotation["threat_tags"]
        if has_exfiltration != bool(annotation["exfiltration_channels"]):
            raise ValueError("Exfiltration tags and mechanism descriptions must agree")
    if record["current_annotation_id"] != identifiers[-1]:
        raise ValueError("Current annotation must reference the latest revision")
    if record["sql"]["unit_kind"] == "full_sql" and not record["sql"]["text"].strip():
        raise ValueError("Blank text cannot be marked as full SQL")
    if record["execution"]["outcome"] in ("succeeded", "failed") and not record["execution"]["evidence_ref"]:
        raise ValueError("An observed execution outcome requires independent evidence")


def current_annotation(record: dict) -> dict:
    validate_observation(record)
    return deepcopy(record["annotations"][-1])


def detector_input(record: dict, *, include_context: bool = False) -> dict:
    """Whitelist model input; never serialize annotations or the complete record."""
    validate_observation(record)
    result = {"sql": record["sql"]["text"]}
    if include_context:
        result["context"] = {
            field: {"availability": record["context"][field]["availability"],
                    "value": deepcopy(record["context"][field]["value"])}
            for field in CONTEXT_FIELDS
        }
        policy = result["context"]["policy"]["value"]
        if policy is not None:
            # Version keys establish provenance; the detector needs the rules.
            policy.pop("policy_id")
            policy.pop("version")
    return result


def primary_eligible(record: dict) -> bool:
    validate_observation(record)
    return (record["source"]["purpose"] == "research"
            and record["sql"]["unit_kind"] == "full_sql"
            and record["annotations"][-1]["status"] == "verified")


def paired_context_eligible(record: dict) -> bool:
    """Initial strict cohort; partial-context analyses require a separate protocol."""
    return primary_eligible(record) and all(
        record["context"][field]["availability"] == "known" for field in CONTEXT_FIELDS
    )


def context_availability(records: Iterable[dict]) -> dict:
    counts = {field: {status: 0 for status in ("known", "unknown", "not_applicable")}
              for field in CONTEXT_FIELDS}
    total = 0
    for record in records:
        validate_observation(record)
        total += 1
        for field in CONTEXT_FIELDS:
            counts[field][record["context"][field]["availability"]] += 1
    return {"records": total, "fields": counts}


def audit_legacy_csv(path: Path) -> dict:
    """Report absent context in the old four-column CSV without inventing values."""
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if set(columns or []) != {"query", "label", "source", "label_name"}:
        raise ValueError("This audit only supports the original four-column dataset")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed legacy CSV record")

    def absent(count: int) -> dict:
        return {field: {"known": 0, "unknown": count, "not_applicable": 0}
                for field in CONTEXT_FIELDS}

    sources = Counter(row["source"] for row in rows)
    return {
        "audit_version": "1.0.0", "input_file": path.name,
        "sha256_bytes": hashlib.sha256(raw).hexdigest(),
        "sha256_crlf_to_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "columns": columns, "records": len(rows),
        "legacy_label_counts": dict(sorted(Counter(row["label_name"] for row in rows).items())),
        "context_availability": absent(len(rows)),
        "by_source": {source: {"records": count, "context_availability": absent(count)}
                      for source, count in sorted(sources.items())},
        "inference_rule": "Only explicit verified context counts as known; source name tokens are not verified context.",
        "recommended_initial_import_status": "needs_review",
        "target_labels_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a record or an object containing records")
    validate.add_argument("path", type=Path)
    audit = commands.add_parser("audit-legacy", help="Audit missing context in the historical CSV")
    audit.add_argument("path", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        value = json.loads(args.path.read_text(encoding="utf-8"))
        records = value["records"] if "records" in value else [value]
        for record in records:
            validate_observation(record)
        result = {"status": "passed", "records": len(records),
                  "annotation_statuses": dict(Counter(record["annotations"][-1]["status"] for record in records)),
                  "primary_research_eligible": sum(primary_eligible(record) for record in records),
                  "context_availability": context_availability(records)}
    else:
        result = audit_legacy_csv(args.path)
        if args.output:
            if args.output.resolve() == args.path.resolve():
                raise ValueError("Audit output must not overwrite the source CSV")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                                   + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
