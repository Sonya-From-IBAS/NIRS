"""Versioned source registry and conservative data-selection decisions."""

import csv
import hashlib
import io
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from source_fetch import COMMIT, DESTINATION, REPOSITORY
from storage import Storage

REGISTRY_VERSION = "stage4-2026-11-24-v1"


def audit_mirror(path: Path, legacy_queries: set[str]) -> dict:
    raw = path.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    valid = [row for row in rows if len(row) >= 2 and row[1] in ("0", "1")]
    positives = [row for row in valid if row[1] == "1"]
    return {"raw_records": len(rows), "valid_binary_label_records": len(valid),
        "other_label_records": len(rows) - len(valid),
        "accepted_records_with_nonempty_extra_fields": sum(any(value for value in row[2:]) for row in valid),
        "strip_matches_with_legacy_kaggle": sum(row[0].strip() in legacy_queries for row in valid),
        "positive_records": len(positives),
        "positive_strip_matches": sum(row[0].strip() in legacy_queries for row in positives),
        "sha256_bytes": hashlib.sha256(raw).hexdigest(),
        "schema": "No header; query column 0, source label column 1. Non-binary label records are excluded.",
        "independent_transfer_source": False}


def make_registry(folder: Path) -> dict:
    mirror = json.loads((folder / "sqliv3_overlap.json").read_text(encoding="utf-8"))
    public = json.loads((folder / "sqlstateguard_audit.json").read_text(encoding="utf-8"))
    candidates = [
        {"candidate_id": "nirs-legacy", "role": "historical_five_class_source_reference",
         "selected": True, "version": "dataset.csv byte hash 30ca83aa9c3de1ac079edd22e8dd8975c213406da18f0fa12f87d8d7e134e569",
         "url": "https://www.kaggle.com/datasets/sajid576/sql-injection-dataset",
         "license": "Kaggle API reports Unknown; upstream permissions and lineage require clarification before new redistribution",
         "unit": "mixed full SQL, fragments and ordinary text", "labels": "five original source classes, not newly verified truth",
         "context": "no verified context fields", "generation": "mixed public payload/text data and LLM-authored additions",
         "selection": "32,529 reference records; one blank removed, source labels preserved; no mass manual relabeling"},
        {"candidate_id": "sqlstateguard", "role": "known_application_SQLi_development_and_transfer",
         "selected": True, "version": COMMIT, "url": f"https://github.com/{REPOSITORY}/tree/{COMMIT}",
         "license": "MIT", "license_evidence": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/LICENSE",
         "unit": "captured statement-level application SQL", "labels": "author normal/attack file categories; attempts rather than exploit success",
         "label_evidence": "https://songli.io/papers/SoCC24CR.pdf#page=8", "context": "roles, permissions, operation policy absent",
         "dialect": "MySQL-compatible workload; trusted dialect field is not inferred from the attack label",
         "generation": "laboratory application templates and injection payload sources, not production incident logs",
         "raw_occurrences": public["rows"], "unique_reference_records": sum(sum(x.values()) for x in public["candidate_unique_reference"].values()),
         "selection": "one exact SQL per application; all original occurrences retained; exact label conflicts excluded if present",
         "overlap_audit": "sqlstateguard_audit.json", "new_blind_holdout": False},
        {"candidate_id": "superviz25", "role": "known_external_control_reserved_for_later_evaluation",
         "selected": False, "version": "Zenodo 17086037 v2", "url": "https://zenodo.org/records/17086037",
         "license": "MIT (verified via Zenodo metadata)", "unit": "full_query plus author metadata",
         "labels": "author generated normal/attack origin", "context": "templates and campaigns; verified operational rights not established",
         "dialect": "MySQL", "generation": "synthetic laboratory workload", "file_bytes": 1053264204,
         "file_md5_from_archive": "7fe0fd69a3cde9b57bc4f1e2d19260f8", "current_data_downloaded": False,
         "selection": "already examined in review; source metadata verified, 1 GB CSV deferred by agreed stage-4 scope; not a new blind holdout"},
        {"candidate_id": "sqlshare", "role": "unlabeled_workload_candidate", "selected": False,
         "version": "Data Release 1", "url": "https://uwescience.github.io/sqlshare/data_release.html",
         "license": "author research release with consent and citation instructions; explicit SPDX license not confirmed",
         "unit": "research SQL workload", "labels": "no verified benign/malicious labels",
         "context": "threat policy and authorized-use labels absent", "generation": "user research workload",
         "selection": "not a labeled negative pool; alert fraction must not be called FPR", "current_data_downloaded": False},
        {"candidate_id": "sqliv3", "role": "overlap_control_not_independent_transfer", "selected": False,
         "version": mirror["commit"], "url": "https://github.com/nidnogg/sqliv5-dataset",
         "license": "not established in this audit", "unit": "mixed SQL and payload records with malformed CSV records",
         "labels": "binary source labels only after strict label parsing", "context": "not supplied", "generation": "lineage not independently established",
         "valid_records": mirror["valid_binary_label_records"], "matching_records": mirror["strip_matches_with_legacy_kaggle"],
         "selection": "97.1% accepted records share text with legacy Kaggle; common content is proven, direction of derivation is not"},
        {"candidate_id": "deployment-sqlia-2026", "role": "secondary_candidate_not_selected", "selected": False,
         "version": "Zenodo 20139451 v1", "url": "https://zenodo.org/records/20139451",
         "license": "CC BY 4.0 (verified via Zenodo metadata)", "unit": "not established from the data package in this audit",
         "labels": "author curated SQLi dataset", "context": "not established", "generation": "lineage requires audit",
         "selection": "metadata inspected; full-query unit and independence are not established, so not added to the primary corpus",
         "data_package_bytes": 658235, "current_data_downloaded": False},
        {"candidate_id": "db-forensic-hybrid", "role": "contextual_candidate_with_access_barrier", "selected": False,
         "version": "public dataset card checked 2026-11-24", "url": "https://huggingface.co/datasets/golfoscar/db-forensic-hybrid",
         "license": "CC BY 4.0 per author card", "unit": "MySQL log events and pre-extracted features; raw rows not inspected",
         "labels": "author multi-class attack scenarios", "context": "potentially useful but data contents not verified",
         "generation": "laboratory scenarios per author card", "selection": "gated manual approval; no access request sent and no account information submitted",
         "current_data_downloaded": False},
        {"candidate_id": "superviz26", "role": "related_cross_domain_family_candidate", "selected": False,
         "version": "Zenodo 19627322", "url": "https://zenodo.org/records/19627322",
         "license": "datasheet states MIT; metadata/license consistency not resolved here",
         "unit": "author full-query cross-domain benchmark", "labels": "normal/attack campaign origin",
         "context": "templates and domains, rights/policy not established", "generation": "synthetic, related to Superviz25",
         "selection": "related generator/domain family; not treated as an automatically independent blind holdout; no large files downloaded"},
        {"candidate_id": "role-behavior-paper", "role": "methodological_reference_not_imported_dataset", "selected": False,
         "version": "arXiv 2508.05690v2", "url": "https://arxiv.org/html/2508.05690v2",
         "license": "dataset license not established", "unit": "role-based SQL sequences described in paper",
         "labels": "role/anomaly protocol differs from the present binary threat definition", "context": "role scenarios described",
         "generation": "simulated and application-derived workloads described by authors",
         "selection": "no independently versioned, permitted static data artifact was established in this audit; no author contact"},
    ]
    return {"registry_version": REGISTRY_VERSION, "checked_date": "2026-11-24", "candidates": candidates,
            "scope": "bounded source screening, not an exhaustive claim that no other suitable datasets exist",
            "model_predictions_used_in_source_selection": False, "new_api_classifications": 0}


def save_registry(store: Storage, folder: Path) -> dict:
    registry = make_registry(folder)
    with store.write() as connection:
        for candidate in registry["candidates"]:
            old = connection.execute("SELECT metadata FROM nirs.source_candidates WHERE candidate_id=%s AND registry_version=%s",
                                     (candidate["candidate_id"], REGISTRY_VERSION)).fetchone()
            if old:
                if old["metadata"] != candidate:
                    raise ValueError("Registry revision changed; create a new version")
            else:
                connection.execute("INSERT INTO nirs.source_candidates VALUES (%s,%s,%s)",
                                   (candidate["candidate_id"], REGISTRY_VERSION, Jsonb(candidate)))
    (folder / "source_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2)
                                              + "\n", encoding="utf-8", newline="\n")
    return registry
