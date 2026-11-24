"""Audit pinned SQLStateGuard data and overlaps before selecting a corpus."""

from collections import Counter, defaultdict
import csv
import json
import hashlib
from pathlib import Path

from data_audit import diagnose, lexical_template, sha
from source_fetch import DESTINATION, COMMIT

FAMILIES = {"bool": "boolean", "illegal": "error", "normal": "normal",
            "tautology": "tautology", "time": "time", "union": "union"}


def remove_record_terminator(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith(("\r", "\n")):
        return line[:-1]
    return line


def source_rows(root: Path = DESTINATION):
    for path in sorted(root.glob("*/*.txt")):
        family = FAMILIES[path.name.split("_", 1)[0]]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, line in enumerate(stream):
                yield {"application": path.parent.name, "source_file": path.relative_to(root).as_posix(),
                    "source_record_id": f"{path.name}:{index}", "ordinal_in_file": index,
                    "query": remove_record_terminator(line), "label": 0 if family == "normal" else 1,
                    "attack_family": family}


def audit_public(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with Path("data/processed/dataset.csv").open(encoding="utf-8-sig", newline="") as stream:
        legacy = list(csv.DictReader(stream))
    legacy_exact = {sha(row["query"]) for row in legacy}
    legacy_strip = {sha(row["query"].strip()) for row in legacy}
    legacy_templates = {sha(lexical_template(row["query"])) for row in legacy}
    exact, templates = {}, {}
    per_file, per_app, overlaps, unit_hints = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    total = 0
    for record in source_rows():
        total += 1
        query, label, app = record["query"], record["label"], record["application"]
        diagnostic = diagnose(query)
        key, template = diagnostic["exact_sha256"], diagnostic["template_sha256"]
        for mapping, fingerprint in ((exact, key), (templates, template)):
            if fingerprint not in mapping:
                mapping[fingerprint] = {"rows": 0, "labels": set(), "applications": set(), "examples": []}
            group = mapping[fingerprint]
            group["rows"] += 1
            group["labels"].add(label)
            group["applications"].add(app)
            if len(group["examples"]) < 3:
                group["examples"].append([app, record["source_record_id"]])
        per_file[record["source_file"]]["rows"] += 1
        per_file[record["source_file"]]["blank"] += int(not query.strip())
        per_file[record["source_file"]]["nul"] += int("\x00" in query)
        per_app[app]["benign" if label == 0 else "attack"] += 1
        unit_hints[diagnostic["unit_hint"]] += 1
        overlaps[app]["exact_with_legacy"] += int(key in legacy_exact)
        overlaps[app]["strip_with_legacy"] += int(diagnostic["strip_sha256"] in legacy_strip)
        overlaps[app]["template_with_legacy"] += int(template in legacy_templates)
        if total % 50000 == 0:
            print(f"Audited {total} public statements", flush=True)
    conflicts = {key for key, group in exact.items() if len(group["labels"]) > 1}
    counts, kept, exclusions = defaultdict(Counter), set(), defaultdict(Counter)
    for record in source_rows():
        key, app, label = sha(record["query"]), record["application"], record["label"]
        if not record["query"].strip():
            reason = "blank"
        elif "\x00" in record["query"]:
            reason = "nul"
        elif key in conflicts:
            reason = "conflicting_query_only_source_labels"
        elif (app, key) in kept:
            reason = "same_application_same_text_repeat"
        else:
            kept.add((app, key))
            counts[app]["benign" if label == 0 else "attack"] += 1
            continue
        exclusions[app][reason] += 1
    result = {"source_commit": COMMIT,
        "download_manifest_sha256": hashlib.sha256((DESTINATION / "download_manifest.json").read_bytes()).hexdigest(),
        "rows": total, "by_application": dict(per_app),
        "by_file": dict(per_file), "unit_hints": dict(unit_hints),
        "exact_unique_texts": len(exact), "exact_duplicate_excess": total - len(exact),
        "exact_mixed_label_groups": len(conflicts),
        "exact_mixed_label_rows": sum(exact[key]["rows"] for key in conflicts),
        "exact_mixed_label_examples": [{"sha256": key, "rows": exact[key]["rows"], "references": exact[key]["examples"]}
                                       for key in sorted(conflicts)],
        "lexical_templates": len(templates),
        "mixed_label_templates": sum(len(group["labels"]) > 1 for group in templates.values()),
        "overlap_with_legacy": dict(overlaps), "candidate_unique_reference": dict(counts),
        "candidate_exclusions": dict(exclusions),
        "selection_policy": "Keep one exact SQL per application; exclude all query-only exact label conflicts and blank/NUL rows. No filtering by detector score, parse success, or attack family.",
        "context": {"role": "unknown", "permissions": "unknown", "application_operation": "unknown", "policy": "unknown",
                    "dialect": "unknown; MySQL-compatible syntax is a source characteristic, not inferred trusted role context"},
        "evaluation_role": "known development/transfer source; not a new blind holdout",
        "label_basis": "author statement-level release and source-file category; attempts, not observed exploitation success",
        "synthetic_origin": "laboratory generation from application templates and injection payload sources",
        "parser_check": "not run; lexical diagnostics only"}
    (output / "sqlstateguard_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)
                                                  + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = audit_public(Path("results/stage4"))
    print(json.dumps({key: result[key] for key in ("rows", "by_application", "exact_unique_texts",
          "exact_mixed_label_groups", "candidate_unique_reference", "candidate_exclusions", "overlap_with_legacy")}, indent=2))
