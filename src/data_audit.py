"""Deterministic, non-executing SQL diagnostics; hints are not ground truth."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

AUDIT_VERSION = "lexical-audit-1.0.0"
STARTERS = {
    "SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL", "COPY", "EXPLAIN",
    "SHOW", "DESCRIBE", "DESC", "USE", "SET", "BEGIN", "COMMIT", "ROLLBACK",
    "DECLARE", "RENAME", "REPLACE", "MERGE",
}
TOKEN = re.compile(
    r"(?P<executable_comment>/\*[!+].*?\*/)"
    r"|(?P<comment>/\*.*?\*/|--(?=\s|$)[^\r\n]*|\#[^\r\n]*)"
    r"|(?P<string>(?:[nNeE])?'(?:''|\\.|[^'\\])*')"
    r'|(?P<quoted_identifier>"(?:""|[^"])*"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\])'
    r"|(?P<hex>0[xX][0-9A-Fa-f]+)"
    r"|(?P<number>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9$]*)"
    r"|(?P<space>\s+)"
    r"|(?P<operator><=|>=|<>|!=|\|\||::|:=|.)",
    re.DOTALL,
)
SIGNALS = {
    "comments": re.compile(r"/\*|--(?=\s|$)|\#"),
    "hex_literal": re.compile(r"0x[0-9a-f]+", re.I),
    "char_function": re.compile(r"\b(?:CHAR|CHR)\s*\(", re.I),
    "url_encoding": re.compile(r"%[0-9a-f]{2}", re.I),
    "string_concatenation": re.compile(r"\|\||\bCONCAT\s*\(", re.I),
}
PRIVILEGE_SIGNALS = {
    "rights_change": re.compile(r"\b(?:GRANT|REVOKE)\b", re.I),
    "account_role_change": re.compile(r"\b(?:CREATE|ALTER|DROP)\s+(?:USER|ROLE)\b|sp_addsrvrolemember", re.I),
    "privileged_execution": re.compile(r"xp_cmdshell|COPY\b.*\bPROGRAM|DBMS_SCHEDULER|CREATE\s+EXTENSION", re.I | re.S),
    "system_catalog": re.compile(r"information_schema|mysql\.user|pg_shadow|\bsys\.|ALL_USERS|ALL_TABLES", re.I),
}
EXFIL_SIGNALS = {
    "explicit_file_operation": re.compile(r"INTO\s+(?:OUTFILE|DUMPFILE)|\bCOPY\b.*\bTO\b|lo_export|UTL_FILE", re.I | re.S),
    "external_access_function": re.compile(r"UTL_HTTP|UTL_INADDR|xp_dirtree|dblink|OPENROWSET", re.I),
    "file_read": re.compile(r"LOAD_FILE|pg_read_file|OPENROWSET", re.I),
    "read_query": re.compile(r"^\s*(?:SELECT|WITH)\b", re.I),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokens(query: str) -> list[tuple[str, str]]:
    return [(match.lastgroup, match.group()) for match in TOKEN.finditer(query)
            if match.lastgroup != "space"]


def lexical_template(query: str) -> str:
    """Grouping only: keep identifiers; mask values and ordinary comments.

    Shared templates may contain both labels. They do not establish equal SQL
    semantics or equal maliciousness, and are never used to assign targets.
    """
    result = []
    for kind, value in tokens(query):
        if kind == "comment":
            continue
        if kind in ("string", "number", "hex"):
            result.append({"string": "<STR>", "number": "<NUM>", "hex": "<HEX>"}[kind])
        elif kind == "word":
            result.append(value.upper())
        else:
            result.append(value)
    return " ".join(result)


def diagnose(query: str, label_name: str = "") -> dict:
    parsed = tokens(query)
    significant = [(kind, value) for kind, value in parsed if kind != "comment"]
    first = significant[0][1].upper() if significant else ""
    words = {value.upper() for kind, value in significant if kind == "word"}
    if not query.strip():
        hint = "blank"
    elif first in STARTERS:
        hint = "full_sql_candidate"
    elif (first in ("UNION", "OR", "AND", "HAVING", "WHERE")
          or words & {"SELECT", "UNION", "SLEEP", "WAITFOR", "BENCHMARK"}
          or re.search(r"\b(?:OR|AND)\b.*[=<>]", query, re.I | re.S)):
        hint = "fragment_candidate"
    elif first.startswith(("'", '"')) or query.count("'") % 2 or re.search(r"%27|0x[0-9a-f]{4,}", query, re.I):
        hint = "fragment_candidate"
    else:
        hint = "non_sql_candidate"
    signals = [name for name, pattern in SIGNALS.items() if pattern.search(query)]
    operations = []
    if label_name == "privilege":
        operations = [name for name, pattern in PRIVILEGE_SIGNALS.items() if pattern.search(query)]
    elif label_name == "exfiltration":
        operations = [name for name, pattern in EXFIL_SIGNALS.items() if pattern.search(query)]
    flags = []
    if hint != "full_sql_candidate":
        flags.append("unit_requires_separate_cohort")
    if label_name in ("privilege", "exfiltration"):
        flags.append("permission_or_operation_evidence_missing")
    return {"unit_hint": hint, "representation_signals": signals, "operation_signals": operations,
            "review_flags": flags, "odd_single_quote_count": bool(query.count("'") % 2),
            "exact_sha256": sha(query), "strip_sha256": sha(query.strip()),
            "template_sha256": sha(lexical_template(query))}


def audit_rows(rows: list[dict]) -> tuple[dict, list[dict]]:
    exact, stripped, templates = defaultdict(list), defaultdict(list), defaultdict(list)
    units, operations, source_labels, signals = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    details = []
    for index, row in enumerate(rows):
        query = row["query"]
        label = row["label_name"]
        diagnostic = diagnose(query, label)
        record = {"row_id": index, "source": row["source"], "source_label": row["label"],
                  "source_label_name": label, **diagnostic}
        details.append(record)
        exact[diagnostic["exact_sha256"]].append(index)
        stripped[diagnostic["strip_sha256"]].append(index)
        templates[diagnostic["template_sha256"]].append(index)
        units[label][diagnostic["unit_hint"]] += 1
        operations[label].update(diagnostic["operation_signals"])
        signals.update(diagnostic["representation_signals"])
        source_labels[(row["source"], label)] += 1

    def duplicates(groups):
        repeated = [indices for indices in groups.values() if len(indices) > 1]
        conflicts = [indices for indices in repeated if len({rows[index]["label"] for index in indices}) > 1]
        return {"groups": len(repeated), "excess_occurrences": sum(len(group) - 1 for group in repeated),
                "mixed_source_label_groups": len(conflicts), "mixed_label_row_ids": conflicts}

    frequencies = {}
    for label in ("legit", "privilege", "exfiltration"):
        subset = [row["query"] for row in rows if row["label_name"] == label]
        frequencies[label] = {name: sum(bool(re.search(r"\b" + re.escape(name) + r"\b", query, re.I)) for query in subset)
                              for name in ("users", "admin", "username", "password", "payroll", "orders")}
    summary = {"audit_version": AUDIT_VERSION, "rows": len(rows),
        "source_label_counts": dict(Counter(row["label_name"] for row in rows)),
        "unit_hints_by_source_label": dict(units), "representation_signal_counts": dict(signals),
        "operation_signals_by_source_label": dict(operations), "identifier_mentions": frequencies,
        "exact_duplicates": duplicates(exact), "strip_duplicates": duplicates(stripped),
        "lexical_templates": {"unique": len(templates), **duplicates(templates)},
        "by_source_and_label": [{"source": source, "source_label": label, "rows": count}
                                for (source, label), count in sorted(source_labels.items())],
        "parser_check": {"status": "not_run", "reason": "SQLGlot installation was not authorized; lexical diagnostics only"},
        "limitations": ["Unit hints are heuristic candidates, not parser or semantic validation.",
                        "Mixed labels for a masked template do not automatically imply a labeling error.",
                        "Source labels, synthetic origin and syntax signals do not automatically verify targets."]}
    return summary, details


def audit_csv(path: Path, output: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary, details = audit_rows(rows)
    summary["input_sha256_bytes"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "legacy_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)
                                           + "\n", encoding="utf-8", newline="\n")
    with (output / "legacy_row_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
        writer.writeheader()
        for record in details:
            writer.writerow({key: ";".join(value) if isinstance(value, list) else value for key, value in record.items()})
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_csv(arguments.path, arguments.output)
    print(json.dumps({key: result[key] for key in ("rows", "source_label_counts", "unit_hints_by_source_label",
                                                 "exact_duplicates", "strip_duplicates")}, ensure_ascii=False, indent=2))
