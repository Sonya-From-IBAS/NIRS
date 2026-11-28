"""Read-only audit of frozen corpora before the five-class MySQL experiment."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import gzip
import io
import json
import re

from data_audit import TOKEN, sha
from ml_repair_cycle import DOMAINS, load as load_public, split as public_split
from split_protocol import UnionFind
from sql_views import KEYWORDS
from storage import ROOT, Storage, file_hash


OUT = ROOT / "results/multiclass_stage2"
STAGE6 = ROOT / "results/stage6"
CLASSES = ("legit", "sqli", "obfuscated", "privilege", "exfiltration")
SOURCES = {"nirs": "results/stage4/cleaned_reference.json",
           "sqlstateguard": "results/stage4/sqlstateguard_import.json",
           "scenarios": "results/stage5/import_report.json"}
# Number of manual review slots, fixed before inspecting any classifier output.
QUOTAS = {
    "legit": {"superviz26": 30, "sqlstateguard": 20, "nirs": 20, "scenarios": 30},
    "sqli": {"superviz26": 40, "sqlstateguard": 30, "nirs": 30},
    **{label: {"nirs": 80, "scenarios": 20} for label in CLASSES[2:]},
}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_table(name, rows, columns):
    path = OUT / name
    with path.open("wb") as raw:
        stream = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) if name.endswith(".gz") else raw
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)


def fingerprints(query):
    # Lexical fingerprint mirrors data_audit.lexical_template. Shape is a coarser
    # collision alarm, including renamed identifiers. Neither supplies a label.
    lexical, shape = [], []
    types = {"string": "<STR>", "number": "<NUM>", "hex": "<HEX>"}
    for match in TOKEN.finditer(query):
        kind, word = match.lastgroup, match.group()
        if kind in {"space", "comment"}:
            continue
        token = types.get(kind, word.upper() if kind == "word" else word)
        lexical.append(token)
        if kind == "quoted_identifier" or (kind == "word" and token not in KEYWORDS):
            token = "<ID>"
        shape.append(token)
    return sha(" ".join(lexical)), sha(" ".join(shape))


@dataclass(slots=True)
class Row:
    id: str
    corpus: str
    source: str
    label: str
    accepted: str
    basis: str
    dialect: str
    unit: str
    context: str
    family: str
    group: str
    scenario: str
    parents: tuple
    version: str
    record: str
    annotation: str
    exact: str
    lexical: str
    shape: str
    style: str
    campaign: str
    role: str


def load_history(connection):
    fit, opened = set(), set()
    base = ROOT / "results/runs/stage1-control"
    assert read(base / "status.json")["status"] == "complete"
    nirs_train = set()
    for path in sorted((base / "splits").glob("cv-*.csv")):
        with path.open(encoding="utf-8") as f:
            nirs_train.update(r["row_id"] for r in csv.DictReader(f) if r["part"] == "train")
    assert len(nirs_train) == 32530
    scopes = set(read("results/stage8/summary.json")["scopes"])
    scopes.update(p.stem for p in (ROOT / "results/stage10/selections").glob("*.json"))
    for name in sorted(scopes):
        for row in read_lines(STAGE6 / "memberships" / (name + ".jsonl.gz")):
            if "T" in row["roles"]:
                fit.add(row["id"])
            if "V" in row["roles"]:
                opened.add(row["id"])
    preview = read("results/preview/evaluation_plan.json")
    fit.update(preview["fit_ids"])
    fit.update(preview["diagnostic_train_ids"])
    opened.update(preview["diagnostic_validation_ids"])
    from preview_ml import balanced
    registry = read(STAGE6 / "split_registry.json")["protocols"]
    for name in ("transfer-web_app-to-wordpress", "transfer-wordpress-to-web_app"):
        entry = registry[name]
        pool = [r["id"] for r in read_lines(STAGE6 / "memberships" / (name + ".jsonl.gz")) if r["roles"] == "T"]
        ids = balanced(Storage(), entry["dataset_version_id"], pool, 1000, "preview-transfer-train:")
        fit.update(ids)
    llm = read(STAGE6 / "llm_development.json")
    for key in ("pilot", "few_shot_candidates"):
        opened.update(r["observation_id"] for r in llm[key])
    # Only identities, never predictions, scores, explanations or performance.
    opened.update(r["observation_id"] for r in connection.execute("SELECT DISTINCT observation_id FROM nirs.predictions"))
    opened.update(r["record_id"] for r in connection.execute(
        "SELECT DISTINCT record_id FROM nirs.mysql_predictions WHERE release_id=%s",
        (read("results/project/deployment.json")["release_id"],)))
    data = load_public()
    for domain in DOMAINS:
        result = read(f"results/ml_repair_cycle/confirm_LinearSVC_{domain}.json")
        train, cal, val = public_split(data, domain)
        assert [len(train), len(cal), len(val)] == [result[k]["rows"] for k in ("train", "calibration", "validation")]
        fit.update(f"sv26:{r['domain']}:{r['row']}" for r in train)
        opened.update(f"sv26:{r['domain']}:{r['row']}" for r in cal + val)
    # Earlier pilot sampled raw occurrence IDs. The current release keeps a
    # representative per (domain, exact SQL), which can have another row number.
    representatives = {(r["domain"], r["sql"]): f"sv26:{r['domain']}:{r['row']}"
                       for records in data.values() for r in records}
    for r in read(".local/superviz26/samples.json"):
        oid = representatives.get((r["domain"], r["sql"]))
        if oid:
            (fit if r["part"] == "train" else opened).add(oid)
    return nirs_train, fit, opened


def load_rows(c):
    versions = {key: read(path)["dataset_version_id"] for key, path in SOURCES.items()}
    versions["mysql"] = read("results/project/deployment.json")["release_id"]
    assert c.execute("SELECT release_id FROM nirs.mysql_active_release WHERE scope='sqli_mysql'").fetchone()["release_id"] == versions["mysql"]
    registry = read(STAGE6 / "split_registry.json")
    assert set(registry["datasets"].values()) == set(versions.values()) - {versions["mysql"]}
    groups = {r["id"]: r["group"] for r in read_lines(STAGE6 / "groups.jsonl.gz")}
    with (ROOT / "results/stage4/legacy_row_diagnostics.csv").open(encoding="utf-8") as f:
        diagnostics = {r["row_id"]: (r["unit_hint"], r["exact_sha256"]) for r in csv.DictReader(f)}
    rows = []
    sql = """SELECT d.observation_id id,d.record_id,d.annotation_id,d.source_version_id,
        o.sql_text,q.source_name,q.source_label,o.sql_sha256,o.unit_kind,o.grouping_json,
        o.context_json->'dialect'->>'value' dialect,
        ARRAY(SELECT key FROM jsonb_each(o.context_json) f
              WHERE f.value->>'availability'='known' AND f.value->>'value' IS NOT NULL
              AND (f.value->>'available_before_decision')::boolean ORDER BY key) known,
        a.target_label,a.status,a.data->'threat_tags' tags,a.data->'representation_tags' reprs
        FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
        JOIN nirs.occurrences q USING(source_version_id,record_id,observation_id)
        JOIN nirs.annotations a USING(annotation_id,observation_id)
        WHERE d.dataset_version_id=%s ORDER BY d.ordinal"""
    for corpus in SOURCES:
        with c.cursor(name="audit_" + corpus) as cursor:
            cursor.execute(sql, (versions[corpus],))
            for r in cursor:
                group = groups[r["id"]]
                g, src = r["grouping_json"], r["source_name"]
                parents = tuple(g["parent_observation_ids"])
                accepted = ""
                if corpus == "scenarios":
                    assert r["status"] == "verified" and set(r["known"]) == {
                        "dialect", "policy", "active_role", "permissions", "application_operation"}
                    mechanism = "privilege" if "privilege_escalation" in r["tags"] else "exfiltration"
                    if r["target_label"]:
                        assert len(r["tags"]) == 1 and r["tags"][0] in ("exfiltration", "privilege_escalation")
                    assert bool(parents) == bool(r["reprs"])
                    label = ("obfuscated" if parents else mechanism) if r["target_label"] else "legit"
                    accepted = label + ("|" + mechanism if label == "obfuscated" else "")
                    dialect, basis = r["dialect"], "authored_policy"
                elif corpus == "nirs":
                    label, basis = r["source_label"]["label_name"], "historical_label_only"
                    batch = re.search(r"_(mysql|postgresql|mssql|oracle)_", src)
                    dialect = batch[1] + "_batch" if batch else "unknown"
                else:
                    assert r["status"] == "verified" and r["target_label"] in (0, 1)
                    label = CLASSES[r["target_label"]]
                    accepted, dialect, basis = label, r["dialect"] or "unknown", "author_file"
                unit = r["unit_kind"]
                if corpus == "nirs":
                    unit, expected_hash = diagnostics[r["record_id"]]
                    assert expected_hash == r["sql_sha256"]
                query = r["sql_text"]
                lexical, shape = fingerprints(query)
                family = g.get("group_id") or (src + ":" + label if src.startswith("generated_") else group)
                rows.append(Row(r["id"], corpus, src, label, accepted, basis, dialect, unit,
                                ",".join(r["known"]) or "sql_only", family, group, g.get("scenario_id") or "",
                                parents, versions[corpus], r["record_id"], r["annotation_id"], r["sql_sha256"],
                                lexical, shape, style(query), g.get("campaign_id") or "", ""))
    assert len(groups) == len(rows)
    with c.cursor(name="audit_release") as cursor:
        cursor.execute("""SELECT r.record_id,r.source_id,r.domain,r.family,r.sql_text,r.sql_sha256,r.target,r.target_basis,
                         r.parameter_origin,r.parameters IS NOT NULL stored,r.data_role,r.metadata->>'campaign' campaign,
                         o.template_sql IS NOT NULL has_template FROM nirs.mysql_records r
                         LEFT JOIN nirs.mysql_operations o USING(release_id,operation_id)
                         WHERE r.release_id=%s ORDER BY r.record_id""", (versions["mysql"],))
        for r in cursor:
            corpus = "superviz26" if r["source_id"].startswith("superviz26:") else "controlled_sqli"
            ctx = ("template" if r["has_template"] else "no_template") + "/" + r["parameter_origin"]
            assert (r["parameter_origin"] == "provided") == r["stored"]
            query = r["sql_text"]
            lexical, shape = fingerprints(query)
            label = CLASSES[r["target"]]
            family = r["domain"] + ":" + (r["family"] or lexical)
            rows.append(Row(r["record_id"], corpus, r["domain"], label, label, r["target_basis"], "mysql",
                            "logged_sql" if corpus == "superviz26" else "authored_sql", ctx, family, "",
                            "", (), versions["mysql"], r["record_id"], r["target_basis"], r["sql_sha256"],
                            lexical, shape, style(query), r["campaign"] or "", r["data_role"]))
    return rows, versions


def style(query):
    return "/".join(("multiline" if "\n" in query else "one_line",
                     "terminal_delimiter" if query.rstrip().endswith(";") else "no_terminal_delimiter",
                     "with_backtick" if "`" in query else "no_backtick"))


def components(rows):
    dsu, seen = UnionFind(len(rows)), {}
    ids = {r.id: i for i, r in enumerate(rows)}
    assert len(ids) == len(rows)
    for i, r in enumerate(rows):
        keys = [("sql", r.exact), ("lexical", r.lexical)]
        if r.group:
            keys.append(("previous_component", r.group))
        else:
            keys.append(("operation_family", r.family))
        if r.campaign:
            keys.append(("campaign", r.source, r.campaign))
        for key in keys:
            dsu.union(i, seen.setdefault(key, i))
        for parent in r.parents:
            dsu.union(i, ids[parent])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[dsu.find(i)].append(r)
    for members in groups.values():
        gid = "m2_" + sha("\n".join(sorted(r.id for r in members)))
        for r in members:
            r.group = gid


def profile(rs):
    pairs = {(r.scenario, bool(r.parents)) for r in rs if r.scenario}
    return (len(rs), len({r.exact for r in rs}), len({r.family for r in rs}), len({r.group for r in rs}),
            len(pairs), sum(len(r.parents) for r in rs))


def coverage(rows):
    keys = ("corpus", "source", "primary_class", "dialect", "unit", "context")
    counts = ("rows", "unique_sql", "family_keys", "components", "scenario_pairs_represented", "parent_links")
    cells, families = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r.corpus, r.source, r.label, r.dialect, r.unit, r.context)
        cells[key].append(r)
        families[key + (r.family, r.group)].append(r)
    write_table("coverage.csv", (k + profile(rs) for k, rs in sorted(cells.items())), keys + counts)
    write_table("family_coverage.csv.gz", (k + profile(rs) for k, rs in sorted(families.items())),
                keys + ("family", "component") + counts)


def shortcuts(rows):
    result = {}
    for name, key in {
        "majority": lambda r: "all", "source": lambda r: r.source,
        "trusted_template_present": lambda r: str(r.context.startswith("template/")),
        "context": lambda r: r.context, "layout_only": lambda r: r.style,
        "origin": lambda r: ("local_authored" if r.source.startswith("generated_") or
                             r.corpus in ("scenarios", "controlled_sqli") else
                             "public_synthetic" if r.corpus == "superviz26" else
                             "application_experiment" if r.corpus == "sqlstateguard" else "public_legacy"),
    }.items():
        counts = defaultdict(Counter)
        for r in rows:
            counts[key(r)][r.label] += 1
        correct = Counter()
        for bucket in counts.values():
            winner = max(CLASSES, key=lambda label: bucket[label])
            correct[winner] += bucket[winner]
        totals = Counter(r.label for r in rows)
        result[name] = {"descriptive_accuracy": sum(correct.values()) / len(rows),
                        "recall_by_class": {k: correct[k] / n for k, n in totals.items()},
                        "cells": dict(sorted(counts.items()))}
    return result


def sample(rows, limit, salt):
    # Give different applications and components turns before taking a second
    # row from a populous family. Each exact SQL occurs once in this stratum.
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[r.source][r.group].append(r)
    def rank(key):
        return sha("multiclass-stage2-v1/" + salt + "/" + key)
    def per_source(groups):
        queues = [sorted(groups[g], key=lambda r: rank(r.id), reverse=True) for g in sorted(groups, key=rank)]
        while queues:
            for q in queues:
                yield q.pop()
            queues = [q for q in queues if q]
    streams = [iter(per_source(buckets[s])) for s in sorted(buckets, key=rank)]
    selected, seen = [], set()
    while streams and len(selected) < limit:
        active = []
        for stream in streams:
            row = next((r for r in stream if r.exact not in seen), None)
            if row is not None:
                seen.add(row.exact)
                selected.append(row)
                active.append(stream)
                if len(selected) == limit:
                    break
        streams = active
    return selected


def review_queue(rows, history):
    chosen = []
    paired = defaultdict(list)
    for r in rows:
        if r.corpus == "scenarios" and r.dialect == "mysql":
            paired[(r.scenario, bool(r.parents))].append(r)
    assert len(paired) == 150 and all(sorted(x.label == "legit" for x in pair) == [False, True] for pair in paired.values())
    for label, quotas in QUOTAS.items():
        for corpus, size in quotas.items():
            if corpus == "scenarios" and label == "legit":
                continue
            pool = [r for r in rows if r.corpus == corpus and r.label == label]
            if corpus == "scenarios":
                pool = [r for r in pool if r.dialect == "mysql"]
            picks = []
            if corpus == "nirs":
                pool = [r for r in pool if r.unit == "full_sql_candidate" and r.dialect in ("unknown", "mysql_batch")]
                if label in ("privilege", "exfiltration"):
                    picks = sample([r for r in pool if not r.source.startswith("generated_")],
                                   10, "original-source/" + label)
                hinted = [r for r in pool if r.dialect == "mysql_batch"]
                if len({r.exact for r in hinted}) >= size - len(picks):
                    pool = hinted
            if corpus in ("sqlstateguard", "superviz26"):
                # Prefer least exposed whole components under the same quota.
                # Shape overlap is a secondary warning even across distinct groups.
                def tier(r):
                    fit, opened, exact, group_fit, group_open, shape = history(r)
                    return (group_fit, group_open, exact, shape, fit, opened)
                tiers = defaultdict(list)
                for r in pool:
                    tiers[tier(r)].append(r)
                seen = set()
                for k in sorted(tiers):
                    extra = sample([r for r in tiers[k] if r.exact not in seen], size - len(picks), corpus + label)
                    picks.extend(extra)
                    seen.update(r.exact for r in extra)
                    if len(picks) == size:
                        break
            else:
                seen = {r.exact for r in picks}
                picks.extend(sample([r for r in pool if r.exact not in seen], size - len(picks), corpus + label))
            chosen.extend(picks)
            if corpus == "scenarios":
                for attack in picks[:10]:
                    chosen.append(next(r for r in paired[(attack.scenario, bool(attack.parents))] if r.label == "legit"))
    return sorted(chosen, key=lambda r: (CLASSES.index(r.label), r.corpus, r.id))


def run():
    with Storage().connect() as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        nirs_train, fit, opened = load_history(connection)
        rows, versions = load_rows(connection)
    for r in rows:
        if r.corpus == "nirs" and r.record in nirs_train:
            fit.add(r.id)
        if r.role == "train":
            fit.add(r.id)
        elif r.role and r.role != "unused_reservoir":
            opened.add(r.id)
    components(rows)
    fit_groups = {r.group for r in rows if r.id in fit}
    open_groups = {r.group for r in rows if r.id in opened}
    fit_shapes = {r.shape for r in rows if r.id in fit}
    fit_exact = {r.exact for r in rows if r.id in fit}
    def history(r):
        return (int(r.id in fit), int(r.id in opened), int(r.exact in fit_exact), int(r.group in fit_groups),
                int(r.group in open_groups), int(r.shape in fit_shapes))
    history_names = ("fit_scope", "opened_or_development", "exact_fit_overlap", "component_fit_overlap",
                     "component_opened_overlap", "coarse_shape_fit_overlap")
    OUT.mkdir(parents=True, exist_ok=True)
    coverage(rows)
    pools = defaultdict(list)
    for r in rows:
        pools[r.corpus].append(r)
    selected = review_queue(rows, history)
    assert len({r.id for r in selected}) == len(selected)
    requested = Counter({k: sum(q.values()) for k, q in QUOTAS.items()})
    achieved = Counter(r.label for r in selected)
    assert all(achieved[k] <= requested[k] for k in CLASSES)
    eligible = [r for r in rows if r.corpus == "superviz26" and r.dialect == "mysql" and r.accepted
                and not any(history(r))]
    leads = [r for r in pools["sqlstateguard"] if not any(history(r))]
    old_groups = {r["id"]:r["group"] for r in read_lines(STAGE6 / "groups.jsonl.gz") if r["cohort"] == "controlled"}
    bridges = defaultdict(set)
    for r in pools["scenarios"]:
        if r.dialect == "mysql":
            bridges[r.group].add(old_groups[r.id])
    # A negative ledger lookup alone does not prove independence.
    save("summary.json", {
        "version": "multiclass-stage2-v1", "versions": versions, "rows": len(rows),
        "unique_sql": len({r.exact for r in rows}), "linked_components": len({r.group for r in rows}),
        "coverage_by_corpus": {k: {"rows": len(rs), "unique_sql": len({r.exact for r in rs}),
            "components": len({r.group for r in rs}), "labels": Counter(r.label for r in rs),
            "history": dict(zip(history_names, map(sum, zip(*(history(r) for r in rs)))))}
            for k, rs in pools.items()},
        "metadata_shortcuts": shortcuts(rows),
        "mysql_scenario_component_crosswalk": {g: sorted(ids) for g, ids in sorted(bridges.items())},
        "mysql_policy_scenarios": dict(zip(("rows","unique_sql","families","components","paired_cases","parent_links"),
                                           profile([r for r in pools["scenarios"] if r.dialect == "mysql"]))),
        "manual_review_quota": QUOTAS, "manual_review_achieved": achieved,
        "queue_sources": {k: Counter(r.corpus for r in selected if r.label == k) for k in CLASSES},
        "queue_components_by_class": {k:len({r.group for r in selected if r.label == k}) for k in CLASSES},
        "queue_fit_scope": sum(r.id in fit for r in selected),
        "queue_group_fit_overlap": sum(r.group in fit_groups for r in selected),
        "strict_public_mysql_no_known_overlap_before_manual_review": len(eligible),
        "sqlstateguard_unknown_dialect_no_known_overlap": {
            k: {"rows": len(part), "components":len({r.group for r in part})}
            for k in ("legit", "sqli") for part in [[r for r in leads if r.label == k]]},
        "independently_certified_external_rows": 0,
        "source_selection_used_model_outputs": False})
    first_pass = set()
    for label in CLASSES:
        options = [r for r in selected if r.label == label]
        # Include every source where possible. Cover multiple groups of the same source.
        for r in sample(options, 5, "initial-reviewed/" + label):
            first_pass.add(r.id)
    cols = ("id","corpus","source","version","record","annotation_or_basis","primary_class","documented_labels",
            "basis","dialect","unit","predecision_context","family","component","scenario","parents","sql_sha256",
            "historical_release_role",*history_names,"initial_25","review_status","independent_second_read")
    def card(r):
        return (r.id,r.corpus,r.source,r.version,r.record,r.annotation,r.label,r.accepted,r.basis,r.dialect,
                r.unit,r.context,r.family,r.group,r.scenario,"|".join(r.parents),r.exact,r.role,*history(r),
                int(r.id in first_pass),"pending_human_review",int(r.corpus == "nirs" or r.dialect == "unknown" or "|" in r.accepted or
                                           int(sha("second-read/" + r.id)[:8],16) % 10 == 0))
    write_table("review_queue.csv", (card(r) for r in selected), cols)
    write_evidence_cards(selected, first_pass)
    save("checksums.json", {name: file_hash(OUT / name) for name in (
        "review_queue.csv","coverage.csv","family_coverage.csv.gz","summary.json","evidence_cards.json")})
    summary = read(OUT / "summary.json")
    print(json.dumps({key: summary[key] for key in ("rows","unique_sql","mysql_policy_scenarios",
                     "manual_review_achieved","queue_fit_scope","queue_group_fit_overlap",
                     "strict_public_mysql_no_known_overlap_before_manual_review")}, indent=2), flush=True)


def write_evidence_cards(selected, first_pass):
    """Trace the first 25 to source fields; keep independent reviews pending."""
    with Storage().connect() as c:
        c.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        result = []
        for r in selected:
            if r.id not in first_pass:
                continue
            card = {"id": r.id, "version": r.version, "class_for_quota": r.label,
                    "allowed_labels_on_current_evidence": r.accepted.split("|") if r.accepted else [],
                    "source": r.source, "record": r.record, "source_basis": r.basis,
                    "human_reviews": [], "disagreement": None}
            if r.corpus in ("superviz26", "controlled_sqli"):
                row = c.execute("""SELECT r.operation_id,r.parameters IS NOT NULL supplied,r.parameter_origin,
                    r.metadata->>'campaign' campaign,o.template_sql,o.provenance,r.target_basis
                    FROM nirs.mysql_records r LEFT JOIN nirs.mysql_operations o USING(release_id,operation_id)
                    WHERE r.release_id=%s AND r.record_id=%s""",(r.version,r.id)).fetchone()
                template = row.pop("template_sql")
                card["source_fields"] = {**row, "template_sha256": sha(template) if template else None,
                                          "role_permissions_policy": "unknown"}
                card["availability"] = "template from configuration; recovered values derived from incoming SQL; campaign is reference-only"
            else:
                row = c.execute("""SELECT q.source_version_id,q.source_label,d.annotation_id,a.status,
                    a.data->'reason' reason,a.data->'evidence' evidence,o.context_json
                    FROM nirs.dataset_members d JOIN nirs.occurrences q USING(source_version_id,record_id,observation_id)
                    JOIN nirs.annotations a USING(annotation_id,observation_id) JOIN nirs.observations o USING(observation_id)
                    WHERE d.dataset_version_id=%s AND d.observation_id=%s""",(r.version,r.id)).fetchone()
                context = row.pop("context_json")
                card["source_fields"] = row
                card["context"] = {k:{"origin":"scenario_author" if r.corpus=="scenarios" else v["availability"],
                                      "available_before_decision":v["available_before_decision"],
                                      "evidence_ref":v["evidence_ref"],
                                      "value": (v["value"].get("policy_id") if k=="policy" else
                                                v["value"].get("operation_id") if k=="application_operation" else
                                                v["value"].get("coverage") if k=="permissions" else v["value"])
                                      if isinstance(v["value"],dict) else v["value"]} for k,v in context.items()}
                card["template_and_separate_parameters"] = "unknown"
            result.append(card)
    save("evidence_cards.json", result)


if __name__ == "__main__":
    run()
