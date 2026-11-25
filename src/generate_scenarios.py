"""Reproducible, AI-authored controlled SQL scenarios and static policy checks.

SQL is text throughout this module. Database persistence has a separate CLI.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from data_contract import CONTEXT_FIELDS, detector_input, validate_observation
from data_audit import lexical_template
from scenario_sql import canonical_sql, decide, extract_action, obfuscate, restore_static_sql

ROOT = Path(__file__).resolve().parent.parent
VERSION = "stage5-static-v1"
DIALECTS = ("mysql", "postgresql", "mssql", "oracle")
TARGETS = {"mysql": "8.4", "postgresql": "17", "mssql": "SQL Server 2022", "oracle": "19c"}
FAMILIES = {
    "privilege": ("p01-object-scope", "p02-principal-scope", "p03-role-membership", "p04-grant-option", "p05-privileged-routine"),
    "exfiltration": ("e01-column-scope", "e02-row-scope", "e03-related-resources", "e04-disclosure-contract", "e05-export-destination"),
}
DOMAINS = (
    ("commerce", ("orders", "payments", "customers"), ("amount", "discount")),
    ("support", ("tickets", "messages", "assignments"), ("priority", "severity")),
    ("finance", ("ledger_entries", "reconciliations", "allocations"), ("amount", "fee")),
    ("research", ("samples", "measurements", "projects"), ("reading_a", "reading_b")),
    ("operations", ("events", "tasks", "devices"), ("metric_a", "metric_b")),
)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def account(name: str, dialect: str, *, role=False) -> str:
    return f"'{name}'@'{'%' if role else 'localhost'}'" if dialect == "mysql" else name


def normalized_account(name: str, dialect: str, *, role=False) -> str:
    return name + ("@%" if role else "@localhost") if dialect == "mysql" else name


def grant_sql(resource: str, recipient: str, privileges: list[str], dialect: str, delegable=False) -> str:
    prefix = "TABLE " if dialect == "postgresql" else "OBJECT::" if dialect == "mssql" else ""
    return (f"GRANT {', '.join(x.upper() for x in privileges)} ON {prefix}{resource} TO {account(recipient, dialect)}"
            + (" WITH GRANT OPTION" if delegable else "") + ";")


def select_sql(resource: str, column: str, tenant: int, style: int) -> str:
    if style % 3 == 1:
        return (f"WITH scope_rows AS (SELECT entry_id, tenant_key, {column} FROM {resource} WHERE tenant_key = {tenant}) "
                f"SELECT entry_id, {column} FROM scope_rows ORDER BY entry_id;")
    if style % 3 == 2:
        return (f"SELECT s.entry_id, s.{column} FROM (SELECT entry_id, tenant_key, {column} FROM {resource} "
                f"WHERE tenant_key = {tenant}) s ORDER BY s.entry_id;")
    return f"SELECT t.entry_id, t.{column} FROM {resource} t WHERE t.tenant_key = {tenant} ORDER BY t.entry_id;"


def routine_definition(name: str, statement: str, dialect: str) -> dict:
    if dialect == "postgresql":
        ddl = (f"CREATE PROCEDURE {name}(request_id integer) LANGUAGE plpgsql SECURITY DEFINER "
               f"SET search_path = pg_catalog AS $body$ BEGIN {statement} END $body$;")
    elif dialect == "mysql":
        ddl = f"CREATE PROCEDURE {name}(IN request_id integer) SQL SECURITY DEFINER BEGIN {statement} END;"
    elif dialect == "mssql":
        ddl = f"CREATE PROCEDURE {name} @request_id int WITH EXECUTE AS OWNER AS BEGIN {statement} END;"
    else:
        ddl = (f"CREATE OR REPLACE PROCEDURE {name}(request_id IN integer) AUTHID DEFINER AS "
               f"BEGIN EXECUTE IMMEDIATE '{statement.rstrip(';')}'; END;")
    return {"ddl": ddl, "rights_statement": statement, "security_context": "fixture_owner",
            "owner_contract": "Owner holds direct grant authority for the referenced fixture table.",
            "request_parameter": "integer application request identifier", "initial_beneficiary_rights": "no access to the granted object"}


def rule(effect: str, action: dict, resources: list[str]) -> dict:
    constraints = {key: value for key, value in action.items() if key not in ("action", "resources")}
    constraints["resources"] = action["resources"]
    return {"effect": effect, "actions": [action["action"]], "resources": resources, "condition": canonical(constraints)}


def build_scenario(block: str, family: str, dialect: str, index: int) -> tuple[dict, list[str]]:
    domain, names, columns = DOMAINS[index // 2]
    flip = index % 2
    schema = "lab_" + domain
    tables = [schema + "." + name for name in names]
    actor, beneficiaries = "ops_service", ("report_service", "batch_service")
    recipient = beneficiaries[flip]
    tenant = (index // 2 + 1) * 10 + 1
    request = 1000 + list(FAMILIES[block]).index(family) * 100 + index
    sid = "sc5_" + sha(f"{family}:{dialect}:{index}")[:24]
    table_defs = {}
    for position, name in enumerate(tables):
        table_defs[name] = {"columns": {"entry_id": "integer", "tenant_key": "integer", **{col: "decimal(12,2)" for col in columns}},
                           "rows": [[position * 100 + 1, tenant, "12.50", "8.25"],
                                    [position * 100 + 2, tenant + 1, "19.00", "4.50"]]}
    # For joined fixtures the identifiers overlap across tables, including both
    # tenant values; the join is not accidentally empty on the sample data.
    for definition in table_defs.values():
        definition["rows"][0][0] = 1
        definition["rows"][1][0] = 2
    scenario = {"scenario_id": sid, "family_id": family, "block": block, "dialect": dialect,
                "target_dialect_version": TARGETS[dialect], "domain": domain, "tables": table_defs,
                "actor": normalized_account(actor, dialect), "request_id": request,
                "principals": [normalized_account(x, dialect) for x in (actor, *beneficiaries)],
                "routine_definitions": {}, "role_definitions": {},
                "operation": {"workflow": "delegate_access_v1" if block == "privilege" else "read_rows_v1",
                              "request_id": request},
                "context_basis": "declared controlled scenario", "sql_mode": "standard quoted literals; fixed fixture schema resolution"}
    queries, actions = [], []
    if family in ("p01-object-scope", "p02-principal-scope", "p04-grant-option"):
        privileges = ["select"] if flip == 0 else ["update"]
        for choice in range(2):
            resource = tables[choice] if family != "p02-principal-scope" else tables[0]
            target = beneficiaries[choice] if family == "p02-principal-scope" else recipient
            delegable = family == "p04-grant-option"
            queries.append(grant_sql(resource, target, privileges, dialect, delegable))
            actions.append({"action": "grant", "resources": [resource], "privileges": privileges,
                            "recipient": normalized_account(target, dialect), "delegable": delegable})
    elif family == "p03-role-membership":
        for choice in range(2):
            role_name = domain + ("_review_team" if choice == 0 else "_delivery_team")
            role_key = normalized_account(role_name, dialect, role=True)
            # Both roles convey real fixture rights; membership is not equated
            # with escalation solely because of a role's name.
            scenario["role_definitions"][role_key] = {"privileges": ["select"], "resource": tables[choice]}
            queries.append(f"ALTER ROLE {role_name} ADD MEMBER {recipient};" if dialect == "mssql" else
                           f"GRANT {account(role_name, dialect, role=True)} TO {account(recipient, dialect)};")
            actions.append({"action": "grant", "resources": ["role:" + role_key], "privileges": ["membership"],
                            "recipient": normalized_account(recipient, dialect), "delegable": False})
    elif family == "p05-privileged-routine":
        for choice in range(2):
            routine = schema + (".assign_review" if choice == 0 else ".assign_delivery")
            rights = grant_sql(tables[choice], recipient, ["select"], dialect)
            scenario["routine_definitions"][routine] = routine_definition(routine, rights, dialect)
            queries.append(f"EXEC {routine} {request};" if dialect == "mssql" else f"CALL {routine}({request});")
            actions.append({"action": "execute", "resources": [routine], "request_id": request,
                            "rights_effect": {"action": "grant", "resources": [tables[choice]], "privileges": ["select"],
                                              "recipient": normalized_account(recipient, dialect), "delegable": False}})
    else:
        for choice in range(2):
            column, row_tenant = columns[choice] if family in ("e01-column-scope", "e04-disclosure-contract") else columns[0], tenant + flip
            related = aggregate = None
            if family == "e02-row-scope":
                row_tenant = tenant + choice
            if family == "e03-related-resources":
                related = tables[choice + 1]
                if index % 2:
                    query = (f"SELECT t.entry_id, t.{column} FROM {tables[0]} t WHERE t.tenant_key = {row_tenant} "
                             f"AND EXISTS (SELECT 1 FROM {related} u WHERE u.entry_id = t.entry_id "
                             "AND u.tenant_key = t.tenant_key) ORDER BY t.entry_id;")
                else:
                    query = (f"SELECT t.entry_id, u.{column} FROM {tables[0]} t JOIN {related} u "
                             "ON t.entry_id = u.entry_id AND t.tenant_key = u.tenant_key "
                             f"WHERE t.tenant_key = {row_tenant} ORDER BY t.entry_id;")
            elif family == "e04-disclosure-contract":
                aggregate = ("sum", "avg", "min", "max")[index % 4]
                query = (f"SELECT t.tenant_key, {aggregate.upper()}(t.{column}) AS total_value FROM {tables[0]} t "
                         f"WHERE t.tenant_key = {row_tenant} GROUP BY t.tenant_key;")
            else:
                query = select_sql(tables[0], column, row_tenant, index)
            queries.append(query)
            effect = {"action": "select", "resources": sorted([tables[0]] + ([related] if related else [])),
                      "column": column, "tenant": row_tenant, "aggregate": aggregate, "related_resource": related}
            if family == "e05-export-destination":
                destination = f"/app/{domain}/reports/{'periodic' if choice == 0 else 'reconciliation'}.csv"
                effect.update(action="export", destination=destination)
            actions.append(effect)
    scope = sorted({resource for action in actions for resource in action["resources"]})
    scenario["policy"] = {"coverage": "complete", "scope_resources": scope,
        "rules": [rule("allow", actions[flip], scope), rule("deny", actions[1 - flip], scope)],
        "policy_id": "scenario-policy:" + sid, "version": VERSION, "sensitive_resources": scope if block == "exfiltration" else []}
    scenario["permissions"] = {"coverage": "complete", "scope_resources": scope,
        "rules": [{"effect": "allow", "actions": sorted({a["action"] for a in actions}), "resources": scope,
                   "condition": canonical({"technical_scope": "both declared candidate actions are technically available; the operation policy is narrower",
                                           "grant_authority": "fixture-owned objects and role administration" if block == "privilege" else None})}]}
    scenario["candidate_operations"] = actions
    if family == "e05-export-destination":
        scenario["application_handler"] = {"handler_id": "export_rows_v1", "contract": "serialize the selected rows as CSV to the destination in the operation request",
                                           "data_origin": "fixture SQL result", "destination_kind": "application_local_file"}
    return scenario, queries


def case_scenario(scenario: dict, slot: int) -> dict:
    current = deepcopy(scenario)
    if scenario["family_id"] == "e05-export-destination":
        current["operation"].update(workflow="export_rows_v1", destination=scenario["candidate_operations"][slot]["destination"],
                                    destination_kind="application_local_file")
    return current


def make_record(scenario: dict, slot: int, query: str, source_hash: str, *, parent=None, tags=None) -> tuple[dict, dict]:
    dialect = scenario["dialect"]
    sid = scenario["scenario_id"]
    current = case_scenario(scenario, slot)
    effect = extract_action(query, dialect, current)
    target = decide(effect, current["policy"])
    oid = "obs_s5_" + sha(f"{VERSION}:{sid}:{slot}:{'variant' if parent else 'base'}")[:32]
    block = "obfuscated" if parent else scenario["block"]
    kind = "privilege_escalation" if scenario["block"] == "privilege" else "exfiltration"
    channel = "file_export" if effect["action"] == "export" else "unauthorized_read"
    reference = f"scenario-catalog:{source_hash}#{sid}"
    values = {"dialect": dialect, "active_role": current["actor"], "permissions": current["permissions"],
              "application_operation": {"operation_id": current["domain"] + "." + current["operation"]["workflow"],
                                        "description": canonical(current["operation"])}, "policy": current["policy"]}
    context = {key: {"availability": "known", "value": values[key], "evidence_ref": reference + "/context/" + key,
                     "available_before_decision": True} for key in CONTEXT_FIELDS}
    annotation = {"annotation_id": oid + "_a1", "revision": 1, "supersedes": None, "instruction_version": "1.0.0",
        "status": "verified", "target_label": target, "threat_tags": [kind] if target else [],
        "representation_tags": sorted(tags or []), "exfiltration_channels": [channel] if target and kind == "exfiltration" else [],
        "reason": ("The parsed action matches an explicit deny rule in the declared operation policy." if target else
                   "The parsed action matches an explicit allow rule in the declared operation policy."),
        "evidence": [{"kind": "scenario_definition", "reference": reference,
                      "statement": "Versioned controlled schema, role, rights and complete operation policy; AI-authored scenario with static review."},
                     {"kind": "policy_check", "reference": f"static-checks:{VERSION}#{oid}",
                      "statement": "Closed-vocabulary SQL action extraction and exact scoped rule matching; reviewer=" + VERSION}],
        "reviewer": {"kind": "rule", "reviewer_id": VERSION}}
    record = {"schema_version": "1.0.0", "observation_id": oid, "sql": {"text": query, "unit_kind": "full_sql"},
        "context": context, "source": {"source_id": "controlled-stage5", "source_version": VERSION,
            "source_record_id": oid, "source_file_sha256": source_hash, "origin": "llm_generated", "purpose": "research",
            "source_label": None, "generator": {"provider": "assistant_session", "model": None,
                                                  "prompt_version": "stage5-controlled-renderers-v1"}},
        "annotations": [annotation], "current_annotation_id": annotation["annotation_id"],
        "grouping": {"group_id": "s5-family:" + scenario["family_id"], "scenario_id": sid,
                     "template_id": "s5-template:" + scenario["family_id"], "campaign_id": None,
                     "parent_observation_ids": [parent["observation_id"]] if parent else []},
        "execution": {"outcome": "not_executed", "evidence_ref": None}}
    detail = {"observation_id": oid, "scenario_id": sid, "family_id": scenario["family_id"], "dialect": dialect,
              "block": block, "case_slot": slot, "parsed_action": effect, "policy_result": target,
              "parent_observation_id": parent["observation_id"] if parent else None,
              "verification_method": "static_scenario_and_policy", "parser_scope": "reviewed_closed_vocabulary"}
    if parent:
        if (restore_static_sql(query, dialect) != canonical_sql(parent["sql"]["text"])
                or parent["annotations"][-1]["target_label"] != target or parent["context"] != context):
            raise ValueError("Variant changed its parent's normalized operation, label or context")
        detail["parent_operation_restored"] = True
    validate_observation(record)
    return record, detail


def build_corpus() -> tuple[list[dict], list[dict], list[dict]]:
    drafts = []
    for block, families in FAMILIES.items():
        for family in families:
            for dialect in DIALECTS:
                for index in range(10):
                    scenario, queries = build_scenario(block, family, dialect, index)
                    drafts.append((scenario, queries, index))
    catalog = [scenario for scenario, _, _ in drafts]
    source_hash = sha(canonical(catalog))
    records, details = [], []
    for scenario, queries, index in drafts:
        for slot, query in enumerate(queries):
            record, detail = make_record(scenario, slot, query, source_hash)
            records.append(record)
            details.append(detail)
            if index < 5:
                transformed, tags, method = obfuscate(query, scenario["dialect"], scenario["block"], index)
                variant, checked = make_record(scenario, slot, transformed, source_hash, parent=record, tags=tags)
                checked["transformation_method"] = method
                records.append(variant)
                details.append(checked)
    return catalog, records, details


def family_links(records: list[dict]) -> list[dict]:
    """Join known family/parent and shared structural-template links for splits."""
    roots = {row["grouping"]["group_id"]: row["grouping"]["group_id"] for row in records}
    def find(key):
        while roots[key] != key:
            roots[key] = roots[roots[key]]
            key = roots[key]
        return key
    templates = {}
    for row in records:
        family = row["grouping"]["group_id"]
        key = lexical_template(restore_static_sql(row["sql"]["text"], row["context"]["dialect"]["value"]))
        if key in templates:
            first, second = find(family), find(templates[key])
            roots[max(first, second)] = min(first, second)
        else:
            templates[key] = family
    components = defaultdict(list)
    for row in records:
        components[find(row["grouping"]["group_id"])].append(row)
    result = []
    for _, members in sorted(components.items()):
        families = sorted({row["grouping"]["group_id"] for row in members})
        result.append({"group_id": "s5-component:" + sha(canonical(families))[:24], "family_group_ids": families,
                       "observation_ids": sorted(row["observation_id"] for row in members),
                       "basis": "family links and shared lexical templates after restoring parent operations"})
    return result


def audit_corpus(catalog: list[dict], records: list[dict], details: list[dict]) -> dict:
    if len(records) != 1200 or len({r["observation_id"] for r in records}) != 1200:
        raise ValueError("Corpus needs exactly 1200 unique observation IDs")
    by_scenario = {row["scenario_id"]: row for row in catalog}
    by_id = {row["observation_id"]: row for row in records}
    by_detail = {row["observation_id"]: row for row in details}
    counts, dialects, tags, methods, channels = defaultdict(Counter), defaultdict(Counter), Counter(), Counter(), Counter()
    pair_labels, input_labels, sql_labels = defaultdict(list), defaultdict(set), defaultdict(set)
    sql_target_counts, sql_members = defaultdict(Counter), defaultdict(list)
    object_labels, role_labels = defaultdict(set), defaultdict(set)
    restored = 0
    for record in records:
        validate_observation(record)
        annotation = record["annotations"][-1]
        oid, target = record["observation_id"], annotation["target_label"]
        detail = by_detail[oid]
        scenario = by_scenario[record["grouping"]["scenario_id"]]
        current = case_scenario(scenario, detail["case_slot"])
        effect = extract_action(record["sql"]["text"], scenario["dialect"], current)
        if decide(effect, current["policy"]) != target or effect != detail["parsed_action"]:
            raise ValueError("Stored annotation/check differs from actual SQL policy decision")
        block, dialect = detail["block"], detail["dialect"]
        counts[block][str(target)] += 1
        dialects[block + ":" + dialect][str(target)] += 1
        pair_labels[(detail["scenario_id"], block)].append(target)
        projection = detector_input(record, include_context=True)
        input_labels[sha(canonical(projection))].add(target)
        sql_hash = sha(record["sql"]["text"])
        sql_labels[sql_hash].add(target)
        sql_target_counts[sql_hash][target] += 1
        sql_members[sql_hash].append({"observation_id": oid, "family_id": detail["family_id"], "dialect": dialect})
        tags.update(annotation["representation_tags"])
        channels.update(annotation["exfiltration_channels"])
        for resource in effect["resources"]:
            object_labels[(dialect, resource)].add(target)
        role_labels[(dialect, scenario["actor"])].add(target)
        parents = record["grouping"]["parent_observation_ids"]
        if parents:
            if len(parents) != 1 or parents[0] not in by_id:
                raise ValueError("Missing variant parent")
            parent = by_id[parents[0]]
            if (parent["context"] != record["context"] or parent["annotations"][-1]["target_label"] != target
                    or parent["grouping"]["group_id"] != record["grouping"]["group_id"]
                    or restore_static_sql(record["sql"]["text"], dialect) != canonical_sql(parent["sql"]["text"])):
                raise ValueError("Broken variant invariant")
            restored += 1
            methods[detail["transformation_method"]] += 1
    if any(value != [0, 1] and value != [1, 0] for value in pair_labels.values()):
        raise ValueError("Every complete scenario pair must have one target of each class")
    if any(dict(counts[block]) != {"0": 200, "1": 200} for block in ("privilege", "exfiltration", "obfuscated")):
        raise ValueError("Block balance differs from agreed quotas")
    if any(dict(value) != {"0": 50, "1": 50} for value in dialects.values()) or restored != 400:
        raise ValueError("Dialect balance or parent count differs")
    if any(len(labels) > 1 for labels in input_labels.values()):
        raise ValueError("Same SQL and detector context has contradictory labels")
    if any(labels != {0, 1} for labels in object_labels.values()) or any(labels != {0, 1} for labels in role_labels.values()):
        raise ValueError("Object or actor occurs in only one class")
    return {"status": "passed", "verification_method": "static_scenario_and_policy", "records": len(records),
        "scenario_pairs": len(catalog), "base_rows": 800, "parent_linked_variants": restored,
        "by_block_and_target": dict(counts), "by_block_dialect_and_target": dict(dialects),
        "representation_tag_counts": dict(tags), "transformation_method_counts": dict(methods),
        "positive_exfiltration_channels": dict(channels),
        "source_families": len({r["family_id"] for r in catalog}),
        "family_group_ids": len({r["grouping"]["group_id"] for r in records}),
        "conservative_split_groups": len(family_links(records)),
        "exact_unique_SQL_texts": len(sql_labels), "exact_SQL_with_both_targets": sum(len(x) > 1 for x in sql_labels.values()),
        "unique_SQL_plus_detector_context": len(input_labels),
        "SQL_plus_context_label_conflicts": 0, "object_and_actor_class_counterbalance": "passed",
        "context_dependent_rows": sum(sum(sql_target_counts[key].values()) for key, labels in sql_labels.items() if len(labels) > 1),
        "minimum_SQL_only_errors_from_exact_conflicts": sum(min(value[0], value[1]) for value in sql_target_counts.values()),
        "context_dependent_family_counts": dict(Counter(member["family_id"] for key, labels in sql_labels.items() if len(labels) > 1 for member in sql_members[key])),
        "same_SQL_different_context_role": "context-dependent cases are evaluated as a separate slice; SQL-only cannot distinguish identical SQL texts",
        "authoring": "AI-authored scenario/SQL templates; deterministic expansion and static rule checks",
        "external_LLM_API_calls": 0}


def main() -> None:
    catalog, records, details = build_corpus()
    report = audit_corpus(catalog, records, details)
    output = ROOT / "data/scenarios"
    # Canonical catalog bytes are the immutable source referenced by observations.
    (output / "catalog_v1.json").write_text(canonical(catalog), encoding="utf-8", newline="\n")
    for filename, rows in (("corpus_v1.jsonl", records), ("static_checks_v1.jsonl", details)):
        with (output / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    save_json(ROOT / "results/stage5/generation_audit.json", report)
    save_json(ROOT / "results/stage5/group_links.json", family_links(records))
    mixed = defaultdict(set)
    for row in records:
        mixed[sha(row["sql"]["text"])].add(row["annotations"][-1]["target_label"])
    save_json(ROOT / "results/stage5/context_dependent_slice.json",
              {"selection": "same exact SQL text with both targets and different known contexts",
               "observation_ids": [row["observation_id"] for row in records if len(mixed[sha(row["sql"]["text"])] ) > 1]})
    previews = []
    for block in ("privilege", "exfiltration", "obfuscated"):
        for dialect in DIALECTS:
            sid = next(row["scenario_id"] for row in details if row["block"] == block and row["dialect"] == dialect)
            chosen = {row["observation_id"] for row in details if row["block"] == block and row["dialect"] == dialect and row["scenario_id"] == sid}
            previews.extend(row for row in records if row["observation_id"] in chosen)
    save_json(ROOT / "results/stage5/examples.json", {"records": previews, "role": "24 examples selected from the accepted corpus"})
    print(json.dumps({key: report[key] for key in ("status", "records", "by_block_and_target", "parent_linked_variants")}, indent=2))


if __name__ == "__main__":
    main()
