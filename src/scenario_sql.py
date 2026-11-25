"""Closed SQL vocabulary for the controlled scenarios; static text operations only.

This recognizer covers the renderers in generate_scenarios, not general SQL.
Unknown syntax is rejected rather than assigned a class by a keyword heuristic.
"""

import re

from data_audit import TOKEN

IDENT = r"[a-z][a-z0-9_]*"
QN = rf"{IDENT}\.{IDENT}"
ACCOUNT = rf"(?:{IDENT}|'[a-z][a-z0-9_]*'@'(?:localhost|%)')"
KEYWORDS = set("grant select update on table object to with option role alter add member call exec execute from where order by join and exists as sum avg min max group begin end integer cast unsigned int chr char concat to_number".split())


def canonical_sql(text: str) -> str:
    pieces = []
    for match in TOKEN.finditer(text):
        kind, value = match.lastgroup, match.group()
        if kind in ("space", "comment"):
            continue
        if kind == "executable_comment":
            raise ValueError("Executable comments need their own reviewed scenario")
        pieces.append(value.lower() if kind == "word" else value)
    value = " ".join(pieces)
    value = re.sub(r"\s*([.,();@])\s*", r"\1", value)
    value = re.sub(r"\s*(\|\||\+|=)\s*", r"\1", value)
    return re.sub(r"\s*::\s*", "::", value).strip()


def account_name(text: str) -> str:
    if text.startswith("'"):
        first, host = text.split("@")
        return first.strip("'") + "@" + host.strip("'")
    return text


def encoded_integer(number: int, dialect: str) -> str:
    codes = [ord(x) for x in str(number)]
    if number < 0:
        raise ValueError("Only nonnegative scenario integers are supported")
    if dialect == "mysql":
        return "CAST(CONCAT(" + ", ".join(f"CHAR({n})" for n in codes) + ") AS UNSIGNED)"
    if dialect == "mssql":
        return "CAST(" + " + ".join(f"CHAR({n})" for n in codes) + " AS int)"
    expression = " || ".join(f"CHR({n})" for n in codes)
    return f"TO_NUMBER({expression})" if dialect == "oracle" else f"CAST({expression} AS integer)"


def _decode_integers(sql: str, dialect: str) -> str:
    # Recognition is deliberately limited to the documented constructor above.
    patterns = {
        "mysql": r"cast\(concat\((?:char\(\d+\),?)+\)as unsigned\)",
        "mssql": r"cast\(char\(\d+\)(?:\+char\(\d+\))*as int\)",
        "postgresql": r"cast\(chr\(\d+\)(?:\|\|chr\(\d+\))*as integer\)",
        "oracle": r"to_number\(chr\(\d+\)(?:\|\|chr\(\d+\))*\)",
    }
    def replace(match):
        codes = [int(x) for x in re.findall(r"(?:chr|char)\((\d+)\)", match.group())]
        if not codes or any(x < 48 or x > 57 for x in codes):
            raise ValueError("Encoded constant is not an ASCII decimal integer")
        number = int("".join(chr(x) for x in codes))
        if canonical_sql(encoded_integer(number, dialect)) != match.group():
            raise ValueError("Unrecognized scalar constructor")
        return str(number) + " "
    return canonical_sql(re.sub(patterns[dialect], replace, sql))


def _decode_string_expression(expression: str, dialect: str) -> str:
    matches = [m for m in TOKEN.finditer(expression) if m.lastgroup != "space"]
    operator = "+" if dialect == "mssql" else "||"
    values = []
    for index, match in enumerate(matches):
        if index % 2:
            if match.group() != operator:
                raise ValueError("Unexpected dynamic string operator")
        else:
            literal = match.group()
            if match.lastgroup != "string" or literal[:1].lower() not in ("'", "n"):
                raise ValueError("Only fixed SQL string fragments are supported")
            if literal[:1].lower() == "n":
                if dialect != "mssql":
                    raise ValueError("Unexpected national string prefix")
                literal = literal[1:]
            if "\\" in literal:
                raise ValueError("Backslash string modes are outside this vocabulary")
            values.append(literal[1:-1].replace("''", "'"))
    if not matches or len(matches) % 2 == 0:
        raise ValueError("Incomplete dynamic string expression")
    return "".join(values)


def restore_static_sql(text: str, dialect: str) -> str:
    wrappers = {
        "postgresql": r"\s*DO \$nirs\$ BEGIN EXECUTE (.*); END \$nirs\$;\s*",
        "oracle": r"\s*BEGIN EXECUTE IMMEDIATE (.*); END;\s*",
        "mssql": r"\s*EXEC\((.*)\);\s*",
    }
    if dialect in wrappers:
        match = re.fullmatch(wrappers[dialect], text, re.I | re.S)
        if match:
            text = _decode_string_expression(match.group(1), dialect) + ";"
    return _decode_integers(canonical_sql(text), dialect)


def _comments(text: str, style: int) -> str:
    output = []
    for match in TOKEN.finditer(text):
        kind, value = match.lastgroup, match.group()
        if kind == "space":
            output.append("/**/" if style % 2 else "/* q */\n")
        elif kind == "word" and value.lower() in KEYWORDS:
            output.append("".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(value)))
        else:
            output.append(value)
    return "".join(output)


def obfuscate(text: str, dialect: str, block: str, style: int) -> tuple[str, list[str], str]:
    if block == "privilege" and dialect != "mysql" and style % 2 == 0:
        payload = _comments(text.rstrip().removesuffix(";"), style)
        split = min(2, len(payload))
        fragments = [payload[:split], payload[split:]]
        literals = ["'" + value.replace("'", "''") + "'" for value in fragments]
        if dialect == "mssql":
            query = "EXEC(" + " + ".join("N" + x for x in literals) + ");"
        else:
            expression = " || ".join(literals)
            query = (f"BEGIN EXECUTE IMMEDIATE {expression}; END;" if dialect == "oracle"
                     else f"DO $nirs$ BEGIN EXECUTE {expression}; END $nirs$;")
        return query, ["comments", "string_concatenation"], "fixed_dynamic_string"
    pieces, encoded = [], False
    for match in TOKEN.finditer(text):
        if match.lastgroup == "number" and match.group().isdigit():
            pieces.append(encoded_integer(int(match.group()), dialect))
            encoded = True
        else:
            pieces.append(match.group())
    tags = ["comments"] + (["char_function", "string_concatenation"] if encoded else [])
    return _comments("".join(pieces), style), tags, "scalar_construction_and_comments" if encoded else "inter_token_comments"


def extract_action(text: str, dialect: str, scenario: dict) -> dict:
    sql = restore_static_sql(text, dialect)
    # Object grants, role membership and fixed procedure calls.
    principal = ACCOUNT if dialect == "mysql" else IDENT
    prefix = "table " if dialect == "postgresql" else "object::" if dialect == "mssql" else ""
    match = re.fullmatch(rf"grant (select|update|select,update) on {prefix}({QN}) to ({principal})( with grant option)?;", sql)
    if match:
        privilege, resource, recipient, option = match.groups()
        return {"action": "grant", "resources": [resource], "privileges": privilege.split(","),
                "recipient": account_name(recipient), "delegable": bool(option)}
    match = re.fullmatch(rf"grant ({principal}) to ({principal});", sql)
    if match and dialect != "mssql":
        role, recipient = map(account_name, match.groups())
        return {"action": "grant", "resources": ["role:" + role], "privileges": ["membership"],
                "recipient": recipient, "delegable": False}
    match = re.fullmatch(rf"alter role ({IDENT}) add member ({IDENT});", sql)
    if match and dialect == "mssql":
        role, recipient = match.groups()
        return {"action": "grant", "resources": ["role:" + role], "privileges": ["membership"],
                "recipient": recipient, "delegable": False}
    match = re.fullmatch(rf"exec ({QN}) (\d+);" if dialect == "mssql" else rf"call ({QN})\((\d+)\);", sql)
    if match:
        routine, arg = match.groups()
        request = int(arg)
        definition = scenario["routine_definitions"].get(routine)
        if not definition or request != scenario["request_id"]:
            raise ValueError("Unknown routine or request argument")
        rights = _routine_rights(routine, definition["ddl"], dialect)
        if canonical_sql(rights) != canonical_sql(definition["rights_statement"]):
            raise ValueError("Procedure source differs from its declared rights effect")
        effect = extract_action(rights, dialect, {"routine_definitions": {}})
        if effect["action"] != "grant":
            raise ValueError("Procedure has no declared rights-changing effect")
        return {"action": "execute", "resources": [routine], "request_id": request,
                "rights_effect": effect}
    return _extract_select(sql, scenario)


def _routine_rights(name: str, ddl: str, dialect: str) -> str:
    name = re.escape(name)
    patterns = {
        "postgresql": rf"CREATE PROCEDURE {name}\(request_id integer\) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS \$body\$ BEGIN (GRANT [^;]+;) END \$body\$;",
        "mysql": rf"CREATE PROCEDURE {name}\(IN request_id integer\) SQL SECURITY DEFINER BEGIN (GRANT [^;]+;) END;",
        "mssql": rf"CREATE PROCEDURE {name} @request_id int WITH EXECUTE AS OWNER AS BEGIN (GRANT [^;]+;) END;",
        "oracle": rf"CREATE OR REPLACE PROCEDURE {name}\(request_id IN integer\) AUTHID DEFINER AS BEGIN EXECUTE IMMEDIATE '(GRANT [^']+)'; END;",
    }
    match = re.fullmatch(patterns[dialect], ddl, re.I)
    if not match:
        raise ValueError("Procedure source is outside the reviewed rights-changing vocabulary")
    return match.group(1).rstrip(";") + ";"


def _extract_select(sql: str, scenario: dict) -> dict:
    resource = column = tenant = related = aggregate = None
    direct = re.fullmatch(rf"select t.entry_id,t.({IDENT}) from ({QN}) t where t.tenant_key=(\d+) order by t.entry_id;", sql)
    cte = re.fullmatch(rf"with scope_rows as\(select entry_id,tenant_key,({IDENT}) from ({QN}) where tenant_key=(\d+)\)select entry_id,\1 from scope_rows order by entry_id;", sql)
    derived = re.fullmatch(rf"select s.entry_id,s.({IDENT}) from\(select entry_id,tenant_key,\1 from ({QN}) where tenant_key=(\d+)\)s order by s.entry_id;", sql)
    if direct or cte or derived:
        column, resource, tenant = (direct or cte or derived).groups()
    else:
        joined = re.fullmatch(rf"select t.entry_id,u.({IDENT}) from ({QN}) t join ({QN}) u on t.entry_id=u.entry_id and t.tenant_key=u.tenant_key where t.tenant_key=(\d+) order by t.entry_id;", sql)
        exists = re.fullmatch(rf"select t.entry_id,t.({IDENT}) from ({QN}) t where t.tenant_key=(\d+) and exists\(select 1 from ({QN}) u where u.entry_id=t.entry_id and u.tenant_key=t.tenant_key\)order by t.entry_id;", sql)
        agg = re.fullmatch(rf"select t.tenant_key,(sum|avg|min|max)\(t.({IDENT})\)as total_value from ({QN}) t where t.tenant_key=(\d+) group by t.tenant_key;", sql)
        if joined:
            column, resource, related, tenant = joined.groups()
        elif exists:
            column, resource, tenant, related = exists.groups()
        elif agg:
            aggregate, column, resource, tenant = agg.groups()
        else:
            raise ValueError("SQL is outside the reviewed static vocabulary")
    known = scenario["tables"]
    if resource not in known or (related is not None and related not in known):
        raise ValueError("Unknown SQL resource")
    column_resource = related if related and " join " in sql else resource
    if column not in known[column_resource]["columns"]:
        raise ValueError("Unknown projected column")
    if aggregate and known[resource]["columns"][column] != "decimal(12,2)":
        raise ValueError("Aggregation requires a numeric fixture column")
    result = {"action": "select", "resources": sorted([resource] + ([related] if related else [])),
              "column": column, "tenant": int(tenant), "aggregate": aggregate, "related_resource": related}
    operation = scenario["operation"]
    if operation["workflow"] == "export_rows_v1":
        # The export route is a pre-decision application input, not a claim of a
        # measured file transfer. The declared handler serializes the SQL result.
        result.update(action="export", destination=operation["destination"])
    return result


def decide(effect: dict, policy: dict) -> int:
    if policy["coverage"] != "complete":
        raise ValueError("Scenario oracle requires explicitly scoped complete policy")
    import json
    decisions = set()
    for rule in policy["rules"]:
        if effect["action"] not in rule["actions"] or not set(effect["resources"]) <= set(rule["resources"]):
            continue
        constraints = json.loads(rule["condition"])
        if all(effect.get(key) == value for key, value in constraints.items()):
            decisions.add(rule["effect"])
    if len(decisions) != 1:
        raise ValueError("Policy decision is missing or contradictory")
    return int(decisions.pop() == "deny")
