"""Versioned text heuristics and a deliberately limited predecision policy baseline."""

import json
import re

import numpy as np

from baseline import RegexDetector
from sql_features import prepare_input
from sql_views import lex, string_value

VERSION = 'stage8-rules-v1'
IDENT = r'[a-z_][a-z0-9_$]*'
RESOURCE = rf'{IDENT}(?:\.{IDENT})?'
PRINCIPAL = rf"(?:{IDENT}|'[^']+'@'[^']+')"


def lexical_text(sql):
    """Normalize inter-token spacing; preserve literals, reject dynamic bodies."""
    tokens = lex(sql)
    if any(token.kind in ('exec_comment', 'dollar') for token in tokens):
        return None
    tokens = [token for token in tokens if token.kind not in ('space', 'block_comment', 'line_comment')]
    text = ' '.join(token.text.lower() if token.kind == 'word' else token.text for token in tokens)
    text = re.sub(r'\s*::\s*', '::', text)
    return re.sub(r'\s*([.,();@=])\s*', r'\1', text).strip().rstrip(';')


class RegexV2:
    """Comments/encoding alone no longer constitute an attack decision."""
    def predict(self, inputs):
        decisions = []
        for item in inputs:
            sql = prepare_input(item).projected['sql']
            # Literal contents cannot trigger a keyword rule by themselves.
            tokens = [token for token in lex(sql) if token.kind not in ('space', 'block_comment', 'line_comment')]
            text = ' '.join(token.text.upper() if token.kind == 'word' else
                            'STRING' if token.kind in ('string', 'dollar') else token.text for token in tokens)
            attack = bool(re.search(r'\b(?:GRANT|REVOKE)\b|\b(?:CREATE|ALTER)\s+(?:USER|ROLE)\b'
                                    r'|\b(?:XP_CMDSHELL|XP_DIRTREE|LOAD_FILE|PG_READ_FILE|PG_READ_BINARY_FILE|LO_EXPORT|DBLINK)\b'
                                    r'|\b(?:SLEEP|PG_SLEEP|BENCHMARK|WAITFOR)\b|\bUNION\s+(?:ALL\s+)?SELECT\b'
                                    r'|\bINTO\s+(?:OUTFILE|DUMPFILE)\b|\bCOPY\b[\s\S]*\bTO\b'
                                    r'|;\s*(?:DROP|TRUNCATE|DELETE|INSERT|UPDATE)\b', text))
            if not attack:
                for i in range(1, len(tokens)-2):
                    if tokens[i-1].text.upper() in ('OR', 'AND') and tokens[i+1].text == '=':
                        left, right = tokens[i], tokens[i+2]
                        if left.kind == right.kind and left.kind in ('number', 'string') and left.text == right.text:
                            attack = True
            decisions.append(int(attack))
        return np.asarray(decisions, dtype=np.int64)


def observed_action(sql, operation):
    """Recognize basic grants, membership and a single flat tenant SELECT.

    No fixture catalog, parent SQL, generated expected action or annotation is
    available here. Joins, procedural calls and dynamic expressions abstain.
    """
    text = lexical_text(sql)
    if text is None:
        return None
    match = re.fullmatch(rf'grant (select|update|select,update) on (?:table |object::)?({RESOURCE}) to ({PRINCIPAL})( with grant option)?', text)
    if match:
        privileges, resource, recipient, option = match.groups()
        return {'action': 'grant', 'resources': [resource], 'privileges': privileges.split(','),
                'recipient': recipient.replace("'", ''), 'delegable': bool(option)}
    match = re.fullmatch(rf'grant ({IDENT}) to ({IDENT})', text)
    if not match:
        match = re.fullmatch(rf'alter role ({IDENT}) add member ({IDENT})', text)
    if match:
        role, recipient = match.groups()
        return {'action': 'grant', 'resources': ['role:'+role], 'privileges': ['membership'],
                'recipient': recipient, 'delegable': False}
    match = re.fullmatch(rf'select (.+) from ({RESOURCE}) ({IDENT}) where \3\.tenant_key=(\d+)(?: order by \3\.{IDENT}| group by \3\.tenant_key)?', text)
    if not match:
        return None
    projection, resource, alias, tenant = match.groups()
    direct = re.fullmatch(rf'{alias}\.entry_id,{alias}\.({IDENT})', projection)
    aggregate = re.fullmatch(rf'(sum|avg|min|max)\({alias}\.({IDENT})\)as total_value', projection.removeprefix(alias+'.tenant_key,'))
    if direct:
        column, function = direct.group(1), None
    elif aggregate:
        function, column = aggregate.groups()
    else:
        return None
    action = {'action': 'select', 'resources': [resource], 'column': column, 'tenant': int(tenant),
              'aggregate': function, 'related_resource': None}
    if operation.get('workflow') == 'export_rows_v1':
        if not isinstance(operation.get('destination'), str):
            return None
        action.update(action='export', destination=operation['destination'])
    return action


def policy_decision(item):
    document = prepare_input(item)
    context = document.projected['context']
    entry = context.get('policy', {})
    policy = entry.get('value')
    if entry.get('availability') != 'known' or not policy or policy.get('coverage') != 'complete':
        return None, 'complete_policy_unavailable'
    operation_entry = context.get('application_operation', {})
    if operation_entry.get('availability') != 'known':
        return None, 'operation_context_unavailable'
    try:
        operation = json.loads(operation_entry['value']['description'])
    except (TypeError, ValueError, KeyError):
        return None, 'operation_context_unsupported'
    if not isinstance(operation, dict):
        return None, 'operation_context_unsupported'
    action = observed_action(document.projected['sql'], operation)
    if action is None:
        return None, 'SQL_operation_unsupported'
    if not set(action['resources']) <= set(policy.get('scope_resources', [])):
        return None, 'resource_outside_declared_policy_scope'
    effects = set()
    for rule in policy.get('rules', []):
        if action['action'] not in rule['actions'] or not set(action['resources']) <= set(rule['resources']):
            continue
        try:
            constraints = json.loads(rule.get('condition') or '{}')
        except (TypeError, ValueError):
            return None, 'policy_condition_unsupported'
        if not isinstance(constraints, dict) or any(key not in action for key in constraints):
            return None, 'policy_condition_unsupported'
        if all(action[key] == value for key, value in constraints.items()):
            effects.add(rule['effect'])
    if len(effects) != 1 or not effects <= {'allow', 'deny'}:
        return None, 'policy_unmatched_or_contradictory'
    return int(effects.pop() == 'deny'), 'explicit_scoped_policy'


def rule_predictions(name, inputs, legacy=False):
    if name == 'regex_v1':
        predicted = RegexDetector().predict([prepare_input(item).projected['sql'] for item in inputs])
        if not legacy:
            predicted = (predicted != 0).astype(np.int64)
        return predicted.tolist(), ['ok'] * len(inputs), ['historical_regex'] * len(inputs)
    if name == 'regex_v2':
        return RegexV2().predict(inputs).tolist(), ['ok'] * len(inputs), ['revised_text_heuristics'] * len(inputs)
    fallback = RegexV2().predict(inputs) if name == 'regex_policy' else None
    if name not in ('policy_only', 'regex_policy'):
        raise ValueError('Unknown rule version')
    predicted, statuses, reasons = [], [], []
    for index, item in enumerate(inputs):
        value, reason = policy_decision(item)
        if value is None and fallback is not None:
            value, reason = int(fallback[index]), 'regex_fallback:' + reason
        predicted.append(value)
        statuses.append('not_run' if value is None else 'ok')
        reasons.append(reason)
    return predicted, statuses, reasons
