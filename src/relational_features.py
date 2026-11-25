"""Observed SQL/declared-policy relationships, without an annotation oracle."""

import json
import math
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler, normalize

from sql_features import SQLFeatureEncoder
from sql_views import views

VERSION = "policy-relations-v1"


def policy_relations(document):
    # Fresh SQL symbols exclude names introduced later by context serialization.
    sql = views(document.projected["sql"])
    names = {str(value).lower() for _, value in sql["symbols"].names}
    strings = {value.lower() for (kind, value) in sql["symbols"].values if kind == "STR"}
    numbers = {value for (kind, value) in sql["symbols"].values if kind == "NUM"}
    words = set(sql["masked"].split())
    result = {}
    context = document.projected["context"]
    operation = (context.get("application_operation", {}).get("value") or {}).get("description", "")
    try:
        operation = json.loads(operation)
    except (json.JSONDecodeError, TypeError):
        operation = {}
    if not isinstance(operation, dict):
        operation = {}
    def mentioned(value):
        if value.startswith("role:"):
            value = value[5:]
        pieces = re.split(r"[.@]", value)
        pieces = [part.strip("'\"`[]").lower() for part in pieces if part]
        return bool(pieces) and all(part in names or part in strings for part in pieces)
    for branch in ("permissions", "policy"):
        entry = context.get(branch, {})
        if entry.get("availability") != "known" or not isinstance(entry.get("value"), dict):
            continue
        for rule in entry["value"].get("rules", []):
            effect = rule.get("effect")
            if effect not in ("allow", "deny"):
                continue
            prefix = branch + "_" + effect
            resources = rule.get("resources", [])
            result[prefix + "_resource_mentions"] = result.get(prefix + "_resource_mentions", 0) + sum(mentioned(x) for x in resources)
            actions = rule.get("actions", [])
            action_words = {"select": {"SELECT"}, "grant": {"GRANT", "MEMBER"}, "execute": {"CALL", "EXEC", "EXECUTE"},
                            "update": {"UPDATE"}, "delete": {"DELETE"}, "export": {"COPY", "OUTFILE"}}
            result[prefix + "_action_mentions"] = result.get(prefix + "_action_mentions", 0) + sum(bool(words & action_words.get(x, {x.upper()})) for x in actions)
            try:
                conditions = json.loads(rule.get("condition") or "null")
            except (json.JSONDecodeError, TypeError):
                conditions = None
            if not isinstance(conditions, dict):
                continue
            for key, value in conditions.items():
                present = None
                if key in ("resources",):
                    present = sum(mentioned(item) for item in value) if isinstance(value, list) else None
                elif key in ("column", "recipient", "related_resource") and isinstance(value, str):
                    present = int(mentioned(value))
                elif key in ("tenant", "request_id") and isinstance(value, (int, float)):
                    present = int(str(value) in numbers)
                elif key == "aggregate" and isinstance(value, str):
                    present = int(value.upper() in words)
                elif key == "delegable" and isinstance(value, bool):
                    present = int(value == ({"GRANT", "OPTION"} <= words))
                elif key == "privileges" and isinstance(value, list):
                    present = sum(str(item).upper() in words for item in value)
                elif key == "destination" and isinstance(value, str) and operation.get("workflow") == "export_rows_v1":
                    present = int(value == operation.get("destination"))
                if present is not None:
                    feature = prefix + "_condition_" + key + "_mentions"
                    result[feature] = result.get(feature, 0) + present
    return {key: math.log1p(value) for key, value in result.items()}


class RelationalFeatureEncoder(SQLFeatureEncoder):
    """The same fitted feature API with an additional relational numeric branch."""
    def _blocks(self, documents, fitting=False):
        if self.text_mode or self.structure or self.context:
            base = super()._blocks(documents, fitting)
        else:
            base = sparse.csr_matrix((len(documents), 0), dtype=np.float32)
            if fitting:
                self.feature_names_ = np.array([], dtype=object)
                self.feature_groups_ = np.array([], dtype=object)
        dictionaries = [policy_relations(item) for item in documents]
        if fitting:
            self.relations_vectorizer_ = DictVectorizer(dtype=np.float32)
            extra = self.relations_vectorizer_.fit_transform(dictionaries)
            self.relations_scaler_ = MaxAbsScaler() if extra.shape[1] else None
            if self.relations_scaler_ is not None:
                extra = self.relations_scaler_.fit_transform(extra)
        else:
            extra = self.relations_vectorizer_.transform(dictionaries)
            if self.relations_scaler_ is not None:
                extra = self.relations_scaler_.transform(extra)
        if extra.shape[1]:
            extra = normalize(extra, norm="l2")
        if fitting:
            names = ["policy_relations:" + name for name in self.relations_vectorizer_.get_feature_names_out()]
            self.feature_names_ = np.concatenate([self.feature_names_, np.array(names, dtype=object)])
            self.feature_groups_ = np.concatenate([self.feature_groups_, np.array(["policy_relations"] * len(names), dtype=object)])
        result = sparse.hstack([base, extra], format="csr", dtype=np.float32)
        if not result.shape[1]:
            raise ValueError("No relational features available; enable a SQL feature branch")
        result.eliminate_zeros(); result.sort_indices()
        return result


CONFIGURATIONS = {
    "relations_only": dict(text_mode=None, structure=False, context=False),
    "text_context_relations": dict(text_mode="masked", structure=False, context=True),
    "structure_context_relations": dict(text_mode=None, structure=True, context=True),
    "full_relations": dict(text_mode="masked", structure=True, context=True),
}
