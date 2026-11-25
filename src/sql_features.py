"""Train-only sparse feature branches from SQL and whitelisted predecision context."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn.utils.validation import check_is_fitted

from data_contract import CONTEXT_FIELDS
from normalize import normalize_sql
from sql_views import KEYWORDS, Symbols, views

VERSION = "sql-features-v1"
CONDITION_KEYS = {"action", "resources", "privileges", "recipient", "delegable", "column", "tenant", "aggregate",
                  "related_resource", "destination", "request_id", "rights_effect", "workflow", "destination_kind"}
SEMANTIC_VALUES = {"allow", "deny", "complete", "partial", "select", "update", "insert", "delete", "grant", "revoke",
                   "execute", "export", "membership", "sum", "avg", "min", "max", "create_user", "alter_role",
                   "read_rows_v1", "delegate_access_v1", "export_rows_v1", "application_local_file"}


def project_input(item):
    if isinstance(item, str):
        return {"sql": item, "context": {}}
    if not isinstance(item, dict):
        raise TypeError("Expected SQL text or an input mapping")
    sql = item.get("sql")
    full_record = isinstance(sql, dict)
    if full_record:
        sql = sql.get("text")
    if not isinstance(sql, str):
        raise TypeError("SQL text is required")
    context = {}
    for key in CONTEXT_FIELDS:
        entry = item.get("context", {}).get(key, {})
        availability = entry.get("availability", "unknown")
        known = availability == "known" and (not full_record or entry.get("available_before_decision") is True)
        value = deepcopy(entry.get("value")) if known else None
        if key == "policy" and isinstance(value, dict):
            value = {name: value[name] for name in ("coverage", "scope_resources", "rules", "sensitive_resources") if name in value}
        if key == "permissions" and isinstance(value, dict):
            value = {name: value[name] for name in ("coverage", "scope_resources", "rules") if name in value}
        if key == "application_operation" and isinstance(value, dict):
            value = {name: value[name] for name in ("operation_id", "description") if name in value}
        context[key] = {"availability": availability if known or availability != "known" else "unknown", "value": value}
    return {"sql": sql, "context": context}


def context_view(context, symbols):
    tokens, numbers = [], {}
    def encode(value, path, *, names=False):
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return str(value).upper()
        if isinstance(value, (int, float)):
            return symbols.literal(value, "NUM")
        if isinstance(value, str):
            if value.lower() in SEMANTIC_VALUES:
                return value.upper()
            if names:
                return symbols.resource(value)
            return symbols.literal(value)
        if isinstance(value, list):
            return "[" + ",".join(sorted(encode(x, path, names=names) for x in value)) + "]"
        if isinstance(value, dict):
            return "{" + ",".join(key + ":" + encode(value[key], path + "." + key,
                names=key in {"resources", "recipient", "column", "related_resource"}) for key in sorted(value) if key in CONDITION_KEYS) + "}"
        return "UNKNOWN"
    for field in CONTEXT_FIELDS:
        entry = context.get(field, {"availability": "unknown", "value": None})
        state, value = entry["availability"], entry["value"]
        tokens.append(field.upper() + "_" + state.upper())
        numbers[field + "_known"] = float(state == "known")
        if state != "known" or value is None:
            continue
        if field == "dialect":
            tokens.append("DIALECT=" + str(value).upper())
        elif field == "active_role":
            tokens.append("ACTOR=" + symbols.resource(value))
        elif field in ("permissions", "policy"):
            tokens.append(field + "_coverage=" + str(value.get("coverage", "unknown")).upper())
            resources = value.get("scope_resources", [])
            for resource in resources:
                tokens.append(field + "_scope=" + symbols.resource(resource))
            rules = value.get("rules", [])
            rules = sorted(rules, key=lambda rule: (rule.get("effect", ""), tuple(rule.get("actions", []))))
            numbers[field + "_rules"] = math.log1p(len(rules))
            for rule in rules:
                effect = rule.get("effect", "unknown")
                prefix = field.upper() + "_" + effect.upper()
                for action in sorted(rule.get("actions", [])):
                    tokens.append(prefix + "_ACTION=" + action.upper())
                for resource in rule.get("resources", []):
                    tokens.append(prefix + "_RESOURCE=" + symbols.resource(resource))
                condition = rule.get("condition")
                if condition:
                    try:
                        parsed = json.loads(condition)
                    except (json.JSONDecodeError, TypeError):
                        # Unstructured policy language gets a masked lexical view,
                        # not a guessed allow/deny decision.
                        parsed = None
                        tokens.append(prefix + "_FREE_CONDITION=" + views(str(condition), symbols)["masked"])
                    if isinstance(parsed, dict):
                        for key in sorted(parsed):
                            if key in CONDITION_KEYS:
                                tokens.append(prefix + "_" + key.upper() + "=" + encode(parsed[key], key,
                                    names=key in {"resources", "recipient", "column", "related_resource"}))
            numbers[field + "_scope_size"] = math.log1p(len(resources))
        elif field == "application_operation":
            try:
                parsed = json.loads(value.get("description", ""))
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                for key in sorted(parsed):
                    if key in CONDITION_KEYS:
                        tokens.append("OP_" + key.upper() + "=" + encode(parsed[key], key))
            else:
                tokens.append("OP_TEXT=" + views(value.get("description", ""), symbols)["masked"])
    return " ".join(sorted(tokens)) or "CONTEXT_UNKNOWN", numbers


@dataclass
class FeatureInput:
    projected: dict
    sql: dict
    context_text: str
    context_numbers: dict


def prepare_input(item):
    if isinstance(item, FeatureInput):
        return item
    projected = project_input(item)
    sql = views(projected["sql"])
    text, numbers = context_view(projected["context"], sql["symbols"])
    return FeatureInput(projected, sql, text, numbers)


def _numeric_dicts(documents, context=False):
    return [doc.context_numbers if context else doc.sql["structure"] for doc in documents]


class SQLFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, text_mode="masked", structure=True, context=False, max_features=50000,
                 structure_weight=0.5, context_weight=1.0, scaling=True, select_k=None):
        self.text_mode = text_mode
        self.structure = structure
        self.context = context
        self.max_features = max_features
        self.structure_weight = structure_weight
        self.context_weight = context_weight
        self.scaling = scaling
        self.select_k = select_k

    def _texts(self, documents):
        if self.text_mode == "legacy":
            return [normalize_sql(doc.projected["sql"]) for doc in documents]
        return [doc.sql[self.text_mode] for doc in documents]

    def _blocks(self, documents, fitting=False):
        blocks, names, groups = [], [], []
        if self.text_mode:
            if fitting:
                self.text_vectorizer_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=self.max_features, dtype=np.float32)
            matrix = (self.text_vectorizer_.fit_transform(self._texts(documents)) if fitting else self.text_vectorizer_.transform(self._texts(documents)))
            blocks.append(matrix)
            features = ["sql_text:" + name for name in self.text_vectorizer_.get_feature_names_out()]
            names.extend(features); groups.extend(["sql_text"] * len(features))
        for branch, enabled in (("structure", self.structure), ("context_numeric", self.context)):
            if not enabled:
                continue
            is_context = branch == "context_numeric"
            if fitting:
                vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
                matrix = vectorizer.fit_transform(_numeric_dicts(documents, is_context))
                setattr(self, branch + "_vectorizer_", vectorizer)
                scaler = MaxAbsScaler() if self.scaling else None
                if scaler:
                    matrix = scaler.fit_transform(matrix)
                setattr(self, branch + "_scaler_", scaler)
            else:
                vectorizer = getattr(self, branch + "_vectorizer_")
                matrix = vectorizer.transform(_numeric_dicts(documents, is_context))
                scaler = getattr(self, branch + "_scaler_")
                if scaler:
                    matrix = scaler.transform(matrix)
            matrix = normalize(matrix, norm="l2") if self.scaling else matrix
            weight = self.context_weight if is_context else self.structure_weight
            blocks.append(matrix * weight)
            features = [branch + ":" + name for name in vectorizer.get_feature_names_out()]
            names.extend(features); groups.extend([branch] * len(features))
        if self.context:
            if fitting:
                self.context_vectorizer_ = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), dtype=np.float32,
                                                           lowercase=False, max_features=15000)
            texts = [doc.context_text for doc in documents]
            matrix = self.context_vectorizer_.fit_transform(texts) if fitting else self.context_vectorizer_.transform(texts)
            blocks.append(matrix * self.context_weight)
            features = ["context_text:" + name for name in self.context_vectorizer_.get_feature_names_out()]
            names.extend(features); groups.extend(["context_text"] * len(features))
        if not blocks:
            raise ValueError("At least one feature branch must be enabled")
        result = sparse.hstack(blocks, format="csr", dtype=np.float32)
        result.eliminate_zeros(); result.sort_indices()
        if not np.isfinite(result.data).all() or (result.data < 0).any():
            raise ValueError("Features must be finite and nonnegative")
        if fitting:
            self.feature_names_ = np.array(names, dtype=object)
            self.feature_groups_ = np.array(groups, dtype=object)
        return result

    def fit_transform(self, X, y=None, **fit_params):
        documents = [prepare_input(item) for item in X]
        matrix = self._blocks(documents, fitting=True)
        self.fit_rows_ = len(documents)
        self.selector_ = None
        if self.select_k is not None:
            if y is None:
                raise ValueError("Supervised selection requires training targets")
            self.selector_ = SelectKBest(chi2, k=min(self.select_k, matrix.shape[1]))
            matrix = self.selector_.fit_transform(matrix, y).tocsr()
            keep = self.selector_.get_support()
            self.feature_names_ = self.feature_names_[keep]
            self.feature_groups_ = self.feature_groups_[keep]
        self.n_features_out_ = matrix.shape[1]
        return matrix

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_out_")
        matrix = self._blocks([prepare_input(item) for item in X])
        return self.selector_.transform(matrix).tocsr() if self.selector_ is not None else matrix

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_")
        return self.feature_names_.copy()


CONFIGURATIONS = {
    "legacy_text": dict(text_mode="legacy", structure=False),
    "raw_text": dict(text_mode="raw", structure=False),
    "masked_text": dict(text_mode="masked", structure=False),
    "masked_parts": dict(text_mode="parts", structure=False),
    "structure_only": dict(text_mode=None, structure=True),
    "text_structure": dict(text_mode="masked", structure=True),
    "text_structure_selected": dict(text_mode="masked", structure=True, select_k=1000),
    "text_structure_unscaled": dict(text_mode="masked", structure=True, scaling=False),
    "text_context": dict(text_mode="masked", structure=False, context=True),
    "full": dict(text_mode="masked", structure=True, context=True),
    "structure_context": dict(text_mode=None, structure=True, context=True),
    "full_raw_text": dict(text_mode="raw", structure=True, context=True),
    "full_selected": dict(text_mode="masked", structure=True, context=True, select_k=1000),
    "full_unscaled": dict(text_mode="masked", structure=True, context=True, scaling=False),
    "full_structure_light": dict(text_mode="masked", structure=True, context=True, structure_weight=0.25),
    "full_structure_equal": dict(text_mode="masked", structure=True, context=True, structure_weight=1.0),
}
