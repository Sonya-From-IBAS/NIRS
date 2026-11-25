"""Deterministic grouping and partition checks, independent of model predictions."""

from collections import Counter, defaultdict
import hashlib
import itertools
import json

import numpy as np

from data_audit import lexical_template
from scenario_sql import restore_static_sql

VERSION = "stage6-groups-v1"
SEED = 20261125


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        self.parent[max(first, second)] = min(first, second)


def group_rows(rows):
    """Link all selected cohorts before choosing any source/outer partition."""
    dsu, seen = UnionFind(len(rows)), {}
    by_id = {row["id"]: index for index, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate observation identities across the selected cohorts")
    reasons = Counter()
    def link(key, index, reason):
        if key in seen:
            dsu.union(index, seen[key])
            reasons[reason] += 1
        else:
            seen[key] = index
    for index, row in enumerate(rows):
        query = row["query"]
        row["exact_sha256"] = sha(query)
        row["strip_sha256"] = sha(query.strip())
        row["raw_template_sha256"] = sha(lexical_template(query))
        row["template_sha256"] = (sha(lexical_template(restore_static_sql(query, row["dialect"])))
                                  if row["cohort"] == "controlled" else row["raw_template_sha256"])
        for name in ("exact", "strip", "template"):
            link((name, row[name + "_sha256"]), index, name)
        for name, value in row.get("grouping", {}).items():
            if name in ("group_id", "scenario_id", "template_id", "campaign_id") and value:
                link(("declared:" + name, value), index, "declared")
        for value in row.get("extra_groups", []):
            link(("declared_extra", value), index, "declared_extra")
        for parent in row.get("grouping", {}).get("parent_observation_ids", []):
            if parent not in by_id:
                raise ValueError("A selected derived observation has an absent parent")
            dsu.union(index, by_id[parent])
            reasons["parent"] += 1
        # The old generator used one label/dialect/complexity prompt per batch.
        # These documented prompt batches are split metadata, never features.
        if row["cohort"] == "legacy" and row["source"].startswith("generated_"):
            link(("legacy_prompt", row["source"], row["label"]), index, "legacy_prompt")
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[dsu.find(index)].append(row)
    for values in members.values():
        identifier = "g6_" + sha("\n".join(sorted(row["id"] for row in values)))
        for row in values:
            row["group"] = identifier
    return {"version": VERSION, "observations": len(rows), "groups": len(members),
            "link_events": dict(reasons), "largest_groups": sorted((len(x) for x in members.values()), reverse=True)[:10]}


def summary(rows):
    counts = Counter(str(row["label"]) for row in rows)
    groups = defaultdict(set)
    sizes = Counter()
    for row in rows:
        groups[str(row["label"])].add(row["group"])
        sizes[row["group"]] += 1
    return {"rows": len(rows), "labels": dict(sorted(counts.items())), "groups": len(sizes),
            "groups_per_label": {key: len(value) for key, value in sorted(groups.items())},
            "largest_group_rows": sorted(sizes.values(), reverse=True)[:8]}


def group_folds(rows, n_splits, *, strata=None):
    """Balance labels (or declared strata) with groups intact and stable ties."""
    if len({row["group"] for row in rows}) < n_splits:
        raise ValueError("Too few independent groups")
    stratify = strata or (lambda row: str(row["label"]))
    classes = sorted({stratify(row) for row in rows})
    positions = {value: index for index, value in enumerate(classes)}
    vectors = defaultdict(lambda: np.zeros(len(classes), dtype=np.int64))
    for row in rows:
        vectors[row["group"]][positions[stratify(row)]] += 1
    totals = np.sum(list(vectors.values()), axis=0)
    for index, key in enumerate(classes):
        if sum(value[index] > 0 for value in vectors.values()) < n_splits:
            raise ValueError(f"Stratum {key} occurs in fewer than {n_splits} groups")
    groups = sorted(vectors, key=lambda key: (-float(np.max(vectors[key] / totals)), -int(vectors[key].sum()), sha(str(SEED) + key)))
    matrix = np.array([vectors[key] for key in groups])
    if len(groups) <= 8:
        best = None
        for candidate in itertools.product(range(n_splits), repeat=len(groups) - 1):
            assignment = (0, *candidate)
            if set(assignment) != set(range(n_splits)):
                continue
            counts = np.array([matrix[np.array(assignment) == fold].sum(axis=0) for fold in range(n_splits)])
            if np.any(counts == 0):
                continue
            # Equivalent label allocations must not be separated by machine
            # rounding before applying the sample-size balance tie-breaker.
            score = round(float(np.square(counts / totals - 1 / n_splits).sum()), 14)
            row_score = float(np.square(counts.sum(axis=1) / totals.sum() - 1 / n_splits).sum())
            key = (score, row_score, assignment)
            if best is None or key < best:
                best = key
        if best is None:
            raise ValueError("No group partition covers every requested stratum")
        chosen = dict(zip(groups, best[2]))
    else:
        counts = np.zeros((n_splits, len(classes)), dtype=np.int64)
        chosen = {}
        denominator = np.square(totals.astype(float))
        for group in groups:
            vector = vectors[group]
            deltas = ((2 * counts * vector + vector * vector) / denominator).sum(axis=1)
            minimum = float(deltas.min())
            candidates = [fold for fold in range(n_splits) if abs(float(deltas[fold]) - minimum) < 1e-15]
            fold = min(candidates, key=lambda value: (int(counts[value].sum()), sha(f"{SEED}:{group}:{value}")))
            chosen[group] = fold
            counts[fold] += vector
        if np.any(counts == 0):
            raise ValueError("Greedy grouping did not cover each class in every fold")
    return {row["id"]: chosen[row["group"]] for row in rows}


def validate_partition(rows, assignment, *, required=("train", "test"), allow_excluded=True, enforce_groups=True):
    by_id = {row["id"]: row for row in rows}
    if set(assignment) != set(by_id):
        raise ValueError("Partition must explicitly account for every cohort observation")
    allowed = {"train", "validation", "test"} | ({"excluded"} if allow_excluded else set())
    if not set(assignment.values()) <= allowed:
        raise ValueError("Unknown partition role")
    group_roles, text_roles, template_roles = defaultdict(set), defaultdict(set), defaultdict(set)
    parts = defaultdict(list)
    for oid, part in assignment.items():
        row = by_id[oid]
        parts[part].append(row)
        if part != "excluded":
            group_roles[row["group"]].add(part)
            text_roles[row["exact_sha256"]].add(part)
            template_roles[row["template_sha256"]].add(part)
    overlaps = [sum(len(value) > 1 for value in mapping.values()) for mapping in (group_roles, text_roles, template_roles)]
    if enforce_groups and any(overlaps):
        raise ValueError("A group, exact SQL or lexical template crosses partition roles")
    for part in required:
        if not parts[part]:
            raise ValueError("Required partition is empty: " + part)
    labels = set(row["label"] for row in rows)
    for part in required:
        if {row["label"] for row in parts[part]} != labels:
            raise ValueError("Required partition misses a reference class: " + part)
    return {"parts": {part: summary(values) for part, values in sorted(parts.items())},
            "cross_part_groups": overlaps[0], "cross_part_exact_SQL": overlaps[1], "cross_part_lexical_templates": overlaps[2]}


def source_transfer(rows, target_source):
    target = [row for row in rows if row["source"] == target_source]
    if not target:
        raise ValueError("Unknown target application")
    target_groups = {row["group"] for row in target}
    assignment = {row["id"]: ("test" if row["source"] == target_source else
                              "excluded" if row["group"] in target_groups else "train") for row in rows}
    validate_partition(rows, assignment)
    return assignment
