"""Leakage, nested scopes, reproducibility and metric-contract regression checks."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from split_protocol import group_folds, group_rows, source_transfer, validate_partition
from scenario_sql import obfuscate
from prepare_splits import reserve_common_sets, write_compressed
from evaluation_metrics import classification_report


def row(oid, query, label=0, *, cohort="legacy", source="unit", grouping=None, dialect=None):
    return {"id": oid, "query": query, "label": label, "cohort": cohort, "source": source,
            "grouping": grouping or {}, "dialect": dialect, "dataset_version_id": cohort}


class SplitProtocolTests(unittest.TestCase):
    def test_transitive_links_and_contextual_duplicates_stay_together(self):
        values = [row("a", "SELECT x FROM first", 0, grouping={"scenario_id": "pair"}),
                  row("b", "SELECT y FROM second", 1, grouping={"scenario_id": "pair"}),
                  row("c", "SELECT y FROM second", 0)]
        group_rows(values)
        self.assertEqual(len({r["group"] for r in values}), 1)

    def test_literal_templates_keep_names_and_link_whitespace_variants(self):
        values = [row("a", " SELECT x FROM first WHERE id=1 "), row("b", "SELECT x FROM first WHERE id=2"),
                  row("c", "SELECT x FROM second WHERE id=1")]
        group_rows(values)
        self.assertEqual(values[0]["group"], values[1]["group"])
        self.assertNotEqual(values[0]["group"], values[2]["group"])

    def test_legacy_generation_prompt_batch_is_a_group(self):
        values = [row("a", "GRANT SELECT ON a TO u", 3, source="generated_groq_mysql_simple"),
                  row("b", "CALL rights_job()", 3, source="generated_groq_mysql_simple"),
                  row("c", "SELECT id FROM orders", 0, source="generated_groq_mysql_simple")]
        group_rows(values)
        self.assertEqual(values[0]["group"], values[1]["group"])
        self.assertNotEqual(values[0]["group"], values[2]["group"])

    def test_known_dynamic_wrapper_does_not_merge_unrelated_payloads(self):
        first, second = "GRANT SELECT ON TABLE lab.a TO receiver;", "CALL lab.assign_review(100);"
        variant1 = obfuscate(first, "postgresql", "privilege", 0)[0]
        variant2 = obfuscate(second, "postgresql", "privilege", 0)[0]
        values = [row("a", first, cohort="controlled", dialect="postgresql"),
                  row("b", second, cohort="controlled", dialect="postgresql"),
                  row("c", variant1, cohort="controlled", dialect="postgresql", grouping={"parent_observation_ids": ["a"]}),
                  row("d", variant2, cohort="controlled", dialect="postgresql", grouping={"parent_observation_ids": ["b"]})]
        group_rows(values)
        self.assertEqual(values[0]["group"], values[2]["group"])
        self.assertEqual(values[1]["group"], values[3]["group"])
        self.assertNotEqual(values[0]["group"], values[1]["group"])

    def test_missing_parent_is_an_error(self):
        with self.assertRaises(ValueError):
            group_rows([row("a", "SELECT 1", grouping={"parent_observation_ids": ["missing"]})])

    def test_source_transfer_prunes_only_source_side_overlap(self):
        values = [row("a0", "SELECT x FROM clean_zero", 0, cohort="sqlstateguard", source="a"),
                  row("a1", "SELECT x FROM clean_one", 1, cohort="sqlstateguard", source="a"),
                  row("overlap", "SELECT x FROM shared WHERE id=1", 1, cohort="sqlstateguard", source="a"),
                  row("b0", "SELECT x FROM shared WHERE id=2", 0, cohort="sqlstateguard", source="b"),
                  row("b1", "SELECT x FROM target_one", 1, cohort="sqlstateguard", source="b")]
        group_rows(values)
        assigned = source_transfer(values, "b")
        self.assertEqual(assigned["overlap"], "excluded")
        self.assertEqual({oid for oid, role in assigned.items() if role == "test"}, {"b0", "b1"})
        self.assertEqual({oid for oid, role in assigned.items() if role == "train"}, {"a0", "a1"})

    def grouped_pairs(self):
        values = [row(f"{group}-{label}", f"SELECT x FROM fixture_{group}_{label}", label,
                      grouping={"scenario_id": f"pair-{group}"}) for group in range(6) for label in (0, 1)]
        group_rows(values)
        return values

    def test_nested_folds_are_reproducible_and_keep_outer_test_outside_tuning(self):
        values = self.grouped_pairs()
        outer = group_folds(values, 3)
        self.assertEqual(outer, group_folds(list(reversed(values)), 3))
        for fold in range(3):
            assignment = {r["id"]: "test" if outer[r["id"]] == fold else "train" for r in values}
            validate_partition(values, assignment)
            train = [r for r in values if assignment[r["id"]] == "train"]
            inner = group_folds(train, 2)
            self.assertFalse(set(inner) & {oid for oid, role in assignment.items() if role == "test"})
            for validation in range(2):
                roles = {r["id"]: "validation" if inner[r["id"]] == validation else "train" for r in train}
                validate_partition(train, roles, required=("train", "validation"))

    def test_insufficient_independent_class_groups_are_rejected(self):
        values = [row("a", "SELECT x FROM one", 0), row("b", "SELECT x FROM two", 0), row("c", "SELECT x FROM three", 1)]
        group_rows(values)
        with self.assertRaises(ValueError):
            group_folds(values, 2)

    def test_equal_stratum_scores_use_sample_balance_before_identifier_ties(self):
        values = []
        for group, (threat, size) in enumerate([('exfiltration', 360), ('exfiltration', 120), ('exfiltration', 120),
                                               ('privilege', 240), ('privilege', 120), ('privilege', 120), ('privilege', 120)]):
            values.extend({'id': f'{group}:{index}', 'group': str(group), 'label': index % 2, 'threat': threat} for index in range(size))
        from collections import Counter
        assignment = group_folds(values, 3, strata=lambda r: r['threat'] + ':' + str(r['label']))
        self.assertEqual(sorted(Counter(assignment.values()).values()), [360, 360, 480])

    def test_diagnostic_random_split_reports_leakage_instead_of_hiding_it(self):
        values = self.grouped_pairs()
        roles = {r["id"]: "train" if int(r["id"].split("-")[0]) < 3 else "test" for r in values}
        # Force one complete pair into the same root as a pair in the other part.
        values[-1]["group"] = values[-2]["group"] = values[0]["group"]
        with self.assertRaises(ValueError):
            validate_partition(values, roles)
        checked = validate_partition(values, roles, enforce_groups=False)
        self.assertGreater(checked["cross_part_groups"], 0)
        with self.assertRaises(ValueError):
            validate_partition(values, {**roles, "outside": "test"})

    def test_compressed_memberships_are_identical_across_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "one.gz", Path(temporary) / "two.gz"
            self.assertEqual(write_compressed(first, [{"id": "a", "roles": "TE"}]),
                             write_compressed(second, [{"id": "a", "roles": "TE"}]))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_llm_development_and_few_shots_do_not_use_final_test_groups(self):
        values, registry, plans = [], {}, {}
        for cohort, tasks, outer_name, inner_name in (
            ("sqlstateguard", ["sqli"], "transfer-web_app-to-wordpress", "transfer-web_app-to-wordpress-inner"),
            ("controlled", ["privilege", "exfiltration"], "controlled-outer", "controlled-inner-0")):
            outer, inner = {}, {}
            registry[outer_name], registry[inner_name] = {"split_id": outer_name}, {"split_id": inner_name}
            for task in tasks:
                for role, amount in (("train", 1), ("validation", 5), ("test", 1)):
                    for label in (0, 1):
                        for index in range(amount):
                            oid = f"{task}:{role}:{label}:{index}"
                            values.append({"id": oid, "cohort": cohort, "threat": task, "label": label,
                                           "dataset_version_id": cohort, "group": f"{task}:{role}"})
                            outer[oid] = "test" if role == "test" else "train"
                            if role != "test":
                                inner[oid] = role
            plans[outer_name], plans[inner_name] = {0: outer}, {0: inner}
        with tempfile.TemporaryDirectory() as temporary, patch("prepare_splits.OUTPUT", Path(temporary)):
            result = reserve_common_sets(values, registry, plans)
            self.assertEqual((result["pilot_rows"], result["few_shot_candidate_rows"]), (30, 6))
            bad = deepcopy(values)
            bad[0]["group"] = "sqli:validation"
            with self.assertRaises(ValueError):
                reserve_common_sets(bad, registry, plans)


class MetricContractTests(unittest.TestCase):
    def test_binary_F1_recall_and_FPR(self):
        report = classification_report([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(report["primary_f1"], 0.8)
        metrics = report["successful_decision_metrics"]
        self.assertEqual(metrics["attack_recall"], 1)
        self.assertEqual(metrics["false_positive_rate"], 0.5)

    def test_legacy_macro_F1_is_distinct_from_collapsed_attack_detection(self):
        report = classification_report([0, 1, 2, 3, 4], [0, 1, 1, 3, 4], label_space="legacy_five_class")
        self.assertAlmostEqual(report["primary_f1"], (1 + 2 / 3 + 0 + 1 + 1) / 5)
        self.assertEqual(report["successful_decision_metrics"]["attack_recall"], 1)

    def test_missing_decision_never_becomes_legitimate_or_primary_score(self):
        result = classification_report([0, 1, 1], [0, 1, None], statuses=["ok", "ok", "parse_error"])
        self.assertIsNone(result["primary_f1"])
        self.assertFalse(result["eligible_for_primary_ranking"])
        self.assertAlmostEqual(result["decision_coverage"], 2 / 3)
        self.assertEqual(result["successful_decision_metrics"]["f1"], 1)
        with self.assertRaises(ValueError):
            classification_report([0], [0], statuses=["api_error"])

    def test_undefined_rate_is_not_reported_as_zero(self):
        result = classification_report([1, 1], [1, 1])
        self.assertIsNone(result["successful_decision_metrics"]["false_positive_rate"])
        self.assertIsNone(classification_report([0], [0])["primary_f1"])


if __name__ == "__main__":
    unittest.main()
