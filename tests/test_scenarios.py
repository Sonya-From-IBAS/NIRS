"""Scenario labels follow actual SQL and policy; variants keep parent semantics."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate_scenarios import (audit_corpus, build_corpus, build_scenario, canonical, family_links,
                                case_scenario, make_record)
from scenario_sql import (canonical_sql, decide, encoded_integer, extract_action,
                          obfuscate, restore_static_sql)
from data_contract import detector_input
from data_audit import lexical_template


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.records, cls.details = build_corpus()

    def test_complete_corpus_has_agreed_balance_and_counterbalancing(self):
        report = audit_corpus(self.catalog, self.records, self.details)
        self.assertEqual(report["records"], 1200)
        self.assertEqual(report["parent_linked_variants"], 400)
        self.assertEqual(report["SQL_plus_context_label_conflicts"], 0)
        self.assertEqual(report["family_group_ids"], 10)
        self.assertLess(report["conservative_split_groups"], 10)
        self.assertTrue(all(x == {"0": 200, "1": 200} for x in report["by_block_and_target"].values()))

    def test_policy_changes_the_target_without_changing_sql(self):
        scenario, queries = build_scenario("privilege", "p01-object-scope", "postgresql", 0)
        effect = extract_action(queries[0], "postgresql", scenario)
        self.assertEqual(decide(effect, scenario["policy"]), 0)
        changed = deepcopy(scenario["policy"])
        for rule in changed["rules"]:
            rule["effect"] = "deny" if rule["effect"] == "allow" else "allow"
        self.assertEqual(decide(effect, changed), 1)

    def test_shared_templates_across_families_are_linked_for_splits(self):
        membership = {oid: group["group_id"] for group in family_links(self.records) for oid in group["observation_ids"]}
        templates = {}
        for row in self.records:
            key = lexical_template(restore_static_sql(row["sql"]["text"], row["context"]["dialect"]["value"]))
            if key in templates:
                self.assertEqual(membership[row["observation_id"]], templates[key])
            templates[key] = membership[row["observation_id"]]
        self.assertEqual(len(membership), 1200)

    def test_changed_projected_column_changes_actual_policy_decision(self):
        scenario, queries = build_scenario("exfiltration", "e01-column-scope", "oracle", 0)
        effect = extract_action(queries[0], "oracle", scenario)
        self.assertEqual(decide(effect, scenario["policy"]), 0)
        changed_sql = queries[0].replace("t.amount", "t.discount")
        changed = extract_action(changed_sql, "oracle", scenario)
        self.assertEqual(decide(changed, scenario["policy"]), 1)

    def test_wrong_dialect_grant_is_rejected(self):
        scenario, queries = build_scenario("privilege", "p01-object-scope", "postgresql", 0)
        with self.assertRaises(ValueError):
            extract_action(queries[0], "oracle", scenario)

    def test_unknown_or_stacked_sql_is_not_assigned_a_label(self):
        scenario, queries = build_scenario("privilege", "p01-object-scope", "postgresql", 0)
        for text in (queries[0] + " DROP TABLE lab_commerce.orders;", "SELECT * FROM missing;", "GRANT ALL ON x TO y;"):
            with self.subTest(sql=text), self.assertRaises(ValueError):
                extract_action(text, "postgresql", scenario)

    def test_conflicting_or_absent_policy_is_not_guessed(self):
        scenario, queries = build_scenario("privilege", "p01-object-scope", "mysql", 0)
        effect = extract_action(queries[0], "mysql", scenario)
        missing = deepcopy(scenario["policy"])
        missing["rules"] = []
        with self.assertRaises(ValueError):
            decide(effect, missing)
        contradictory = deepcopy(scenario["policy"])
        extra = deepcopy(contradictory["rules"][0])
        extra["effect"] = "deny"
        contradictory["rules"].append(extra)
        with self.assertRaises(ValueError):
            decide(effect, contradictory)

    def test_all_variant_payloads_restore_parent_and_match_context(self):
        by_id = {r["observation_id"]: r for r in self.records}
        for row in self.records:
            if row["grouping"]["parent_observation_ids"]:
                parent = by_id[row["grouping"]["parent_observation_ids"][0]]
                dialect = row["context"]["dialect"]["value"]
                self.assertEqual(restore_static_sql(row["sql"]["text"], dialect), canonical_sql(parent["sql"]["text"]))
                self.assertEqual(row["context"], parent["context"])

    def test_tampered_variant_is_not_silently_given_parent_target(self):
        scenario, queries = build_scenario("exfiltration", "e01-column-scope", "postgresql", 0)
        parent, _ = make_record(scenario, 0, queries[0], "a" * 64)
        variant, tags, _ = obfuscate(queries[0], "postgresql", "exfiltration", 0)
        changed = variant.replace("t.amount", "t.discount")
        with self.assertRaises(ValueError):
            make_record(scenario, 0, changed, "a" * 64, parent=parent, tags=tags)

    def test_procedure_effect_is_extracted_from_its_source(self):
        scenario, queries = build_scenario("privilege", "p05-privileged-routine", "mssql", 0)
        extract_action(queries[0], "mssql", scenario)
        name = next(iter(scenario["routine_definitions"]))
        changed = deepcopy(scenario)
        changed["routine_definitions"][name]["ddl"] = changed["routine_definitions"][name]["ddl"].replace("GRANT SELECT", "GRANT UPDATE")
        with self.assertRaises(ValueError):
            extract_action(queries[0], "mssql", changed)

    def test_export_route_is_predecision_context_not_a_sql_keyword(self):
        scenario, queries = build_scenario("exfiltration", "e05-export-destination", "postgresql", 0)
        self.assertEqual(queries[0], queries[1])
        first = extract_action(queries[0], "postgresql", case_scenario(scenario, 0))
        second = extract_action(queries[1], "postgresql", case_scenario(scenario, 1))
        self.assertEqual((decide(first, scenario["policy"]), decide(second, scenario["policy"])), (0, 1))
        self.assertNotEqual(first["destination"], second["destination"])

    def test_scalar_construction_keeps_integer_values_in_all_dialects(self):
        for dialect in ("mysql", "postgresql", "mssql", "oracle"):
            scenario, queries = build_scenario("exfiltration", "e02-row-scope", dialect, 0)
            encoded = queries[0].replace("= 11", "= " + encoded_integer(11, dialect))
            self.assertEqual(extract_action(encoded, dialect, scenario), extract_action(queries[0], dialect, scenario))

    def test_provenance_groups_and_annotations_are_not_detector_inputs(self):
        record = deepcopy(self.records[0])
        before = detector_input(record, include_context=True)
        record["source"]["generator"]["provider"] = "different-author"
        record["grouping"]["template_id"] = "different-hidden-id"
        record["annotations"][0]["reason"] = "changed retrospective explanation"
        self.assertEqual(before, detector_input(record, include_context=True))
        self.assertEqual(set(before), {"sql", "context"})


if __name__ == "__main__":
    unittest.main()
