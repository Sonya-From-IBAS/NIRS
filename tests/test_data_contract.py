"""Check target ambiguity, provenance, revisions and detector input leakage."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_contract as contract


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = json.loads((ROOT / "tests/fixtures/contract_examples.json").read_text(encoding="utf-8"))["records"]
        cls.by_id = {item["observation_id"]: item for item in cls.examples}

    def example(self, number):
        return deepcopy(self.by_id[f"obs-{number:03d}"])

    def test_schema_and_every_contract_fixture(self):
        schema = json.loads(contract.SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        for example in self.examples:
            with self.subTest(example=example["observation_id"]):
                contract.validate_observation(example)
        self.assertEqual(len(self.examples), 17)

    def test_review_needed_never_silently_becomes_benign(self):
        example = self.example(12)
        self.assertIsNone(contract.current_annotation(example)["target_label"])
        example["annotations"][-1]["target_label"] = 0
        with self.assertRaises(ValueError):
            contract.validate_observation(example)

    def test_missing_context_cannot_hide_values_or_post_event_information(self):
        example = self.example(12)
        example["context"]["active_role"]["value"] = "administrator"
        with self.assertRaises(ValueError):
            contract.validate_observation(example)
        example = self.example(1)
        example["context"]["active_role"]["available_before_decision"] = False
        with self.assertRaises(ValueError):
            contract.validate_observation(example)

    def test_source_labels_and_generator_outputs_are_not_verified_targets(self):
        for kind in ("source_label", "generator_output", "syntax_only"):
            example = self.example(2)
            example["annotations"][-1]["evidence"][0]["kind"] = kind
            with self.subTest(evidence=kind), self.assertRaises(ValueError):
                contract.validate_observation(example)

    def test_model_input_excludes_annotations_source_groups_and_execution(self):
        example = self.example(2)
        before = contract.detector_input(example, include_context=True)
        example["source"]["source_label"] = {"malicious": True, "generator_id": "hidden"}
        example["grouping"]["campaign_id"] = "hidden-attack-campaign"
        example["execution"] = {"outcome": "failed", "evidence_ref": "test:failure"}
        example["annotations"][-1]["reason"] = "Different retrospective annotation explanation"
        self.assertEqual(before, contract.detector_input(example, include_context=True))
        self.assertEqual(set(before), {"sql", "context"})
        for field in before["context"].values():
            self.assertEqual(set(field), {"availability", "value"})
        self.assertNotIn("policy_id", before["context"]["policy"]["value"])
        self.assertNotIn("version", before["context"]["policy"]["value"])

    def test_detector_projection_does_not_mutate_or_normalize_record(self):
        example = self.example(5)
        example["sql"]["text"] = "  SELECT 1 /* keep */\r\n;  "
        original = deepcopy(example)
        projected = contract.detector_input(example, include_context=True)
        self.assertEqual(projected["sql"], original["sql"]["text"])
        projected["context"]["permissions"]["value"]["rules"].clear()
        self.assertEqual(example, original)

    def test_same_sql_can_have_different_labels_under_verified_context(self):
        for benign_id, malicious_id in ((6, 7), (8, 9), (10, 11)):
            benign, malicious = self.example(benign_id), self.example(malicious_id)
            self.assertEqual(contract.detector_input(benign), contract.detector_input(malicious))
            self.assertNotEqual(contract.detector_input(benign, include_context=True),
                                contract.detector_input(malicious, include_context=True))
            self.assertEqual(benign["grouping"]["group_id"], malicious["grouping"]["group_id"])
            self.assertEqual(contract.current_annotation(benign)["target_label"], 0)
            self.assertEqual(contract.current_annotation(malicious)["target_label"], 1)

    def test_benign_comments_are_not_a_separate_malicious_class(self):
        annotation = contract.current_annotation(self.example(5))
        self.assertEqual(annotation["target_label"], 0)
        self.assertIn("comments", annotation["representation_tags"])
        self.assertEqual(annotation["threat_tags"], [])

    def test_observed_failure_does_not_change_attack_target(self):
        example = self.example(7)
        example["execution"] = {"outcome": "failed", "evidence_ref": "test:db-denied"}
        self.assertEqual(contract.current_annotation(example)["target_label"], 1)
        example["execution"]["evidence_ref"] = None
        with self.assertRaises(ValueError):
            contract.validate_observation(example)

    def test_annotation_correction_keeps_source_label_and_history(self):
        example = self.example(16)
        self.assertEqual(example["source"]["source_label"], 0)
        self.assertEqual(contract.current_annotation(example)["target_label"], 1)
        self.assertIsNone(example["annotations"][0]["target_label"])
        example["annotations"][-1]["supersedes"] = None
        with self.assertRaises(ValueError):
            contract.validate_observation(example)

    def test_fixtures_fragments_and_unreviewed_records_are_not_primary_data(self):
        for fixture in self.examples:
            self.assertFalse(contract.primary_eligible(fixture))
        fragment = self.example(13)
        fragment["source"]["purpose"] = "research"
        self.assertFalse(contract.primary_eligible(fragment))
        unreviewed = self.example(12)
        unreviewed["source"]["purpose"] = "research"
        self.assertFalse(contract.primary_eligible(unreviewed))
        reviewed = self.example(1)
        reviewed["source"]["purpose"] = "research"
        self.assertTrue(contract.paired_context_eligible(reviewed))
        reviewed["context"]["active_role"] = deepcopy(unreviewed["context"]["active_role"])
        self.assertTrue(contract.primary_eligible(reviewed))
        self.assertFalse(contract.paired_context_eligible(reviewed))

    def test_availability_reports_missing_context_in_denominator(self):
        report = contract.context_availability([self.example(1), self.example(12)])
        self.assertEqual(report["records"], 2)
        for counts in report["fields"].values():
            self.assertEqual(counts, {"known": 1, "unknown": 1, "not_applicable": 0})

    def test_unexpected_context_fields_are_rejected_instead_of_leaking(self):
        example = self.example(1)
        example["context"]["ground_truth"] = {"target_label": 0}
        with self.assertRaises(ValidationError):
            contract.validate_observation(example)

    def test_legacy_audit_does_not_infer_dialect_or_rights_from_source_or_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            raw = ('query,label,source,label_name\r\n'
                   'SELECT 1,0,generated_groq_mysql_simple,legit\r\n'
                   'GRANT SELECT ON t TO x,3,generated_groq_mysql_simple,privilege\r\n').encode()
            path.write_bytes(raw)
            report = contract.audit_legacy_csv(path)
            self.assertEqual(report["records"], 2)
            for counts in report["context_availability"].values():
                self.assertEqual(counts, {"known": 0, "unknown": 2, "not_applicable": 0})
            self.assertEqual(path.read_bytes(), raw)
            self.assertFalse(report["target_labels_modified"])

    def test_policy_versions_identify_consistent_rule_content(self):
        policies = {}
        for record in self.examples:
            policy = record["context"]["policy"]["value"]
            if policy is None:
                continue
            key = (policy["policy_id"], policy["version"])
            if key in policies:
                self.assertEqual(policy, policies[key])
            policies[key] = policy


if __name__ == "__main__":
    unittest.main()
