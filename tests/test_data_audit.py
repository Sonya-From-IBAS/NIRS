"""Audit diagnostics must preserve SQL and must not invent target labels."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_audit import audit_rows, diagnose, lexical_template
from public_audit import remove_record_terminator
from source_fetch import git_blob_hash
from source_catalog import audit_mirror
from source_ingest import make_record
from data_contract import validate_observation, detector_input, paired_context_eligible


class AuditTests(unittest.TestCase):
    def test_group_template_masks_literals_but_preserves_object_names(self):
        first = lexical_template("SELECT name FROM users WHERE id=1 AND state='x'")
        self.assertEqual(first, lexical_template("select name from users where id=42 and state='y'"))
        self.assertNotEqual(first, lexical_template("SELECT name FROM orders WHERE id=1 AND state='x'"))

    def test_executable_comments_are_not_silently_removed_from_grouping(self):
        self.assertNotEqual(lexical_template("SELECT 1 /*! UNION SELECT 2 */"), lexical_template("SELECT 1"))
        self.assertEqual(lexical_template("SELECT 1 /* note */"), lexical_template("SELECT 2"))

    def test_parse_or_execution_success_is_not_claimed_by_a_unit_hint(self):
        result = diagnose("SELECT * FROM orders WHERE '")
        self.assertEqual(result["unit_hint"], "full_sql_candidate")
        self.assertTrue(result["odd_single_quote_count"])
        self.assertNotIn("target_label", result)

    def test_numeric_predicate_payloads_are_fragments_not_full_queries(self):
        self.assertEqual(diagnose("1 OR 1=1")['unit_hint'], "fragment_candidate")
        self.assertEqual(diagnose("UNION SELECT password FROM users")['unit_hint'], "fragment_candidate")
        self.assertEqual(diagnose("ordinary customer text")['unit_hint'], "non_sql_candidate")

    def test_conflicts_are_reported_without_overwriting_source_labels_or_sql(self):
        rows = [{"query": " SELECT 1 ", "label": "0", "label_name": "legit", "source": "a"},
                {"query": "SELECT 1", "label": "1", "label_name": "sqli", "source": "b"}]
        original = deepcopy(rows)
        summary, diagnostics = audit_rows(rows)
        self.assertEqual(rows, original)
        self.assertEqual(summary['strip_duplicates']['mixed_source_label_groups'], 1)
        self.assertNotEqual(diagnostics[0]['exact_sha256'], diagnostics[1]['exact_sha256'])

    def test_comment_marker_signals_do_not_turn_benign_queries_into_attacks(self):
        rows = [{"query": "SELECT 1 /* normal comment */", "label": "0", "label_name": "legit", "source": "test"}]
        summary, details = audit_rows(rows)
        self.assertEqual(rows[0]['label'], "0")
        self.assertEqual(summary['source_label_counts'], {"legit": 1})
        self.assertIn("comments", details[0]['representation_signals'])

    def test_only_the_record_separator_is_removed(self):
        self.assertEqual(remove_record_terminator("SELECT 1-- \r\n"), "SELECT 1-- ")
        self.assertEqual(remove_record_terminator(" SELECT 1  \n"), " SELECT 1  ")

    def test_pinned_git_blob_hash_uses_the_git_object_format(self):
        self.assertEqual(git_blob_hash(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

    def test_headerless_mirror_keeps_first_record_and_rejects_nonbinary_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mirror.csv"
            path.write_text('"SELECT 1",1,\n"SELECT 2",0,\n"fragment",invalid,\n', encoding="utf-8")
            result = audit_mirror(path, {"SELECT 1"})
            self.assertEqual(result['valid_binary_label_records'], 2)
            self.assertEqual(result['positive_records'], 1)
            self.assertEqual(result['positive_strip_matches'], 1)

    def test_release_label_mapping_is_separate_from_missing_role_context(self):
        row = {"query": "SELECT salary FROM payroll;", "label": 0, "application": "web_app",
               "source_record_id": "normal.txt:0", "attack_family": "normal"}
        record = make_record(row, "test-version", "a" * 64, False)
        validate_observation(record)
        self.assertEqual(record['annotations'][0]['target_label'], 0)
        self.assertEqual(record['source']['origin'], "public")
        self.assertFalse(paired_context_eligible(record))
        self.assertEqual(detector_input(record), {"sql": row['query']})
        ambiguous = make_record(row, "test-version", "a" * 64, True)
        validate_observation(ambiguous)
        self.assertIsNone(ambiguous['annotations'][0]['target_label'])


if __name__ == "__main__":
    unittest.main()
