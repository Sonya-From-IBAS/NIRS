"""Selection invariants that can change experimental conclusions."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ml_models import choose_threshold, new_classifier, feature_candidates, new_encoder
from rule_detectors import observed_action, policy_decision, rule_predictions


class ModelSelectionTests(unittest.TestCase):
    def test_threshold_moves_equal_scores_together_and_matches_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            truth = rng.integers(0, 2, 30)
            scores = rng.integers(-3, 4, 30).astype(float)
            selected = choose_threshold(truth, scores)
            candidates = [np.nextafter(scores.max(), np.inf), *np.unique(scores)]
            def ranking(t):
                predicted = scores >= t
                return (-f1_score(truth, predicted, zero_division=0),
                        int(np.count_nonzero(predicted & (truth == 0))),
                        -int(np.count_nonzero(predicted & (truth == 1))), -t)
            self.assertEqual(selected, min(candidates, key=ranking))

    def test_invalid_scores_and_alignment_are_rejected(self):
        for truth, scores in (([0, 1], [1]), ([0, 2], [0, 1]), ([0, 1], [0, np.nan])):
            with self.assertRaises(ValueError):
                choose_threshold(truth, scores)

    def test_factories_never_share_fitted_estimators_or_feature_state(self):
        first = new_classifier('LogisticRegression', {'C': .05, 'class_weight': 'balanced'})
        second = new_classifier('LogisticRegression', {'C': .05, 'class_weight': 'balanced'})
        first.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        self.assertFalse(hasattr(second, 'coef_'))
        spec = feature_candidates('controlled')[0]
        a, b = new_encoder(spec), new_encoder(spec)
        a.fit(['SELECT 1', 'GRANT SELECT ON t TO r'], [0, 1])
        self.assertFalse(hasattr(b, 'fit_rows_'))

    def test_rule_revision_does_not_call_comments_or_literal_keywords_attacks(self):
        inputs = ['SELECT 1 /* comment */', "SELECT 'DROP TABLE'", 'SELECT CONCAT(first_name,last_name) FROM people']
        self.assertEqual(rule_predictions('regex_v2', inputs)[0], [0, 0, 0])
        self.assertEqual(rule_predictions('regex_v2', ['SELECT id FROM users WHERE 1=1', 'SELECT id FROM users WHERE 1=2'])[0], [0, 0])
        self.assertEqual(rule_predictions('regex_v2', ['SELECT id FROM users WHERE id=3 OR 1=1'])[0], [1])

    def test_object_grants_are_parsed_across_spacing_comments_and_dialects(self):
        for sql in ('GRANT SELECT ON TABLE lab.t TO r;', 'GRANT/*x*/SELECT ON OBJECT::lab.t TO r;', 'GRANT SELECT ON lab.t TO r;'):
            value = observed_action(sql, {})
            self.assertEqual(value['resources'], ['lab.t'])
            self.assertEqual(value['recipient'], 'r')

    def test_policy_result_depends_on_declared_input_not_annotation(self):
        condition = {'resources': ['lab.t'], 'recipient': 'r', 'privileges': ['select'], 'delegable': False}
        item = {'sql': 'GRANT SELECT ON lab.t TO r;', 'context': {
            'application_operation': {'availability': 'known', 'value': {'description': '{"workflow":"delegate_access_v1"}'}},
            'policy': {'availability': 'known', 'value': {'coverage': 'complete', 'scope_resources': ['lab.t'],
                        'rules': [{'effect': 'allow', 'actions': ['grant'], 'resources': ['lab.t'], 'condition': json.dumps(condition)}]}}}}
        self.assertEqual(policy_decision(item)[0], 0)
        changed = deepcopy(item)
        changed['annotations'] = [{'target_label': 1}]
        self.assertEqual(policy_decision(changed)[0], 0)
        changed['context']['policy']['value']['rules'][0]['effect'] = 'deny'
        self.assertEqual(policy_decision(changed)[0], 1)

    def test_missing_policy_or_unsupported_SQL_does_not_become_legit(self):
        predicted, statuses, _ = rule_predictions('policy_only', ['SELECT 1'])
        self.assertEqual(predicted, [None])
        self.assertEqual(statuses, ['not_run'])
        predicted, statuses, reasons = rule_predictions('regex_policy', ['SELECT 1'])
        self.assertEqual(predicted, [0])
        self.assertEqual(statuses, ['ok'])
        self.assertTrue(reasons[0].startswith('regex_fallback:'))

    def test_historical_alarm_is_explicitly_nonzero_class(self):
        sql = ['SELECT id FROM t', 'GRANT SELECT ON lab.t TO r']
        original, _, _ = rule_predictions('regex_v1',sql,legacy=True)
        alarm, _, _ = rule_predictions('regex_v1',sql)
        self.assertEqual(original,[0,3])
        self.assertEqual(alarm,[int(value != 0) for value in original])


if __name__ == '__main__':
    unittest.main()
