"""Policy relationships use only available input and train-fitted statistics."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from relational_features import RelationalFeatureEncoder, policy_relations
from sql_features import prepare_input


def observation(resource='orders', before=True):
    return {'sql': {'text': 'SELECT id FROM orders WHERE tenant_id=7'}, 'context': {
        'policy': {'availability': 'known', 'available_before_decision': before,
                   'value': {'rules': [{'effect': 'allow', 'actions': ['select'],
                                        'resources': [resource], 'condition': json.dumps({'tenant': 7})}]}}}}


class RelationalFeatureTests(unittest.TestCase):
    def test_relations_follow_policy_resource_and_predecision_availability(self):
        first = policy_relations(prepare_input(observation()))
        second = policy_relations(prepare_input(observation('other_table')))
        self.assertGreater(first['policy_allow_resource_mentions'], second['policy_allow_resource_mentions'])
        self.assertGreater(first['policy_allow_condition_tenant_mentions'], 0)
        self.assertEqual(policy_relations(prepare_input(observation(before=False))), {})

    def test_retrospective_labels_and_source_do_not_affect_relations(self):
        original = observation()
        changed = deepcopy(original)
        changed.update(annotations=[{'target_label': 1}], source={'source_label': 'attack'})
        self.assertEqual(policy_relations(prepare_input(original)), policy_relations(prepare_input(changed)))

    def test_unknown_context_does_not_break_sql_branch_and_empty_only_branch_is_explicit(self):
        encoder = RelationalFeatureEncoder(text_mode='masked', structure=False)
        matrix = encoder.fit_transform(['SELECT a FROM t', 'DELETE FROM t'], [0, 1])
        self.assertGreater(matrix.shape[1], 0)
        self.assertEqual(encoder.transform(['SELECT b FROM u']).shape[1], matrix.shape[1])
        with self.assertRaisesRegex(ValueError, 'No relational features'):
            RelationalFeatureEncoder(text_mode=None, structure=False).fit(['SELECT a FROM t'])

    def test_validation_cannot_extend_relation_vocabulary_or_scaler(self):
        encoder = RelationalFeatureEncoder(text_mode=None, structure=False)
        encoder.fit([observation(), observation('other')], [0, 1])
        names = encoder.get_feature_names_out()
        scale = encoder.relations_scaler_.max_abs_.copy()
        changed = observation()
        changed['context']['policy']['value']['rules'][0]['effect'] = 'deny'
        encoder.transform([changed])
        np.testing.assert_array_equal(names, encoder.get_feature_names_out())
        np.testing.assert_array_equal(scale, encoder.relations_scaler_.max_abs_)


if __name__ == '__main__':
    unittest.main()
