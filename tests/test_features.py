"""Feature fidelity, train-only fitting, context isolation and cache roundtrips."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sql_views import lex, views
from sql_features import SQLFeatureEncoder, prepare_input, project_input
from study_features import renamed_input, reformatted_input
from normalize import normalize_sql, sql_views


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(line) for line in (ROOT / "data/scenarios/corpus_v1.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_lexer_and_raw_view_preserve_all_bytes_and_unicode(self):
        query = "  SELECT имя, '/* literal */' FROM таблица\r\n/* note */ WHERE n=1;  "
        self.assertEqual("".join(t.text for t in lex(query)), query)
        self.assertEqual(sql_views(query)["raw"], query)
        self.assertNotIn("таблица", views(query)["masked"])
        self.assertEqual(views(query)["structure"]["block_comment_count"], math.log1p(1))

    def test_equal_and_different_literals_remain_distinguishable(self):
        equal = views("SELECT x FROM a WHERE 1=1")
        different = views("SELECT x FROM a WHERE 1=2")
        self.assertNotEqual(equal["masked"], different["masked"])
        self.assertEqual(equal["structure"]["comparison_same_literal"], math.log1p(1))
        self.assertEqual(different["structure"]["comparison_different_literal"], math.log1p(1))
        self.assertEqual(equal["masked"], views("SELECT renamed FROM other WHERE 9=9")["masked"])

    def test_comments_encoding_and_executable_payload_are_retained_as_signals(self):
        query = "SELECT CHAR(65),0x4142 /* keep */ /*! UNION SELECT 2 */;"
        item = views(query)
        self.assertIn("EXEC_COMMENT", item["masked"])
        self.assertIn("UNION", item["masked"])
        self.assertGreater(item["structure"]["char_function_markers_raw"], 0)
        self.assertGreater(item["structure"]["hex_markers_raw"], 0)
        self.assertGreater(item["structure"]["block_comment_count"], 0)

    def test_dollar_body_and_fixed_dynamic_text_hint_preserve_operation(self):
        for tag in ("", "procedure_body"):
            query = f"DO ${tag}$ BEGIN EXECUTE 'GR' || 'ANT SELECT ON lab.table_a TO receiver'; END ${tag}$;"
            self.assertEqual("".join(t.text for t in lex(query)), query)
            masked = views(query)["masked"]
            self.assertIn("DOLLAR_BODY", masked)
            self.assertIn("FIXED_SQL_TEXT_HINT GRANT SELECT ON", masked)

    def test_character_constructors_share_numeric_symbols_with_context(self):
        item = prepare_input({'sql': 'SELECT x FROM t WHERE tenant = CAST(CHR(49) || CHR(49) AS integer);',
                              'context': {'application_operation': {'availability': 'known', 'value': {
                                  'description': '{"tenant":11}'}}}})
        numeric_symbol = item.sql['symbols'].literal(11, 'NUM')
        self.assertIn('SCALAR_DIGITS_HINT ' + numeric_symbol, item.sql['masked'])
        self.assertIn('OP_TENANT=' + numeric_symbol, item.context_text)

    def test_where_parts_keep_prefix_suffix_nested_where_and_non_select(self):
        item = views("SELECT id FROM t WHERE id=1 UNION SELECT value FROM other;")
        self.assertIn("SEG_TAIL UNION SELECT", item["parts"])
        self.assertIn("GRANT", views("GRANT SELECT ON x TO y;")["parts"])
        self.assertIn("SEG_NO_OUTER_WHERE", views("SELECT * FROM (SELECT * FROM t WHERE id=1) s;")["parts"])

    def test_quoted_identifiers_keep_identity_and_case(self):
        self.assertEqual(views('SELECT "a" FROM "a"')["masked"], 'SELECT Q_ID0 FROM Q_ID0')
        self.assertEqual(views('SELECT "a" FROM "A"')["masked"], 'SELECT Q_ID0 FROM Q_ID1')
        self.assertIn('Q_ID0', views('SELECT [a]]b] FROM [a]]b]')["masked"])

    def test_projection_ignores_source_annotation_and_retrospective_metadata(self):
        first = deepcopy(self.records[0])
        second = deepcopy(first)
        second['source']['source_label'] = {'class': 'attack', 'generator': 'hidden'}
        second['annotations'][0]['target_label'] = 1
        second['grouping']['scenario_id'] = 'hidden-test-id'
        second['context']['policy']['value']['policy_id'] = 'hidden-class-label'
        second['context']['policy']['value']['version'] = 'new'
        second['context']['policy']['evidence_ref'] = 'retrospective'
        self.assertEqual(prepare_input(first).context_text, prepare_input(second).context_text)
        encoder = SQLFeatureEncoder(context=True).fit([first, self.records[2]], [0, 1])
        self.assertEqual((encoder.transform([first]) != encoder.transform([second])).nnz, 0)

    def test_postdecision_context_is_not_available_to_features(self):
        item = deepcopy(self.records[0])
        item['context']['policy']['available_before_decision'] = False
        projected = project_input(item)
        self.assertEqual(projected['context']['policy'], {'availability': 'unknown', 'value': None})

    def test_consistent_sql_and_policy_renaming_preserves_masked_representation(self):
        # Exercise multiple families, all dialects and raw/obfuscated parents.
        for record in self.records[::17]:
            original = prepare_input(record)
            renamed = renamed_input(original)
            self.assertEqual(original.sql['masked'], renamed.sql['masked'], record['observation_id'])
            self.assertEqual(original.context_text, renamed.context_text, record['observation_id'])

    def test_allowed_keyword_case_and_whitespace_preserve_masked_tokens(self):
        original = prepare_input('SELECT alias.id FROM orders alias WHERE alias.id=1;')
        self.assertEqual(original.sql['masked'], reformatted_input(original).sql['masked'])

    def test_fit_statistics_vocabulary_and_selection_do_not_change_on_validation(self):
        train = ['SELECT id FROM orders WHERE id=1', 'GRANT SELECT ON users TO receiver',
                 'SELECT id FROM orders WHERE id=2', 'GRANT UPDATE ON users TO receiver']
        encoder = SQLFeatureEncoder(text_mode='raw', context=True, select_k=30)
        encoder.fit(train, [0, 1, 0, 1])
        vocabulary = encoder.text_vectorizer_.vocabulary_.copy()
        idf = encoder.text_vectorizer_.idf_.copy()
        scale = encoder.structure_scaler_.max_abs_.copy()
        selected = encoder.selector_.get_support().copy()
        encoder.transform(['SELECT zqxyz_unique_validation_only FROM an_unseen_table ' * 25])
        self.assertEqual(vocabulary, encoder.text_vectorizer_.vocabulary_)
        np.testing.assert_array_equal(idf, encoder.text_vectorizer_.idf_)
        np.testing.assert_array_equal(scale, encoder.structure_scaler_.max_abs_)
        np.testing.assert_array_equal(selected, encoder.selector_.get_support())
        self.assertNotIn('zqx', vocabulary)
        self.assertEqual(encoder.fit_rows_, 4)

    def test_sparse_row_column_and_pipeline_roundtrip(self):
        inputs = [self.records[i] for i in (0, 2, 4, 6)]
        targets = [r['annotations'][0]['target_label'] for r in inputs]
        encoder = SQLFeatureEncoder(context=True, select_k=50)
        matrix = encoder.fit_transform(inputs, targets)
        self.assertTrue(sparse.isspmatrix_csr(matrix))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape[1], len(encoder.get_feature_names_out()))
        model = LogisticRegression(solver='liblinear', random_state=1).fit(matrix, targets)
        pipeline = Pipeline([('features', encoder), ('classifier', model)])
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / 'pipeline.joblib'
            matrix_path = Path(temporary) / 'matrix.npz'
            joblib.dump(pipeline, model_path)
            sparse.save_npz(matrix_path, matrix)
            restored = joblib.load(model_path)
            np.testing.assert_array_equal(pipeline.predict(inputs), restored.predict(inputs))
            self.assertEqual((matrix != sparse.load_npz(matrix_path)).nnz, 0)
            self.assertEqual((matrix != restored.named_steps['features'].transform(inputs)).nnz, 0)

    def test_historical_normalizer_behavior_stays_available(self):
        self.assertEqual(normalize_sql('SELECT CHAR(65) /* note */'), 'SELECT A')
        self.assertIn('COMMENT', sql_views('SELECT CHAR(65) /* note */')['masked'])


if __name__ == '__main__':
    unittest.main()
