"""Conservative missing-answer accounting and uncertainty on a small exploratory sample."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preview_analysis import statistics,wilson


class PreviewMetricsTests(unittest.TestCase):
    def test_missing_answers_are_errors_of_their_true_class(self):
        rows=[{'true_label':truth,'predicted_label':pred,'status':status,'group_id':group} for truth,pred,status,group in (
            (0,0,'ok','a'),(0,None,'api_error','a'),(1,1,'ok','b'),(1,None,'parse_error','b'))]
        result=statistics(rows)
        self.assertIsNone(result['metrics']['primary_f1'])
        self.assertEqual(result['metrics']['decision_coverage'],.5)
        self.assertEqual(result['conservative_all_cases']['successful_decision_metrics']['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(result['conservative_all_cases']['primary_f1'],.5)
        self.assertEqual(result['unique_groups'],2)

    def test_zero_false_alarms_in_fifteen_normals_has_nonzero_uncertainty(self):
        interval=wilson(0,15)
        self.assertAlmostEqual(interval[0],0)
        self.assertGreater(interval[1],.20)
        rows=[{'true_label':0,'predicted_label':0,'status':'ok','group_id':'a'}]
        self.assertIsNone(statistics(rows)['group_bootstrap_F1_95'])


if __name__=='__main__':unittest.main()
