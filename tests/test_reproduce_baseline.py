"""Checks for errors that could silently change the stage-1 scientific control."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reproduce_baseline as control
from baseline import ML_MODELS


class HistoricalControlTests(unittest.TestCase):
    def test_newline_hash_preserves_content_and_reports_distinct_byte_hashes(self):
        with tempfile.TemporaryDirectory(dir=control.ROOT) as tmp:
            path = Path(tmp) / "input.csv"
            path.write_bytes(b'query,label\r\n"SELECT 1\r\nFROM t",0\r\n')
            windows = control.file_manifest(path)
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            unix = control.file_manifest(path)
            self.assertNotEqual(windows["sha256_bytes"], unix["sha256_bytes"])
            self.assertEqual(windows["sha256_crlf_to_lf"], unix["sha256_bytes"])
            self.assertEqual(windows["rows"], 1)
            path.write_bytes(path.read_bytes().replace(b"SELECT 1", b"SELECT 2"))
            self.assertNotEqual(control.file_manifest(path)["sha256_crlf_to_lf"],
                                windows["sha256_crlf_to_lf"])

    def test_ids_remain_csv_record_positions_after_historical_dropna(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/processed").mkdir(parents=True)
            (root / "data/processed/dataset.csv").write_text(
                'query,label,source,label_name\n"SELECT 1",0,x,legit\n'
                ',1,x,sqli\n"  ",0,x,legit\n', encoding="utf-8",
            )
            frame = control.dataset(root)
            self.assertEqual(frame["row_id"].tolist(), [0, 2])
            self.assertEqual(frame["query"].tolist(), ["SELECT 1", "  "])

    def test_original_holdout_and_cv_order_are_preserved(self):
        frame = pd.DataFrame({"label": np.repeat(np.arange(5), 20)})
        actual = control.splits(frame)
        original = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for (_, _, train, test), (old_train, old_test) in zip(actual, original.split(frame, frame["label"])):
            np.testing.assert_array_equal(train, old_train)
            np.testing.assert_array_equal(test, old_test)
            self.assertEqual(len(set(train) & set(test)), 0)
        old_train, old_test, _, _ = train_test_split(
            list(range(len(frame))), frame["label"].to_numpy(),
            test_size=0.2, random_state=42, stratify=frame["label"],
        )
        np.testing.assert_array_equal(actual[-1][2], old_train)
        np.testing.assert_array_equal(actual[-1][3], old_test)

    def test_each_model_is_independent_and_only_reproducibility_parameters_change(self):
        for name, original in ML_MODELS.items():
            first, second = control.new_model(name), control.new_model(name)
            self.assertIsNot(first, second)
            for step in original.named_steps:
                self.assertIsNot(first.named_steps[step], second.named_steps[step])
            for key, value in original.get_params().items():
                if "__" not in key:
                    continue
                current = first.get_params()[key]
                if key.endswith("random_state"):
                    self.assertEqual(current, 42)
                elif key.endswith("n_jobs"):
                    self.assertEqual(current, 1)
                else:
                    self.assertEqual(current, value)
            self.assertIsNone(original.named_steps["clf"].random_state)

    def test_aggregate_f1_is_recomputed_from_predictions_not_averaged_across_rows(self):
        y_true = [0, 0, 0, 1, 1, 2, 3, 4]
        y_pred = [0, 0, 1, 1, 0, 2, 4, 4]
        blocks = []
        for protocol, fold in [("holdout", 0), ("cv", 0), ("cv", 1)]:
            blocks.append(pd.DataFrame({"protocol": protocol, "fold": fold,
                                        "method": "Regex", "y_true": y_true, "y_pred": y_pred}))
        holdout, cv, _ = control.derived_tables(pd.concat(blocks))
        self.assertAlmostEqual(holdout.iloc[0]["f1_macro"], f1_score(y_true, y_pred, average="macro"))
        self.assertAlmostEqual(holdout.iloc[0]["f1_weighted"], f1_score(y_true, y_pred, average="weighted"))
        self.assertAlmostEqual(cv.iloc[0]["cv_f1_macro_mean"], holdout.iloc[0]["f1_macro"])
        self.assertEqual(cv.iloc[0]["cv_f1_macro_std"], 0)


if __name__ == "__main__":
    unittest.main()
