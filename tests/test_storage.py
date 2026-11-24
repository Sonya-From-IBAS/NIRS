"""Real PostgreSQL integration checks; enabled with NIRS_RUN_DB_TESTS=1."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import psycopg
    from psycopg.types.json import Jsonb
    from storage import Storage, file_hash
    from storage_backup import create_scratch_database, drop_scratch_database, restore_check
except ImportError:
    psycopg = None


@unittest.skipUnless(psycopg is not None and os.environ.get("NIRS_RUN_DB_TESTS") == "1",
                     "Enable PostgreSQL integration tests explicitly in the project venv")
class StorageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".local").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".local")
        cls.folder = Path(cls.temporary.name).resolve()
        cls.admin = Storage()
        cls.name = create_scratch_database(cls.admin)
        cls.store = Storage(dbname=cls.name, artifact_root=cls.folder / "artifacts")
        cls.store.migrate()
        cls.records = json.loads((ROOT / "tests/fixtures/contract_examples.json").read_text(encoding="utf-8"))["records"]
        cls.version = cls.store.import_records(cls.records, source_id="contract-test", version="1")
        cls.dataset = cls.store.create_dataset(cls.version, "contract-test")

    @classmethod
    def tearDownClass(cls):
        drop_scratch_database(cls.admin, cls.name)
        assert cls.folder.is_relative_to((ROOT / ".local").resolve())
        cls.temporary.cleanup()

    def csv_file(self, rows):
        path = self.folder / (uuid.uuid4().hex + ".csv")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query", "label", "source", "label_name"])
            writer.writerows(rows)
        return path

    def test_migrations_are_idempotent_and_modified_history_is_rejected(self):
        self.assertEqual(self.store.migrate(), [])
        with tempfile.TemporaryDirectory(dir=self.folder) as temporary:
            root = Path(temporary)
            (root / "migrations").mkdir()
            (root / "migrations/001_initial.sql").write_text("-- changed history\n", encoding="utf-8")
            with patch("storage.ROOT", root), self.assertRaises(ValueError):
                self.store.migrate()

    def test_contract_roundtrip_keeps_context_labels_and_annotation_history(self):
        actual = self.store.dataset_records(self.dataset)
        self.assertEqual(actual, self.records)
        self.assertEqual(actual[5]["sql"]["text"], actual[6]["sql"]["text"])
        self.assertNotEqual(actual[5]["context"], actual[6]["context"])

    def test_sql_is_stored_as_text_with_whitespace_and_not_executed(self):
        query = "  SELECT 'a';\r\nDROP SCHEMA nirs CASCADE; -- preserved as data  "
        path = self.csv_file([[query, 1, "test", "sqli"]])
        result = self.store.import_csv(path)
        with self.store.connect() as connection:
            row = connection.execute("SELECT o.sql_text FROM nirs.occurrences c JOIN nirs.observations o USING(observation_id) WHERE c.source_version_id=%s", (result["source_version_id"],)).fetchone()
            self.assertEqual(row["sql_text"], query)
            self.assertIsNotNone(connection.execute("SELECT to_regclass('nirs.annotations') AS name").fetchone()["name"])

    def test_reimport_is_idempotent_and_conflicting_source_labels_are_preserved(self):
        path = self.csv_file([["SELECT 1", 0, "test", "legit"], ["SELECT 1", 3, "test", "privilege"]])
        first = self.store.import_csv(path)
        second = self.store.import_csv(path)
        self.assertTrue(second["reused"])
        self.assertEqual(first["source_version_id"], second["source_version_id"])
        with self.store.connect() as connection:
            rows = connection.execute("SELECT source_label FROM nirs.occurrences WHERE source_version_id=%s ORDER BY ordinal", (first["source_version_id"],)).fetchall()
            self.assertEqual([row["source_label"]["label"] for row in rows], ["0", "3"])

    def test_internal_import_failure_rolls_back_partial_batches(self):
        path = self.csv_file([["SELECT 1", 0, "test", "legit"], ["SELECT 2", 0, "test", "legit"]])
        source_id = "rollback-" + uuid.uuid4().hex
        original = self.store._store_batch
        def fail_after_write(*args):
            original(*args)
            raise RuntimeError("Injected integration-test failure")
        with patch.object(self.store, "_store_batch", side_effect=fail_after_write), self.assertRaises(RuntimeError):
            self.store.import_csv(path, source_id=source_id, batch_size=1)
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) AS n FROM nirs.source_versions WHERE source_id=%s", (source_id,)).fetchone()["n"], 0)
            self.assertEqual(connection.execute("SELECT status FROM nirs.import_jobs WHERE source_id=%s", (source_id,)).fetchone()["status"], "failed")

    def test_rejected_rows_are_counted_without_losing_good_rows(self):
        path = self.csv_file([["SELECT 1", 0, "test", "legit"], ["bad\x00text", 1, "test", "sqli"], ["SELECT 2", 0, "test", "legit", "extra"]])
        result = self.store.import_csv(path)
        self.assertEqual((result["rows"], result["accepted"], result["rejected"]), (3, 1, 2))
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) AS n FROM nirs.rejected_records WHERE source_version_id=%s", (result["source_version_id"],)).fetchone()["n"], 2)

    def test_frozen_dataset_members_and_annotations_cannot_be_overwritten(self):
        with self.assertRaises(psycopg.errors.RaiseException):
            with self.store.write() as connection:
                connection.execute("DELETE FROM nirs.dataset_members WHERE dataset_version_id=%s", (self.dataset,))
        with self.assertRaises(psycopg.errors.RaiseException):
            with self.store.write() as connection:
                connection.execute("UPDATE nirs.annotations SET target_label=1 WHERE annotation_id='obs-001-ann-1'")

    def test_new_annotation_does_not_change_a_frozen_snapshot(self):
        annotation = deepcopy(self.records[0]["annotations"][0])
        annotation.update({"annotation_id": "obs-001-ann-2", "revision": 2, "supersedes": "obs-001-ann-1",
            "status": "needs_review", "target_label": None, "reason": "Additional review requested in this test."})
        self.store.append_annotation("obs-001", annotation)
        with self.store.connect() as connection:
            row = connection.execute("SELECT current_annotation_id FROM nirs.observations WHERE observation_id='obs-001'").fetchone()
            self.assertEqual(row["current_annotation_id"], "obs-001-ann-2")
        self.assertEqual(self.store.dataset_records(self.dataset)[0]["current_annotation_id"], "obs-001-ann-1")

    def test_verified_annotation_cannot_have_a_null_target_in_postgresql(self):
        annotation = deepcopy(self.records[14]["annotations"][0])
        annotation.update({"annotation_id": "invalid-null", "revision": 99, "status": "verified"})
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.store.write() as connection:
                connection.execute("INSERT INTO nirs.annotations VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    ("invalid-null", "obs-015", 99, None, "verified", None, Jsonb(annotation)))

    def test_artifact_registration_is_idempotent_and_detects_corruption(self):
        source = self.folder / "artifact.bin"
        source.write_bytes(b"artifact-content")
        artifact = self.store.register_artifact(source)
        self.assertEqual(artifact, self.store.register_artifact(source))
        target = self.store.get_artifact(artifact)
        target.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.store.get_artifact(artifact)
        self.assertIn(artifact, self.store.verify_artifacts()["failures"])
        target.write_bytes(source.read_bytes())

    def test_failed_file_write_remains_pending_and_unusable(self):
        source = self.folder / "unwritten.bin"
        source.write_bytes(b"pending-file-test")
        with patch("storage.shutil.copyfile", side_effect=OSError("Injected write failure")), self.assertRaises(OSError):
            self.store.register_artifact(source)
        with self.assertRaises(ValueError):
            self.store.get_artifact(file_hash(source))
        self.assertGreaterEqual(self.store.verify_artifacts()["not_ready"], 1)

    def test_sparse_matrix_can_be_registered_with_row_metadata(self):
        import numpy as np
        from scipy.sparse import csr_matrix, save_npz, load_npz
        matrix = csr_matrix([[1, 0], [0, 2]])
        path = self.folder / "matrix.npz"
        save_npz(path, matrix)
        artifact = self.store.register_artifact(path)
        with self.store.write() as connection:
            connection.execute("INSERT INTO nirs.representations VALUES (%s,%s,%s)",
                               ("matrix-test", self.dataset, Jsonb({"test_only": True})))
            connection.execute("INSERT INTO nirs.representation_artifacts VALUES (%s,%s,%s,%s)",
                ("matrix-test", artifact, "matrix", Jsonb({"row_ids": ["obs-001", "obs-002"], "shape": [2, 2], "nnz": 2, "format": "csr"})))
        np.testing.assert_array_equal(load_npz(self.store.get_artifact(artifact)).toarray(), matrix.toarray())

    def test_artifact_path_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            self.store.artifact_path("../outside.bin")

    def test_prediction_truth_is_tied_to_the_pinned_annotation_and_test_split(self):
        with self.store.write() as connection:
            connection.execute("INSERT INTO nirs.splits VALUES ('prediction-test',%s,'{}')", (self.dataset,))
            connection.execute("INSERT INTO nirs.split_members VALUES ('prediction-test',%s,'obs-001',0,'test')", (self.dataset,))
            connection.execute("INSERT INTO nirs.experiment_runs VALUES ('prediction-test',%s,'test','binary','prepared','{}')", (self.dataset,))
        statement = "INSERT INTO nirs.predictions VALUES ('prediction-test',%s,'prediction-test','obs-001','holdout',0,%s,%s,'ok','{}')"
        with self.assertRaises(psycopg.errors.RaiseException):
            with self.store.write() as connection:
                connection.execute(statement, (self.dataset, 1, 0))
        with self.assertRaises(psycopg.errors.RaiseException):
            with self.store.write() as connection:
                connection.execute(statement, (self.dataset, 0, 2))
        with self.store.write() as connection:
            connection.execute(statement, (self.dataset, 0, 0))

    def test_restore_rejects_a_changed_dump_before_creating_a_database(self):
        directory = self.folder / "bad-backup"
        directory.mkdir()
        (directory / "database.dump").write_bytes(b"changed")
        (directory / "manifest.json").write_text(json.dumps({"dump_sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaises(ValueError):
            restore_check(self.store, directory, self.folder / "bad-restore")
        with self.assertRaises(ValueError):
            drop_scratch_database(self.admin, self.admin.config["dbname"])

    def test_subset_preserves_pinned_labels_and_rejects_foreign_ids(self):
        subset = self.store.create_subset(self.dataset, ["obs-001", "obs-002"], "test-subset", {"test_only": True})
        self.assertEqual([row['observation_id'] for row in self.store.dataset_records(subset)], ["obs-001", "obs-002"])
        with self.assertRaises(ValueError):
            self.store.create_subset(self.dataset, ["outside"], "bad-subset", {})
        with self.assertRaises(ValueError):
            self.store.create_subset(self.dataset, ["obs-001", "obs-001"], "bad-subset", {})

    def test_dataset_loader_uses_source_reference_without_forcing_manual_relabeling(self):
        from dataset import load_source_reference, load_verified
        path = self.csv_file([["SELECT 1", 0, "test", "legit"], ["GRANT SELECT ON t TO x", 3, "test", "privilege"]])
        source = self.store.import_csv(path)
        version = self.store.create_dataset(source['source_version_id'], "source-reference-test")
        reference = load_source_reference(version, self.store)
        self.assertEqual(reference['label'].tolist(), [0, 3])
        self.assertTrue((reference['label_basis'] == 'original_source_reference').all())
        self.assertEqual(len(load_verified(version, self.store)), 0)


if __name__ == "__main__":
    unittest.main()
