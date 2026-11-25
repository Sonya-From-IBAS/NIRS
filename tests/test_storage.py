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

    def test_frozen_split_is_idempotent_and_rejects_mutations(self):
        ids = [row['observation_id'] for row in self.store.dataset_records(self.dataset)]
        assignments = {0: {oid: 'test' if index < 2 else 'train' for index, oid in enumerate(ids)}}
        spec = {'unit_test': 'frozen-split'}
        saved = self.store.save_split(self.dataset, spec, assignments)
        self.assertFalse(saved['reused'])
        self.assertTrue(self.store.save_split(self.dataset, spec, assignments)['reused'])
        self.assertEqual(self.store.split_ids(saved['split_id'], 0, 'test'), ids[:2])
        for query, parameters in (
            ("UPDATE nirs.splits SET specification='{}' WHERE split_id=%s", (saved['split_id'],)),
            ("DELETE FROM nirs.split_members WHERE split_id=%s", (saved['split_id'],)),
            ("INSERT INTO nirs.split_members VALUES (%s,%s,%s,99,'test')", (saved['split_id'], self.dataset, ids[0]))):
            with self.subTest(query=query), self.assertRaises(psycopg.errors.RaiseException), self.store.write() as connection:
                connection.execute(query, parameters)

    def test_inner_split_scope_cannot_include_outer_test(self):
        ids = [row['observation_id'] for row in self.store.dataset_records(self.dataset)]
        outer = self.store.save_split(self.dataset, {'unit_test': 'parent-scope'},
                                     {0: {oid: 'test' if index < 2 else 'train' for index, oid in enumerate(ids)}})
        spec = {'parent_scope': {'split_id': outer['split_id'], 'fold': 0, 'part': 'train'}}
        members = {oid: 'validation' if index < 2 else 'train' for index, oid in enumerate(ids[2:])}
        inner = self.store.save_split(self.dataset, spec, {0: members})
        self.assertEqual(set(self.store.split_ids(inner['split_id'], 0, 'validation')), set(ids[2:4]))
        with self.assertRaises(ValueError):
            self.store.save_split(self.dataset, spec, {0: {**members, ids[0]: 'validation'}})

    def test_sparse_representation_cache_reuses_fitted_pipeline_and_row_order(self):
        import numpy as np
        from sql_features import prepare_input
        from study_features import cache_representation
        ids = [row['observation_id'] for row in self.store.dataset_records(self.dataset)]
        saved = self.store.save_split(self.dataset, {'unit_test': 'representation-cache'},
            {0: {oid: 'train' if index < 6 else 'validation' if index < 8 else 'test' for index, oid in enumerate(ids)}})
        entry = {'split_id': saved['split_id'], 'dataset_version_id': self.dataset}
        by_id = {row['observation_id']: row for row in self.records}
        train = [prepare_input(by_id[oid]) for oid in ids[:6]]
        validation = [prepare_input(by_id[oid]) for oid in ids[6:8]]
        y_train = np.array([by_id[oid]['annotations'][0]['target_label'] for oid in ids[:6]])
        y_validation = np.array([by_id[oid]['annotations'][0]['target_label'] for oid in ids[6:8]])
        with patch('study_features.LOCAL', self.folder / 'feature-cache'):
            first = cache_representation(self.store, entry, 'unit-feature-cache',
                {'text_mode': 'masked', 'structure': True, 'context': True, 'select_k': 50},
                ids[:6], ids[6:8], train, validation, y_train, y_validation, 'controlled', {})
            second = cache_representation(self.store, entry, 'unit-feature-cache',
                {'text_mode': 'masked', 'structure': True, 'context': True, 'select_k': 50},
                ids[:6], ids[6:8], train, validation, y_train, y_validation, 'controlled', {})
        self.assertEqual(first['representation_id'], second['representation_id'])
        self.assertEqual(first['artifacts'], second['artifacts'])
        self.assertEqual(second['fit_rows_in_encoder'], 6)
        with self.store.connect() as connection:
            matrix = connection.execute("SELECT metadata FROM nirs.representation_artifacts WHERE representation_id=%s AND role='validation.npz'", (first['representation_id'],)).fetchone()['metadata']
        self.assertEqual(matrix['shape'][0], 2)
        self.assertEqual(matrix['encoder_artifact'], first['artifacts']['encoder.joblib'])

    def test_inner_validation_storage_is_pinned_and_cannot_masquerade_as_test(self):
        from experiment_store import persist_run
        ids = [row['observation_id'] for row in self.store.dataset_records(self.dataset)]
        saved = self.store.save_split(self.dataset, {'unit_test': 'stage8-validation'},
            {0: {oid: 'validation' if oid == 'obs-001' else 'train' for oid in ids}})
        entry = {'split_id': saved['split_id'], 'dataset_version_id': self.dataset}
        config = {'role': 'inner_validation_rule_development', 'split_id': saved['split_id']}
        rows = [{'fold': 0, 'observation_id': 'obs-001', 'truth': 0, 'prediction': 0, 'status': 'ok', 'metadata': {}}]
        run_id, _ = persist_run(self.store, entry, 'stage8-storage-test', 'binary', config, rows, {}, {})
        again, _ = persist_run(self.store, entry, 'stage8-storage-test', 'binary', config, rows, {}, {})
        self.assertEqual(run_id, again)
        with self.assertRaises(ValueError):
            persist_run(self.store, entry, 'stage8-storage-test', 'binary', config, [dict(rows[0], truth=1)], {}, {})
        with self.assertRaises(psycopg.errors.RaiseException), self.store.write() as connection:
            connection.execute("INSERT INTO nirs.predictions VALUES (%s,%s,%s,'obs-001','holdout',0,0,0,'ok','{}')",
                               (run_id, self.dataset, saved['split_id']))
        with self.assertRaises(psycopg.errors.RaiseException), self.store.write() as connection:
            connection.execute("INSERT INTO nirs.experiment_runs VALUES ('unbound-validation',%s,'test','binary','prepared','{}')", (self.dataset,))
            connection.execute("INSERT INTO nirs.predictions VALUES ('unbound-validation',%s,%s,'obs-001','inner_validation',0,0,0,'ok','{}')",
                               (self.dataset,saved['split_id']))

    def test_stage8_selection_refits_reload_and_resume_without_outer_test(self):
        import experiment_runner as runner
        from ml_models import classifier_candidates
        ids = [row['observation_id'] for row in self.store.dataset_records(self.dataset)]
        parent = self.store.save_split(self.dataset, {'unit_test': 'stage8-parent'},
            {0: {oid: 'train' if index < 8 else 'test' for index,oid in enumerate(ids)}})
        parent_scope = {'split_id':parent['split_id'], 'fold':0, 'part':'train'}
        inner = self.store.save_split(self.dataset, {'parent_scope':parent_scope},
            {fold: {oid:'validation' if (index//4)==fold else 'train' for index,oid in enumerate(ids[:8])} for fold in (0,1)})
        entry = {'split_id':inner['split_id'], 'dataset_version_id':self.dataset,
                 'cohort':'controlled','parent_scope':parent_scope,'folds':{'0':{},'1':{}}}
        root = self.folder/'stage8-e2e'
        (root/'results/stage6').mkdir(parents=True)
        (root/'results/stage6/split_registry.json').write_text(json.dumps({'protocols':{'unit-inner':entry}}), encoding='utf-8')
        features = [{'name':'masked','input_mode':'sql_only','relations':False,
                     'parameters':{'text_mode':'masked','structure':True}}]
        def small_grid(method):
            values = classifier_candidates(method)[0]
            if method == 'RandomForest':
                values = dict(values,n_estimators=5)
            return [values]
        with patch('experiment_runner.ROOT',root), patch('experiment_runner.OUTPUT',root/'results/stage8'), \
             patch('experiment_runner.LOCAL',root/'local'), patch('experiment_store.LOCAL',root/'local'), \
             patch('experiment_runner.code_signature',return_value={}), \
             patch('experiment_runner.feature_candidates',return_value=features), \
             patch('experiment_runner.classifier_candidates',side_effect=small_grid):
            first = runner.run_scope(self.store,'unit-inner',{}, {})
            with patch('experiment_runner.new_classifier', side_effect=AssertionError('Resume must not fit again')):
                second = runner.run_scope(self.store,'unit-inner',{}, {})
        self.assertEqual([x['run_id'] for x in first['selected_models']], [x['run_id'] for x in second['selected_models']])
        self.assertEqual(len(first['selected_models']),3)
        self.assertFalse(first['outer_test_used'])
        for model in first['selected_models']:
            with self.store.connect() as connection:
                rows = connection.execute('SELECT observation_id FROM nirs.predictions WHERE run_id=%s',(model['run_id'],)).fetchall()
            self.assertEqual({row['observation_id'] for row in rows},set(ids[:8]))

    def test_LLM_cache_budget_and_finalized_response_are_enforced_in_PostgreSQL(self):
        from detector_llm import MODELS,build_specification,prompt_observation
        from llm_store import prepare_pilot,reserve_request,finish_request
        prepare_pilot(self.store,'unit-llm-budget',{'test':True},1)
        spec=build_specification(MODELS[0],'zero_shot',[prompt_observation('q001','SELECT 1')])
        with patch('llm_store.LOCAL',self.folder/'llm-budget'):
            key,result,new=reserve_request(self.store,'unit-llm-budget',spec)
            self.assertTrue(new);self.assertIsNone(result)
            finish_request(self.store,key,{'status':'ok','labels':[0],'usage':None})
            repeated,cached,new=reserve_request(self.store,'unit-llm-budget',spec)
            self.assertEqual(repeated,key);self.assertFalse(new);self.assertEqual(cached['labels'],[0])
            changed=build_specification(MODELS[0],'zero_shot',[prompt_observation('q001','SELECT 2')])
            with self.assertRaises(psycopg.errors.RaiseException):
                reserve_request(self.store,'unit-llm-budget',changed)
        with self.assertRaises(psycopg.errors.RaiseException),self.store.write() as connection:
            connection.execute('DELETE FROM nirs.llm_requests WHERE request_key=%s',(key,))

    def test_stage10_final_test_rejects_train_rows_and_changed_replayed_predictions(self):
        from stage10_store import load_test,persist_test
        verified=[row for row in self.records if row['annotations'][-1]['status']=='verified']
        ids=[row['observation_id'] for row in verified]
        all_ids=[row['observation_id'] for row in self.records]
        selected=set(ids[-2:])
        split=self.store.save_split(self.dataset,{'unit_test':'stage10-final'},
            {0:{oid:'test' if oid in selected else 'train' for oid in all_ids}})
        records=load_test(self.store,self.dataset,split['split_id'],0,sorted(selected))
        rows=[{'observation_id':row['observation_id'],'split_id':split['split_id'],'fold':0,
               'truth':row['truth'],'prediction':row['truth'],'status':'ok'} for row in records]
        config={'stage':10,'role':'final_test'}
        first=persist_test(self.store,self.dataset,'stage10-test',config,rows,{})
        self.assertEqual(persist_test(self.store,self.dataset,'stage10-test',config,rows,{})['run_id'],first['run_id'])
        changed=deepcopy(rows);changed[0]['prediction']=1-changed[0]['truth']
        with self.assertRaises(ValueError):
            persist_test(self.store,self.dataset,'stage10-test',config,changed,{})
        with self.assertRaises(ValueError):
            load_test(self.store,self.dataset,split['split_id'],0,[ids[0]])

    def test_stage10_frozen_fit_outer_prediction_and_resume_do_not_refit_or_predict_again(self):
        import stage10_evaluate as evaluation
        from experiment_store import load_inputs
        verified=[row['observation_id'] for row in self.records if row['annotations'][-1]['status']=='verified']
        test_ids=set(verified[-2:])
        subset=self.store.create_subset(self.dataset,verified,'stage10 verified fixture',{'unit_test':'verified-fixture'})
        split=self.store.save_split(subset,{'unit_test':'stage10-model-roundtrip'},
            {0:{oid:'test' if oid in test_ids else 'train' for oid in verified}})
        entry={'dataset_version_id':subset,'split_id':split['split_id'],'cohort':'controlled',
            'parent_scope':{'split_id':split['split_id'],'fold':0,'part':'train'}}
        ids,inputs,labels=load_inputs(self.store,entry)
        selection={'candidate':{'method':'LogisticRegression','parameters':{'C':1.,'class_weight':'balanced'},
            'feature':{'name':'raw','input_mode':'sql_only','relations':False,'parameters':{'text_mode':'raw','structure':False,'max_features':1000}}},'threshold':0.}
        folder=self.folder/'stage10-roundtrip';folder.mkdir()
        (folder/'evaluation_plan.json').write_text('{}',encoding='utf-8')
        with patch('stage10_evaluate.LOCAL',folder/'local'),patch('stage10_evaluate.OUTPUT',folder), \
             patch('experiment_store.LOCAL',folder/'matrices'):
            model,_=evaluation.fit_frozen(self.store,entry,selection,ids,inputs,labels,evaluation.signature())
            first=evaluation.evaluate_models(self.store,entry,[model],'unit','test',evaluation.signature())
            with patch.object(model['estimator'],'predict',side_effect=AssertionError('Resume must not predict again')), \
                 patch('stage10_evaluate.new_classifier',side_effect=AssertionError('Resume must not fit again')):
                again=evaluation.evaluate_models(self.store,entry,[model],'unit','test',evaluation.signature())
            self.assertEqual(first,again)
            self.assertEqual(first[0]['metrics']['observations'],2)

    def test_LLM_interrupted_request_is_never_automatically_reissued(self):
        from detector_llm import MODELS,build_specification,prompt_observation
        from llm_store import prepare_pilot,reserve_request
        prepare_pilot(self.store,'unit-llm-interrupted',{},1)
        spec=build_specification(MODELS[1],'zero_shot',[prompt_observation('q001','SELECT 3')])
        with patch('llm_store.LOCAL',self.folder/'llm-interrupted'):
            reserve_request(self.store,'unit-llm-interrupted',spec)
            key,result,new=reserve_request(self.store,'unit-llm-interrupted',spec)
            self.assertFalse(new);self.assertEqual(result['status'],'api_error');self.assertIsNone(result['labels'])
            _,_,new=reserve_request(self.store,'unit-llm-interrupted',spec)
            self.assertFalse(new)
        with self.store.connect() as connection:
            row=connection.execute('SELECT state,api_attempts FROM nirs.llm_requests WHERE request_key=%s',(key,)).fetchone()
        self.assertEqual(row,{'state':'interrupted','api_attempts':1})

    def test_LLM_failed_model_stops_while_other_model_continues_and_resume_uses_cache(self):
        from detector_llm import MODELS,build_specification,prompt_observation,request_key
        from llm_pilot import request_results
        from llm_store import prepare_pilot
        from unittest.mock import Mock
        prepare_pilot(self.store,'unit-llm-stops',{},2)
        requests=[]
        for model,sql in ((MODELS[0],'SELECT 5'),(MODELS[0],'SELECT 6'),(MODELS[2],'SELECT 7')):
            spec=build_specification(model,'zero_shot',[prompt_observation('q001',sql)])
            requests.append({'model_name':model['name'],'mode':'zero_shot','batch':0,'specification':spec,'request_key':request_key(spec)})
        def answer(client,spec):
            return {'status':'api_error','labels':None,'http_status':404,'stop_model':True,'stop_provider':False} if spec['provider']=='groq' else {'status':'ok','labels':[0],'stop_model':False,'stop_provider':False}
        caller=Mock(side_effect=answer)
        with patch('llm_store.LOCAL',self.folder/'llm-stops'):
            result=request_results(self.store,{'pilot_id':'unit-llm-stops'},requests,lambda model:object(),caller,False)
            self.assertEqual([item['status'] for item in result],['api_error','not_run','ok'])
            self.assertEqual(caller.call_count,2)
            replay=request_results(self.store,{'pilot_id':'unit-llm-stops'},requests,lambda model:object(),caller,False)
            self.assertEqual(caller.call_count,2)
            self.assertEqual([item['status'] for item in replay],['api_error','not_run','ok'])


if __name__ == "__main__":
    unittest.main()
