"""Recompute development metrics and verify frozen stage-8 model provenance."""

import json
from functools import lru_cache

import joblib
import numpy as np

from evaluation_metrics import classification_report
from experiment_runner import candidate_record, rank_candidate
from experiment_store import save_json
from storage import ROOT, Storage, canonical, digest


def main():
    store = Storage()
    get_artifact = lru_cache(maxsize=None)(store.get_artifact)
    split_ids = lru_cache(maxsize=None)(store.split_ids)
    @lru_cache(maxsize=None)
    def representation_record(rid):
        with store.connect() as connection:
            return connection.execute('SELECT * FROM nirs.representations WHERE representation_id=%s',(rid,)).fetchone()
    summary = json.loads((ROOT/'results/stage8/summary.json').read_text(encoding='utf-8'))
    checked, trial_ids = [], set()
    for filename in summary['reports']:
        scope = json.loads((ROOT/'results/stage8'/filename).read_text(encoding='utf-8'))
        entry = scope['entry']
        parent = entry['parent_scope']
        parent_ids = split_ids(parent['split_id'], parent['fold'], 'train')
        expected_coordinates = {(int(fold),oid) for fold in entry['folds'] for oid in split_ids(entry['split_id'],int(fold),'validation')}
        assert {oid for _,oid in expected_coordinates} == set(parent_ids)
        assert len(expected_coordinates) == len(parent_ids)
        for item in scope['selected_models'] + scope['rules']:
            run_id = item['run_id']
            with store.connect() as connection:
                run = connection.execute('SELECT * FROM nirs.experiment_runs WHERE run_id=%s',(run_id,)).fetchone()
                rows = connection.execute('SELECT * FROM nirs.predictions WHERE run_id=%s ORDER BY fold,observation_id',(run_id,)).fetchall()
                stored = connection.execute("SELECT value FROM nirs.metrics WHERE run_id=%s AND protocol='inner_validation' AND fold=-1",(run_id,)).fetchone()['value']
                links = {row['role']:row['artifact_id'] for row in connection.execute('SELECT role,artifact_id FROM nirs.run_artifacts WHERE run_id=%s',(run_id,))}
            assert run['status'] == 'complete' and run['configuration']['outer_test_used'] is False
            assert run['configuration']['split_id'] == entry['split_id']
            assert {(row['fold'],row['observation_id']) for row in rows} == expected_coordinates
            assert all(row['protocol'] == 'inner_validation' and row['split_id'] == entry['split_id'] for row in rows)
            actual = classification_report([row['true_label'] for row in rows], [row['predicted_label'] for row in rows],
                                          label_space=scope['label_space'], statuses=[row['status'] for row in rows])
            reported = item['selection']['metrics'] if 'selection' in item else item['metrics']
            assert actual == stored == reported
            assert links == item['artifacts']
            for filename, sha in scope['code_sha256_lf'].items():
                path = get_artifact(links['code/'+filename])
                assert digest(path.read_bytes().replace(b'\r\n',b'\n')) == sha
                assert digest((ROOT/'src'/filename).read_bytes().replace(b'\r\n',b'\n')) == sha
            if 'selection' in item:
                frozen = joblib.load(get_artifact(links['model.joblib']))
                assert frozen.threshold == item['selection']['threshold']
                assert frozen.pipeline.named_steps['features'].fit_rows_ == len(parent_ids)
                refit = representation_record(item['refit_representation_id'])
                refit_config = refit['configuration']
                assert refit['dataset_version_id'] == entry['dataset_version_id']
                assert refit_config['split_id'] == parent['split_id'] and refit_config['fold'] == parent['fold']
                assert refit_config['fit_ids_sha256'] == digest(canonical(parent_ids).encode())
                if scope['label_space'] == 'binary':
                    assert all(int(row['metadata']['score'] >= frozen.threshold) == row['predicted_label'] for row in rows)
                # Every candidate in this selection is reconstructed from its
                # archived fold outputs, not copied from the summary score.
                same_mode = [c for c in scope['candidates'] if c['candidate']['method'] == item['method']
                             and c['candidate']['feature']['input_mode'] == item['input_mode']]
                assert min(same_mode,key=rank_candidate) == item['selection']
                for candidate in same_mode:
                    parts = []
                    for fold, tid in enumerate(candidate['trial_ids']):
                        trial_ids.add(tid)
                        prefix = 'trials/'+tid+'/'
                        trial = json.loads(get_artifact(links[prefix+'metadata.json']).read_text(encoding='utf-8'))
                        representation = representation_record(trial['configuration']['representation_id'])
                        config = representation['configuration']
                        assert representation['dataset_version_id'] == entry['dataset_version_id']
                        assert config['split_id'] == entry['split_id'] and config['fold'] == fold
                        assert config['feature'] == candidate['candidate']['feature']
                        assert config['code_sha256_lf'] == scope['code_sha256_lf']
                        train_ids = split_ids(entry['split_id'],fold,'train')
                        assert config['fit_ids_sha256'] == digest(canonical(train_ids).encode())
                        with np.load(get_artifact(links[prefix+'validation.npz']),allow_pickle=False) as data:
                            truth, output = data['truth'].copy(), data['output'].copy()
                        ids = split_ids(entry['split_id'],fold,'validation')
                        assert config['validation_ids_sha256'] == digest(canonical(ids).encode())
                        assert len(ids) == len(truth) == len(output)
                        assert trial['configuration']['method'] == candidate['candidate']['method']
                        assert trial['configuration']['parameters'] == candidate['candidate']['parameters']
                        parts.append({'fold':fold,'ids':ids,'truth':truth,'output':output,'trial':trial})
                    assert candidate_record(candidate['candidate'],parts,scope['label_space']) == candidate
            checked.append(run_id)
    integrity = store.verify_artifacts()
    assert not integrity['failures'] and not integrity['not_ready']
    with store.connect() as connection:
        total = connection.execute("SELECT count(*) AS n FROM nirs.predictions WHERE run_id=ANY(%s)",(checked,)).fetchone()['n']
    result = {'stage':8,'checked_runs':checked,'reconstructed_classifier_trials':len(trial_ids),
              'checked_predictions':total,'artifact_integrity':integrity,
              'checks':['pinned_validation_coordinates','pooled_metrics_from_DB_rows','candidate_metrics_from_archived_fold_scores',
                        'selection_ranking','binary_threshold_predictions','refit_training_scope','archived_code_hashes','file_checksums'],
              'outer_test_predictions_in_these_runs':0}
    save_json(ROOT/'results/stage8/run_audit.json',result)
    print(json.dumps({**result,'checked_runs':len(checked)}))


if __name__ == '__main__':
    main()
