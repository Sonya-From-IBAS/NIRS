"""Read-only verification of published representation IDs and private caches."""

import json

import joblib
from scipy import sparse

from storage import ROOT, Storage, canonical, digest
from study_features import save_json


def main():
    store = Storage()
    study = json.loads((ROOT / 'results/stage7/representation_study.json').read_text(encoding='utf-8'))
    verified = []
    for report in study['reports']:
        configuration = report['configuration']
        rid = report['representation_id']
        with store.connect() as connection:
            registered = connection.execute('SELECT * FROM nirs.representations WHERE representation_id=%s', (rid,)).fetchone()
            roles = connection.execute('SELECT * FROM nirs.representation_artifacts WHERE representation_id=%s', (rid,)).fetchall()
        assert registered['configuration'] == configuration
        assert rid == 'repr_' + digest(canonical({'dataset': registered['dataset_version_id'], 'configuration': configuration}).encode())
        assert {row['role']: row['artifact_id'] for row in roles} == report['artifacts']
        paths = {role: store.get_artifact(aid) for role, aid in report['artifacts'].items()}
        for filename, checksum in configuration['code_sha256_lf'].items():
            assert digest(paths['code/' + filename].read_bytes().replace(b'\r\n', b'\n')) == checksum
            assert digest((ROOT / 'src' / filename).read_bytes().replace(b'\r\n', b'\n')) == checksum
        encoder = joblib.load(paths['encoder.joblib'])
        pipeline = joblib.load(paths['pipeline.joblib'])
        names = json.loads(paths['feature_names.json'].read_text(encoding='utf-8'))
        assert names == encoder.get_feature_names_out().tolist() == pipeline.named_steps['features'].get_feature_names_out().tolist(), (report['cohort'], report['candidate'])
        assert encoder.fit_rows_ == report['train_rows'] == pipeline.named_steps['features'].fit_rows_
        row_ids = {}
        for part in ('train', 'validation'):
            ids = json.loads(paths['rows_' + part + '.json'].read_text(encoding='utf-8'))
            row_ids[part] = ids
            assert ids == store.split_ids(configuration['split_id'], configuration['fold'], part)
            key = 'fit_ids_sha256' if part == 'train' else 'validation_ids_sha256'
            assert digest(canonical(ids).encode()) == configuration[key]
            matrix = sparse.load_npz(paths[part + '.npz'])
            expected = report[part + '_matrix']
            assert sparse.isspmatrix_csr(matrix) and str(matrix.dtype) == 'float32'
            assert list(matrix.shape) == expected['shape'] == [len(ids), len(names)]
            assert matrix.nnz == expected['nnz']
            metadata = next(row['metadata'] for row in roles if row['role'] == part + '.npz')
            assert metadata['encoder_artifact'] == report['artifacts']['encoder.joblib']
            assert metadata['row_ids_artifact'] == report['artifacts']['rows_' + part + '.json']
            assert metadata['feature_names_artifact'] == report['artifacts']['feature_names.json']
        assert not set(row_ids['train']) & set(row_ids['validation'])
        predictions = json.loads(paths['validation_predictions.json'].read_text(encoding='utf-8'))
        assert predictions['observation_ids'] == row_ids['validation']
        verified.append(rid)
    integrity = store.verify_artifacts()
    assert not integrity.get('failures') and not integrity.get('not_ready')
    with store.connect() as connection:
        counts = {table: connection.execute('SELECT count(*) AS n FROM nirs.' + table).fetchone()['n']
                  for table in ('representations', 'representation_artifacts', 'experiment_runs', 'predictions')}
    result = {'version': 'feature-cache-audit-v1', 'active_representations': verified,
              'checks': ['source_hashes', 'sealed_split_row_order', 'matrix_CSR_float32_shape_nnz',
                         'feature_column_order', 'encoder_training_rows', 'matrix_metadata_links',
                         'validation_prediction_row_order', 'artifact_checksums'],
              'artifact_integrity': integrity, 'database_counts': counts}
    save_json(ROOT / 'results/stage7/cache_audit.json', result)
    print(json.dumps({'active_representations': len(verified), 'artifact_integrity': integrity, 'database_counts': counts}))


if __name__ == '__main__':
    main()
