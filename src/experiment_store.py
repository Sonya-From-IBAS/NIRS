"""Sealed-split input loading, sparse caches and post-row experiment storage."""

import json
from pathlib import Path

import joblib
import numpy as np
from psycopg.types.json import Jsonb
from scipy import sparse

from ml_models import new_encoder
from sql_features import prepare_input
from storage import ROOT, canonical, digest, file_hash

LOCAL = ROOT / '.local/stage8'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def load_inputs(store, entry):
    """Read only the pinned outer TRAIN scope, never outer test observations."""
    parent = entry['parent_scope']
    ids = store.split_ids(parent['split_id'], parent['fold'], 'train')
    with store.connect() as connection:
        connection.execute('SET LOCAL max_parallel_workers_per_gather = 0')
        rows = connection.execute('''SELECT o.observation_id,o.sql_text,o.context_json,a.target_label,c.source_label
            FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
            JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
            JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE d.dataset_version_id=%s AND o.observation_id=ANY(%s)''', (entry['dataset_version_id'], ids)).fetchall()
    indexed = {row['observation_id']: row for row in rows}
    if set(indexed) != set(ids) or len(indexed) != len(rows):
        raise ValueError('Pinned training observations are missing or ambiguous')
    inputs, labels = {}, {}
    for oid in ids:
        row = indexed[oid]
        inputs[oid] = prepare_input({'sql': {'text': row['sql_text']}, 'context': row['context_json']})
        label = int(row['source_label']['label']) if entry['cohort'] == 'legacy' else row['target_label']
        if label not in (range(5) if entry['cohort'] == 'legacy' else (0, 1)):
            raise ValueError('The selected task has no pinned reference label')
        labels[oid] = label
    return ids, inputs, labels


def cache_matrix(store, dataset, split_id, fold, feature, train_ids, validation_ids, inputs, labels, code):
    configuration = {'version': 'stage8-matrices-v1', 'split_id': split_id, 'fold': fold,
                     'fit_part': 'train', 'validation_part': 'validation' if validation_ids else None,
                     'feature': feature, 'fit_ids_sha256': digest(canonical(train_ids).encode()),
                     'validation_ids_sha256': digest(canonical(validation_ids).encode()), 'code_sha256_lf': code}
    rid = 'repr_'+digest(canonical({'dataset': dataset, 'configuration': configuration}).encode())
    with store.connect() as connection:
        known = connection.execute('SELECT configuration FROM nirs.representations WHERE representation_id=%s', (rid,)).fetchone()
        registered = connection.execute('SELECT role,artifact_id FROM nirs.representation_artifacts WHERE representation_id=%s', (rid,)).fetchall() if known else []
    if known:
        if known['configuration'] != configuration:
            raise ValueError('Cached representation configuration differs')
        artifacts = {row['role']: row['artifact_id'] for row in registered}
        paths = {role: store.get_artifact(aid) for role, aid in artifacts.items()}
        for part, ids in (('train', train_ids), ('validation', validation_ids)):
            if json.loads(paths['rows_'+part+'.json'].read_text(encoding='utf-8')) != ids:
                raise ValueError('Cached matrix row order differs')
        encoder = joblib.load(paths['encoder.joblib'])
        train, validation = sparse.load_npz(paths['train.npz']), sparse.load_npz(paths['validation.npz'])
        names = json.loads(paths['feature_names.json'].read_text(encoding='utf-8'))
        if names != encoder.get_feature_names_out().tolist() or train.shape != (len(train_ids), len(names)) or validation.shape != (len(validation_ids), len(names)):
            raise ValueError('Cached matrix columns differ')
        return rid, encoder, train, validation, artifacts
    train_inputs = [inputs[oid] for oid in train_ids]
    encoder = new_encoder(feature)
    train = encoder.fit_transform(train_inputs, [labels[oid] for oid in train_ids])
    validation = encoder.transform([inputs[oid] for oid in validation_ids]) if validation_ids else sparse.csr_matrix((0, train.shape[1]), dtype=np.float32)
    folder = LOCAL/'matrices'/rid
    folder.mkdir(parents=True, exist_ok=True)
    for part, matrix, ids in (('train', train, train_ids), ('validation', validation, validation_ids)):
        sparse.save_npz(folder/(part+'.npz'), matrix, compressed=True)
        save_json(folder/('rows_'+part+'.json'), ids)
        restored = sparse.load_npz(folder/(part+'.npz'))
        if matrix.shape != restored.shape or (matrix != restored).nnz:
            raise ValueError('Sparse cache roundtrip differs')
    save_json(folder/'feature_names.json', encoder.get_feature_names_out().tolist())
    joblib.dump(encoder, folder/'encoder.joblib', compress=3)
    roles = ['train.npz','validation.npz','rows_train.json','rows_validation.json','feature_names.json','encoder.joblib']
    artifacts = {role: store.register_artifact(folder/role) for role in roles}
    for name, expected in code.items():
        path = ROOT/'src'/name
        if digest(path.read_bytes().replace(b'\r\n', b'\n')) != expected:
            raise ValueError('Representation source changed during fitting')
        artifacts['code/'+name] = store.register_artifact(path)
    with store.write() as connection:
        connection.execute('INSERT INTO nirs.representations VALUES (%s,%s,%s)', (rid, dataset, Jsonb(configuration)))
        for role, aid in artifacts.items():
            metadata = {'fit_rows': len(train_ids), 'fit_split_id': split_id, 'fit_fold': fold}
            if role in ('train.npz', 'validation.npz'):
                part, matrix = ('train', train) if role == 'train.npz' else ('validation', validation)
                metadata.update(shape=list(matrix.shape), dtype=str(matrix.dtype), format='csr', nnz=matrix.nnz,
                                row_ids_artifact=artifacts['rows_'+part+'.json'], encoder_artifact=artifacts['encoder.joblib'],
                                feature_names_artifact=artifacts['feature_names.json'])
            connection.execute('INSERT INTO nirs.representation_artifacts VALUES (%s,%s,%s,%s)', (rid, aid, role, Jsonb(metadata)))
    return rid, encoder, train, validation, artifacts


def persist_run(store, entry, method, label_space, configuration, observations, reports, files):
    coordinates = [(row['fold'],row['observation_id']) for row in observations]
    with store.connect() as connection:
        references = connection.execute('''SELECT s.fold,s.observation_id,a.target_label,c.source_label
            FROM nirs.split_members s JOIN nirs.dataset_members d
              ON d.dataset_version_id=s.dataset_version_id AND d.observation_id=s.observation_id
            JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
            JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE s.split_id=%s AND s.dataset_version_id=%s AND s.part='validation' ''',
            (entry['split_id'],entry['dataset_version_id'])).fetchall()
    pinned = {(row['fold'],row['observation_id']): int(row['source_label']['label']) if label_space == 'legacy_five_class'
              else row['target_label'] for row in references}
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != set(pinned):
        raise ValueError('Predictions must exactly cover every frozen inner validation coordinate')
    if any(row['truth'] != pinned[(row['fold'],row['observation_id'])] for row in observations):
        raise ValueError('Prediction reference differs from the pinned dataset annotation')
    run_id = 'run_'+digest(canonical({'dataset': entry['dataset_version_id'], 'method': method, 'configuration': configuration}).encode())
    artifacts = {role: store.register_artifact(path) for role, path in files.items()}
    with store.write() as connection:
        known = connection.execute('SELECT configuration,status FROM nirs.experiment_runs WHERE run_id=%s', (run_id,)).fetchone()
        if known:
            if known['configuration'] != configuration or known['status'] != 'complete':
                raise ValueError('An existing run is inconsistent')
            count = connection.execute('SELECT count(*) AS n FROM nirs.predictions WHERE run_id=%s', (run_id,)).fetchone()['n']
            if count != len(observations):
                raise ValueError('An existing run has incomplete predictions')
            saved = {row['role']: row['artifact_id'] for row in connection.execute('SELECT role,artifact_id FROM nirs.run_artifacts WHERE run_id=%s', (run_id,))}
            if saved != artifacts:
                raise ValueError('An existing run has different artifact roles')
            return run_id, artifacts
        connection.execute("INSERT INTO nirs.experiment_runs VALUES (%s,%s,%s,%s,'prepared',%s)",
                           (run_id, entry['dataset_version_id'], method, label_space, Jsonb(configuration)))
        store._copy(connection, 'predictions', ['run_id','dataset_version_id','split_id','observation_id','protocol','fold','true_label','predicted_label','status','metadata'],
                    [(run_id,entry['dataset_version_id'],entry['split_id'],row['observation_id'],'inner_validation',row['fold'],
                      row['truth'],row['prediction'],row['status'],Jsonb(row['metadata'])) for row in observations])
        for fold, report in reports.items():
            connection.execute('INSERT INTO nirs.metrics VALUES (%s,%s,%s,%s,%s)', (run_id,'inner_validation',int(fold),'classification_report',Jsonb(report)))
        for role, aid in artifacts.items():
            connection.execute('INSERT INTO nirs.run_artifacts VALUES (%s,%s,%s,%s)', (run_id,aid,role,Jsonb({'stage':8})))
        connection.execute("UPDATE nirs.experiment_runs SET status='complete' WHERE run_id=%s", (run_id,))
    return run_id, artifacts
