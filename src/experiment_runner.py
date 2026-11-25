"""Validation-only ML selection, frozen refits and versioned rule controls."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import platform
import time
import warnings
from importlib.metadata import version as package_version

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from evaluation_metrics import classification_report
from experiment_store import LOCAL, cache_matrix, load_inputs, persist_run, save_json
from ml_models import (VERSION, FrozenDecisionModel, binary_scores, choose_threshold,
                       classifier_candidates, feature_candidates, new_classifier)
from rule_detectors import rule_predictions
from storage import ROOT, Storage, canonical, digest, file_hash

OUTPUT = ROOT/'results/stage8'
METHODS = ('LogisticRegression', 'LinearSVC', 'RandomForest')
SCOPES = ('controlled-inner-0', 'legacy-inner-0', 'transfer-web_app-to-wordpress-inner')
SOURCES = ('baseline.py','ml_models.py','rule_detectors.py','experiment_store.py','experiment_runner.py',
           'sql_views.py','sql_features.py','relational_features.py','normalize.py','evaluation_metrics.py')


def code_signature():
    return {name: digest((ROOT/'src'/name).read_bytes().replace(b'\r\n', b'\n')) for name in SOURCES}


def trial_fit(representation, encoder, train, validation, y_train, y_validation, method, parameters, label_space, signature):
    configuration = {'representation_id': representation, 'method': method, 'parameters': parameters,
                     'label_space': label_space, 'code_sha256_lf': signature}
    trial_id = 'trial_'+digest(canonical(configuration).encode())
    folder = LOCAL/'trials'/trial_id
    metadata = folder/'metadata.json'
    if metadata.exists():
        report = json.loads(metadata.read_text(encoding='utf-8'))
        if report['configuration'] != configuration or any(file_hash(folder/name) != sha for name, sha in report['files_sha256'].items()):
            raise ValueError('Trial cache differs from its frozen manifest')
        with np.load(folder/'validation.npz', allow_pickle=False) as data:
            if not np.array_equal(data['truth'], y_validation):
                raise ValueError('Trial cache references differ')
            return report, data['output'].copy(), folder
    folder.mkdir(parents=True, exist_ok=True)
    model = new_classifier(method, parameters)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(train, y_train)
    messages = [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError('A candidate did not converge; do not silently select it')
    binary = label_space == 'binary'
    output = binary_scores(model, validation) if binary else model.predict(validation)
    predicted = ((output >= (.5 if method == 'RandomForest' else 0)).astype(int) if binary else output)
    train_prediction = model.predict(train)
    pipeline = Pipeline([('features', deepcopy(encoder)), ('classifier', model)])
    joblib.dump(pipeline, folder/'pipeline.joblib', compress=3)
    np.savez_compressed(folder/'validation.npz', truth=y_validation, output=output)
    restored = joblib.load(folder/'pipeline.joblib')
    repeated = binary_scores(restored.named_steps['classifier'], validation) if binary else restored.named_steps['classifier'].predict(validation)
    if not np.array_equal(repeated, output):
        raise ValueError('Reloaded trial classifier scores differ')
    report = {'trial_id': trial_id, 'configuration': configuration, 'columns': train.shape[1],
              'fit_rows': train.shape[0], 'validation_rows': validation.shape[0],
              'train_default_metrics': classification_report(y_train.tolist(), train_prediction.tolist(), label_space=label_space),
              'validation_default_metrics': classification_report(y_validation.tolist(), predicted.tolist(), label_space=label_space),
              'fit_and_reload_seconds': round(time.perf_counter()-started, 4), 'warnings': messages,
              'classifier_score_reload': 'exact',
              'files_sha256': {name: file_hash(folder/name) for name in ('pipeline.joblib','validation.npz')}}
    save_json(metadata, report)
    return report, output, folder


def rank_candidate(candidate):
    metric = candidate['metrics']
    secondary = metric['successful_decision_metrics']
    return (-metric['primary_f1'], secondary['false_positive_rate'] if secondary['false_positive_rate'] is not None else float('inf'),
            -(secondary['attack_recall'] or 0), tuple(candidate['complexity']))


def candidate_record(candidate, parts, label_space):
    ids = [oid for part in parts for oid in part['ids']]
    if len(ids) != len(set(ids)):
        raise ValueError('Inner validation observations must occur exactly once')
    truth = np.concatenate([part['truth'] for part in parts])
    output = np.concatenate([part['output'] for part in parts])
    threshold = choose_threshold(truth, output) if label_space == 'binary' else None
    predictions = (output >= threshold).astype(int) if threshold is not None else output
    metrics = classification_report(truth.tolist(), predictions.tolist(), label_space=label_space)
    parameters = candidate['parameters']
    widths = [part['trial']['columns'] for part in parts]
    complexity = [float(np.mean(widths)), parameters.get('C', 0), parameters.get('max_depth') or 100000,
                  -parameters.get('min_samples_leaf', 1), canonical(candidate)]
    return {'candidate': candidate, 'threshold': threshold, 'metrics': metrics, 'complexity': complexity,
            'trial_ids': [part['trial']['trial_id'] for part in parts],
            'folds': {str(part['fold']): classification_report(part['truth'].tolist(),
                      ((part['output'] >= threshold).astype(int) if threshold is not None else part['output']).tolist(),
                      label_space=label_space) for part in parts}}


def refit_selected(store, entry, selected, parent_ids, inputs, labels, signature):
    candidate = selected['candidate']
    parent = entry['parent_scope']
    rid, encoder, matrix, _, matrix_artifacts = cache_matrix(store, entry['dataset_version_id'], parent['split_id'], parent['fold'],
                 candidate['feature'], parent_ids, [], inputs, labels, signature)
    configuration = {'selection': selected, 'refit_representation': rid, 'training_ids_sha256': digest(canonical(parent_ids).encode())}
    folder = LOCAL/'refits'/digest(canonical(configuration).encode())
    model_file, metadata_file = folder/'model.joblib', folder/'metadata.json'
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        if metadata['configuration'] != configuration or file_hash(model_file) != metadata['model_sha256']:
            raise ValueError('Frozen refit manifest differs')
        frozen = joblib.load(model_file)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        model = new_classifier(candidate['method'], candidate['parameters'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(matrix, [labels[oid] for oid in parent_ids])
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ValueError('The full-training refit did not converge')
        pipeline = Pipeline([('features', encoder), ('classifier', model)])
        frozen = FrozenDecisionModel(pipeline, selected['threshold'])
        joblib.dump(frozen, model_file, compress=3)
        metadata = {'configuration': configuration, 'model_sha256': file_hash(model_file), 'fit_rows': len(parent_ids),
                    'matrix_artifacts': matrix_artifacts, 'outer_test_used': False,
                    'warnings': [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]}
        save_json(metadata_file, metadata)
    # This proof reads training inputs only. Outer test remains closed.
    sample = [inputs[oid] for oid in parent_ids[:31]]
    loaded = joblib.load(model_file)
    if not np.array_equal(frozen.predict(sample), loaded.predict(sample)):
        raise ValueError('Frozen model predictions differ after reload')
    return {'model.joblib': model_file, 'refit_metadata.json': metadata_file}, rid


def run_scope(store, name, signature, code_files):
    registry = json.loads((ROOT/'results/stage6/split_registry.json').read_text(encoding='utf-8'))['protocols']
    entry = registry[name]
    label_space = 'legacy_five_class' if entry['cohort'] == 'legacy' else 'binary'
    parent_ids, inputs, labels = load_inputs(store, entry)
    assignments = {}
    for fold in sorted(map(int, entry['folds'])):
        train_ids = store.split_ids(entry['split_id'], fold, 'train')
        validation_ids = store.split_ids(entry['split_id'], fold, 'validation')
        if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != set(parent_ids):
            raise ValueError('Inner scope differs from its frozen parent training set')
        assignments[fold] = train_ids, validation_ids
    candidates = {}
    for feature in feature_candidates(entry['cohort']):
        for fold, (train_ids, validation_ids) in assignments.items():
            rid, encoder, train, validation, _ = cache_matrix(store, entry['dataset_version_id'], entry['split_id'], fold,
                        feature, train_ids, validation_ids, inputs, labels, signature)
            y_train = np.asarray([labels[oid] for oid in train_ids], dtype=np.int64)
            y_validation = np.asarray([labels[oid] for oid in validation_ids], dtype=np.int64)
            for method in METHODS:
                for parameters in classifier_candidates(method):
                    candidate = {'feature': feature, 'method': method, 'parameters': parameters}
                    key = digest(canonical(candidate).encode())
                    record = candidates.setdefault(key, {'candidate': candidate, 'parts': []})
                    trial, output, folder = trial_fit(rid, encoder, train, validation, y_train, y_validation,
                                                       method, parameters, label_space, signature)
                    record['parts'].append({'fold': fold, 'ids': validation_ids, 'truth': y_validation, 'output': output,
                                            'trial': trial, 'folder': folder})
            print(f'{name}: {feature["name"]}, inner fold {fold} completed', flush=True)
    all_records = {key: candidate_record(value['candidate'], value['parts'], label_space) for key, value in candidates.items()}
    result = {'scope': name, 'entry': entry, 'role': 'inner_validation_model_development', 'outer_test_used': False,
              'label_space': label_space, 'training_scope_rows': len(parent_ids), 'code_sha256_lf': signature,
              'candidate_count': len(candidates), 'classifier_fits': sum(len(v['parts']) for v in candidates.values()),
              'candidates': list(all_records.values()), 'selected_models': [], 'rules': []}
    for mode in sorted({candidate['candidate']['feature']['input_mode'] for candidate in candidates.values()}):
        for method in METHODS:
            choices = [key for key, value in candidates.items() if value['candidate']['method'] == method and value['candidate']['feature']['input_mode'] == mode]
            winner = min(choices, key=lambda key: rank_candidate(all_records[key]))
            selected, parts = all_records[winner], candidates[winner]['parts']
            observations, files = [], dict(code_files)
            # Every tried configuration remains reconstructible, including scores
            # before threshold selection and each fitted inner pipeline.
            for key in choices:
                for part in candidates[key]['parts']:
                    trial = part['trial']['trial_id']
                    for filename in ('pipeline.joblib','validation.npz','metadata.json'):
                        files['trials/'+trial+'/'+filename] = part['folder']/filename
            for part in parts:
                pipeline = joblib.load(part['folder']/'pipeline.joblib')
                sample_inputs = [inputs[oid] for oid in part['ids']]
                actual = (binary_scores(pipeline.named_steps['classifier'], pipeline.named_steps['features'].transform(sample_inputs))
                          if label_space == 'binary' else pipeline.predict(sample_inputs))
                if not np.array_equal(actual, part['output']):
                    raise ValueError('Selected full pipeline validation scores differ after reload')
                prediction = (actual >= selected['threshold']).astype(int) if selected['threshold'] is not None else actual
                for oid, truth, value, output in zip(part['ids'], part['truth'], prediction, actual):
                    observations.append({'observation_id': oid, 'fold': part['fold'], 'truth': int(truth), 'prediction': int(value),
                        'status': 'ok', 'metadata': {'stage':8, 'role':'inner_validation_used_for_selection',
                                                   'score':float(output), 'threshold':selected['threshold']}})
            if {row['observation_id'] for row in observations} != set(parent_ids):
                raise ValueError('Selected validation predictions do not cover the parent scope')
            refit_files, rid = refit_selected(store, entry, selected, parent_ids, inputs, labels, signature)
            files.update(refit_files)
            folder = LOCAL/'selections'/name/(method+'_'+mode)
            save_json(folder/'selection.json', {'selected':selected, 'trials':[all_records[key] for key in choices]})
            files['selection.json'] = folder/'selection.json'
            configuration = {'version':VERSION, 'scope':name, 'input_mode':mode, 'role':'inner_validation_used_for_selection',
                'split_id':entry['split_id'], 'parent_scope':entry['parent_scope'], 'code_sha256_lf':signature,
                'selection_sha256':file_hash(folder/'selection.json'), 'selected':selected['candidate'],
                'threshold':selected['threshold'], 'refit_representation_id':rid, 'outer_test_used':False}
            run_id, artifacts = persist_run(store, entry, method, label_space, configuration, observations,
                                            {-1:selected['metrics'], **selected['folds']}, files)
            result['selected_models'].append({'run_id':run_id,'method':method,'input_mode':mode,'selection':selected,
                                              'artifacts':artifacts,'refit_representation_id':rid})
            print(f'{name}/{method}/{mode}: inner tuned F1={selected["metrics"]["primary_f1"]:.4f}', flush=True)
    rule_names = ['regex_v1'] if label_space == 'legacy_five_class' else ['regex_v1','regex_v2']
    if entry['cohort'] == 'controlled':
        rule_names += ['policy_only','regex_policy']
    for rule in rule_names:
        observations, reports, reasons = [], {}, Counter()
        for fold, (_, ids) in assignments.items():
            prediction, statuses, detail = rule_predictions(rule, [inputs[oid] for oid in ids], legacy=label_space != 'binary')
            truth = [labels[oid] for oid in ids]
            reports[str(fold)] = classification_report(truth, prediction, label_space=label_space, statuses=statuses)
            for oid, target, value, status, reason in zip(ids, truth, prediction, statuses, detail):
                observations.append({'observation_id':oid,'fold':fold,'truth':target,'prediction':value,'status':status,'metadata':{'reason':reason}})
                reasons[reason] += 1
        reports[-1] = classification_report([x['truth'] for x in observations], [x['prediction'] for x in observations],
                         label_space=label_space, statuses=[x['status'] for x in observations])
        configuration = {'version':VERSION,'scope':name,'rule_version':rule,'split_id':entry['split_id'],
                         'role':'inner_validation_rule_development','code_sha256_lf':signature,'outer_test_used':False,
                         'historical_binary_signal':'nonzero_class_is_alarm' if rule == 'regex_v1' else None}
        run_id, artifacts = persist_run(store, entry, rule, label_space, configuration, observations, reports, code_files)
        result['rules'].append({'run_id':run_id,'method':rule,'metrics':reports[-1],'folds':{str(k):v for k,v in reports.items() if k != -1},
                                'reason_counts':dict(reasons),'artifacts':artifacts})
    if code_signature() != signature:
        raise ValueError('Experimental source changed during execution')
    save_json(OUTPUT/(name+'.json'), result)
    return result


def freeze_plan(scope_names):
    signature = code_signature()
    registry = json.loads((ROOT/'results/stage6/split_registry.json').read_text(encoding='utf-8'))['protocols']
    plan = {'version':VERSION,'scopes':list(scope_names),'phase':'inner_validation_and_train_refits', 'outer_test_used':False,
            'code_sha256_lf':signature,'selection':'pooled inner out-of-fold F1; lower FPR; higher recall; fixed complexity order',
            'threshold':'exact sweep of pooled inner validation scores; equal scores move together',
            'features':{registry[name]['cohort']:feature_candidates(registry[name]['cohort']) for name in scope_names},
            'classifiers':{method:classifier_candidates(method) for method in METHODS},
            'environment':{'python':platform.python_version(), 'packages':{name:package_version(name) for name in
                           ('numpy','scipy','scikit-learn','joblib','threadpoolctl','psycopg')},
                           'threads':1, 'random_seed':20261125},
            'migration_hashes':{path.name:file_hash(path) for path in sorted((ROOT/'migrations').glob('*.sql'))}}
    save_json(OUTPUT/'search_plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', action='append', help='Frozen inner protocol name; repeatable')
    parser.add_argument('--all-scopes', action='store_true', help='All eight frozen inner protocols')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.scope and args.all_scopes:
        parser.error('Choose explicit scopes or --all-scopes')
    registry = json.loads((ROOT/'results/stage6/split_registry.json').read_text(encoding='utf-8'))['protocols']
    scopes = tuple(name for name,value in registry.items() if value['role'] == 'inner_tuning') if args.all_scopes else tuple(args.scope or SCOPES)
    if not scopes or len(set(scopes)) != len(scopes) or any(name not in registry or registry[name]['role'] != 'inner_tuning' for name in scopes):
        parser.error('Every requested scope must be a distinct frozen inner protocol')
    plan = freeze_plan(scopes)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    store = Storage()
    store.migrate()
    signature = plan['code_sha256_lf']
    code_files = {'code/'+name:ROOT/'src'/name for name in SOURCES}
    code_files.update({'schema/'+path.name:path for path in sorted((ROOT/'migrations').glob('*.sql'))})
    # Capture executable source before fitting, including interrupted attempts.
    for path in code_files.values():
        store.register_artifact(path)
    reports = []
    with threadpool_limits(limits=1):
        for scope in scopes:
            reports.append(run_scope(store, scope, signature, code_files))
    save_json(OUTPUT/'summary.json', {'version':VERSION,'scopes':list(scopes),'outer_test_used':False,
              'classifier_fits':sum(report['classifier_fits'] for report in reports),
              'selected_models':sum(len(report['selected_models']) for report in reports),
              'rule_runs':sum(len(report['rules']) for report in reports),
              'reports':[scope+'.json' for scope in scopes]})


if __name__ == '__main__':
    main()
