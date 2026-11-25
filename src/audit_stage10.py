"""Recompute stage10 metrics from pinned database rows and verify the main API ledger."""

from collections import Counter
import json

import numpy as np
from sklearn.metrics import confusion_matrix,f1_score

from detector_llm import parse_batch_response,request_key
from evaluation_metrics import classification_report
from experiment_store import save_json
from main_llm import read_requests
from storage import ROOT,Storage,canonical,digest

OUTPUT=ROOT/'results/stage10'


def audit_run(store,run_id):
    with store.connect() as connection:
        run=connection.execute('SELECT * FROM nirs.experiment_runs WHERE run_id=%s',(run_id,)).fetchone()
        rows=connection.execute('SELECT * FROM nirs.predictions WHERE run_id=%s ORDER BY protocol,fold,observation_id',(run_id,)).fetchall()
        metrics=connection.execute("SELECT * FROM nirs.metrics WHERE run_id=%s AND metric_name='classification_report'",(run_id,)).fetchall()
        artifacts=connection.execute('SELECT * FROM nirs.run_artifacts WHERE run_id=%s',(run_id,)).fetchall()
        invalid=connection.execute('''SELECT count(*) AS n FROM nirs.predictions p
            LEFT JOIN nirs.split_members s ON s.split_id=p.split_id AND s.dataset_version_id=p.dataset_version_id
            AND s.fold=p.fold AND s.observation_id=p.observation_id
            JOIN nirs.dataset_members d ON d.dataset_version_id=p.dataset_version_id AND d.observation_id=p.observation_id
            JOIN nirs.annotations a USING(annotation_id) JOIN nirs.occurrences c
            ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE p.run_id=%s AND (s.part IS DISTINCT FROM CASE WHEN p.protocol='inner_validation' THEN 'validation' ELSE 'test' END
            OR p.true_label IS DISTINCT FROM CASE WHEN %s THEN (c.source_label->>'label')::smallint ELSE a.target_label END
            OR (NOT %s AND a.status<>'verified'))''',(run_id,run['label_space']=='legacy_five_class',run['label_space']=='legacy_five_class')).fetchone()['n']
    assert run and run['status']=='complete' and rows and invalid==0
    for metric in metrics:
        selected=[row for row in rows if row['protocol']==metric['protocol'] and (metric['fold']==-1 or row['fold']==metric['fold'])]
        actual=classification_report([row['true_label'] for row in selected],[row['predicted_label'] for row in selected],
            statuses=[row['status'] for row in selected],label_space=run['label_space'])
        assert actual==metric['value'],run_id
        successful=[row for row in selected if row['status']=='ok']
        if successful:
            labels=list(range(5 if run['label_space']=='legacy_five_class' else 2))
            truth=[row['true_label'] for row in successful];pred=[row['predicted_label'] for row in successful]
            assert confusion_matrix(truth,pred,labels=labels).tolist()==actual['successful_decision_metrics']['confusion_matrix']
            expected=f1_score(truth,pred,labels=labels,average='macro' if len(labels)==5 else 'binary',zero_division=0)
            measured=actual['successful_decision_metrics']['f1']
            if measured is not None:
                assert abs(expected-measured)<1e-12
    for artifact in artifacts:
        store.get_artifact(artifact['artifact_id'])
    return {'run_id':run_id,'method':run['method'],'predictions':len(rows),'protocols':sorted({row['protocol'] for row in rows}),
            'label_space':run['label_space'],'metrics_checked':len(metrics),'artifact_roles':len(artifacts)}


def audit_main(store):
    folder=OUTPUT/'llm'
    plan=json.loads((folder/'main_plan.json').read_text(encoding='utf-8'))
    report=json.loads((folder/'main_report.json').read_text(encoding='utf-8'))
    requests=read_requests(folder/'request_plan.jsonl.gz');outcomes=read_requests(folder/'request_outcomes.jsonl.gz')
    inputs=json.loads((folder/'main_inputs.json').read_text(encoding='utf-8'))
    assert len(requests)==len(outcomes)==plan['max_api_attempts']==768
    assert [row['request_key'] for row in requests]==plan['ordered_request_keys']
    with store.connect() as connection:
        job=connection.execute('SELECT * FROM nirs.llm_pilots WHERE pilot_id=%s',(plan['pilot_id'],)).fetchone()
        calls=connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=ANY(%s)',(plan['ordered_request_keys'],)).fetchall()
    assert job['configuration']==plan and job['max_api_attempts']==768
    assert sum(row['api_attempts'] for row in calls)==report['actual_API_attempts']<=768
    cached={row['request_key']:row for row in calls}; aligned={}
    for request,outcome in zip(requests,outcomes):
        key=request_key(request['specification'])
        assert key==request['request_key']==outcome['request_key']
        spec=request['specification']
        assert spec['sdk_max_retries']==0
        if outcome['status']!='not_run':
            saved=cached[key]
            assert saved['state']=='complete' and saved['specification']==spec
            assert all(outcome.get(name)==value for name,value in saved['result'].items())
            assert json.loads(store.get_artifact(saved['request_artifact']).read_text(encoding='utf-8'))==spec
            assert json.loads(store.get_artifact(saved['response_artifact']).read_text(encoding='utf-8'))=={'request_key':key,'result':saved['result']}
        if outcome['status']=='ok':
            assert outcome['finish_reason']=='stop'
            assert parse_batch_response(outcome['final_content'],spec['batch_aliases'])==outcome['labels']
        labels=outcome.get('labels') if outcome['status']=='ok' else [None]*len(request['observation_ids'])
        for oid,label in zip(request['observation_ids'],labels):
            aligned[(request['model_name'],request['input_mode'],request['mode'],oid)]=(label,outcome['status'],key)
        if request['input_mode']=='sql_only':
            observations=json.loads(spec['messages'][-1]['content'].split('\n',1)[1])
            assert all(row['context']=={} for row in observations)
            if request['mode']=='few_shot':
                demos=json.loads(spec['messages'][1]['content'].split('\n',1)[1])
                assert all(row['observation']['context']=={} for row in demos)
    assert len(aligned)==3000
    sample_ids={row['observation_id'] for row in inputs['samples']}
    controlled_ids={row['observation_id'] for row in inputs['samples'] if row['task']!='sqli'}
    for result in report['runs']:
        config=result['configuration']
        with store.connect() as connection:
            rows=connection.execute('SELECT * FROM nirs.predictions WHERE run_id=%s',(result['run_id'],)).fetchall()
        assert {row['observation_id'] for row in rows}==(sample_ids if config['input_mode']=='sql_context' else controlled_ids)
        for row in rows:
            expected=aligned[(config['model'],config['input_mode'],config['shot_mode'],row['observation_id'])]
            assert (row['predicted_label'],row['status'],row['metadata']['request_key'])==expected
    for name,aid in plan['code_artifacts'].items():
        assert digest(store.get_artifact(aid).read_bytes().replace(b'\r\n',b'\n'))==plan['code_sha256_lf'][name]
    return {'requests_planned':768,'API_attempts':report['actual_API_attempts'],'classifications_planned':3000,
            'decision_status_counts':dict(Counter(value[1] for value in aligned.values()))}


def main():
    store=Storage(); identifiers=set()
    for path in (OUTPUT/'selections').glob('*.json'):
        data=json.loads(path.read_text(encoding='utf-8'))
        identifiers.update(row['run_id'] for row in data['selected_models']+data['rules'])
    for name in ('outer_results','representation_results','generation_results','diagnostic_results','historical_results'):
        data=json.loads((OUTPUT/(name+'.json')).read_text(encoding='utf-8'))
        identifiers.update(row['run_id'] for row in data)
    llm=json.loads((OUTPUT/'llm/main_report.json').read_text(encoding='utf-8'))
    identifiers.update(row['run_id'] for row in llm['runs'])
    checked=[]
    for index,run_id in enumerate(sorted(identifiers)):
        checked.append(audit_run(store,run_id))
        if index%25==0:
            print(f'Audited {index+1}/{len(identifiers)} stage10 runs',flush=True)
    main_check=audit_main(store)
    integrity=store.verify_artifacts()
    assert not integrity['not_ready'] and not integrity['failures']
    result={'stage':10,'checked_runs':checked,'total_prediction_rows':sum(row['predictions'] for row in checked),
            'main_LLM':main_check,'artifact_integrity':integrity,'checks':['pinned_split_parts','pinned_annotations',
                'metrics_recomputed_from_rows','independent_sklearn_confusion_and_F1','per_request_API_cache',
                'exact_sample_ID_alignment','context_ablated_from_queries_and_demos','archived_sources','artifact_checksums']}
    save_json(OUTPUT/'audit_report.json',result)
    print(json.dumps({'runs':len(checked),'predictions':result['total_prediction_rows'],'main_API_attempts':main_check['API_attempts']}))


if __name__=='__main__':
    main()
