"""Stage11: database-based preview metrics, grouped uncertainty, error slices and audits."""

from collections import Counter,defaultdict
import json
import math

import numpy as np

from audit_stage10 import audit_run
from evaluation_metrics import classification_report
from experiment_store import save_json
from storage import ROOT,Storage

OUTPUT=ROOT/'results/preview'


def wilson(successes,total):
    if not total:return None
    z=1.959963984540054;p=successes/total;denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-width),min(1.,center+width)]


def statistics(rows):
    truth=[row['true_label'] for row in rows];pred=[row['predicted_label'] for row in rows];status=[row['status'] for row in rows]
    normal=classification_report(truth,pred,statuses=status)
    conservative=[value if state=='ok' else 1-target for target,value,state in zip(truth,pred,status)]
    lower=classification_report(truth,conservative)
    sizes=Counter(row['group_id'] for row in rows)
    weighted=classification_report(truth,pred,statuses=status,sample_weight=[1/sizes[row['group_id']] for row in rows])
    groups=defaultdict(lambda:np.zeros(4,dtype=float))
    for row,value in zip(rows,conservative):
        groups[row['group_id']][2*row['true_label']+value]+=1
    vectors=np.asarray(list(groups.values()))
    rng=np.random.default_rng(20261125)
    totals=vectors[rng.integers(0,len(vectors),size=(1000,len(vectors)))].sum(axis=1)
    denominator=2*totals[:,3]+totals[:,1]+totals[:,2]
    scores=np.divide(2*totals[:,3],denominator,out=np.full(1000,np.nan),where=denominator>0)
    finite=scores[np.isfinite(scores)]
    interval=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
    matrix=lower['successful_decision_metrics']['confusion_matrix'];tn,fp=matrix[0];fn,tp=matrix[1]
    return {'observations':len(rows),'unique_groups':len(groups),'metrics':normal,
        'conservative_all_cases':lower,'equal_group_weighted':weighted,'group_bootstrap_F1_95':interval,
        'FPR_row_Wilson_95_descriptive':wilson(fp,tn+fp),'recall_row_Wilson_95_descriptive':wilson(tp,tp+fn),
        'uncertainty_note':'group bootstrap uses empirical linked groups; row Wilson intervals are descriptive and assume independent rows; neither guarantees operational FPR'}


def read(name):
    return json.loads((OUTPUT/name).read_text(encoding='utf-8'))


def main():
    store=Storage()
    local=read('ml_results.json');llm=read('llm_results.json')['runs'];transfer=read('transfer_results.json')
    common=[row for row in local if row['phase']=='common_30']+llm
    samples={row['observation_id']:row for row in read('common_inputs.json')['samples']}
    results=[];slices=[];errors=[]
    timings=[]
    for result in common:
        with store.connect() as connection:
            rows=connection.execute('''SELECT p.*,m.group_id FROM nirs.predictions p JOIN nirs.group_members m USING(observation_id)
                JOIN nirs.sample_groups g USING(group_id) WHERE p.run_id=%s AND g.kind='stage6_component' ORDER BY p.observation_id''',(result['run_id'],)).fetchall()
        if len(rows)!=30 or {row['observation_id'] for row in rows}!=set(samples):
            raise ValueError('Preview methods must use exactly the same thirty IDs')
        config=result['configuration']; name=result['method']
        if not name.startswith('LLM '):name+=' / '+config['input_mode']
        summary={'run_id':result['run_id'],'method':name,'input_mode':config['input_mode'],**statistics(rows)}
        results.append(summary)
        if name.startswith('LLM '):
            keys=sorted({row['metadata']['request_key'] for row in rows})
            with store.connect() as connection:
                calls=connection.execute('SELECT request_key,specification,result FROM nirs.llm_requests WHERE request_key=ANY(%s)',(keys,)).fetchall()
            lookup={row['request_key']:row for row in calls}
            values=[lookup[row['metadata']['request_key']]['result']['elapsed_s']/len(lookup[row['metadata']['request_key']]['specification']['ordered_observation_ids'])
                    for row in rows if row['status']=='ok']
            batch=[row['result']['elapsed_s'] for row in calls if row['result'] and row['result'].get('elapsed_s') is not None]
            timings.append({'method':name,'seconds_per_observation_amortized':float(np.mean(values)) if values else None,
                'batch_latency_p50_s':float(np.quantile(batch,.5)) if batch else None,
                'batch_latency_p95_s':float(np.quantile(batch,.95)) if batch else None,'unique_API_batches_referenced':len(calls)})
        else:
            timings.append({'method':name,'seconds_per_observation_amortized':result['prediction_seconds']/30,
                'batch_latency_p50_s':None,'batch_latency_p95_s':None,'model_bytes':result.get('model_bytes')})
        by_slice=defaultdict(list)
        for row in rows:
            entry=samples[row['observation_id']];context=entry['payload']['context'];dialect=context.get('dialect',{}).get('value') or 'unknown'
            task=entry['task'];tags=entry['representation_tags'];length=len(entry['payload']['sql']['text'])
            labels=[('threat',task),('dialect',str(dialect)),('source','SQLStateGuard' if task=='sqli' else 'controlled'),
                ('length','short' if length<=200 else 'medium' if length<=1000 else 'long'),
                ('context','known_policy' if context.get('policy',{}).get('availability')=='known' else 'unknown_policy')]
            if task!='sqli':labels.append(('representation','obfuscated' if 'obfuscated' in tags else 'controlled_base'))
            for kind,label in labels:by_slice[(kind,label)].append(row)
            if row['status']!='ok' or row['predicted_label']!=row['true_label']:
                errors.append({'method':name,'run_id':result['run_id'],'observation_id':row['observation_id'],
                    'task':task,'dialect':dialect,'tags':tags,'true_label':row['true_label'],'predicted_label':row['predicted_label'],
                    'status':row['status'],'SQL_excerpt':entry['payload']['sql']['text'][:900],
                    'category':'no_decision' if row['status']!='ok' else 'false_alarm' if row['true_label']==0 else 'missed_attack'})
        for (kind,label),selected in by_slice.items():
            slices.append({'run_id':result['run_id'],'method':name,'slice_kind':kind,'slice':label,**statistics(selected)})
    save_json(OUTPUT/'analysis.json',{'purpose':'preliminary_preview','common':results,'slices':slices,
        'error_cases':errors,'failure_rule':'every missing decision flips its true binary class in the conservative estimate',
        'test_status':'exploratory slice is now open; a repeat on it is not a new blind test'})
    llm_info=read('llm_results.json');llm_plan=read('llm_plan.json')
    with store.connect() as connection:
        wall=connection.execute('SELECT extract(epoch FROM max(finished_at)-min(started_at)) AS seconds FROM nirs.llm_requests WHERE origin_pilot_id=%s',(llm_plan['pilot_id'],)).fetchone()['seconds']
        previous=connection.execute('SELECT count(*) AS n FROM nirs.llm_requests WHERE origin_pilot_id=%s',(llm_plan['source_main_id'],)).fetchone()['n']
    save_json(OUTPUT/'timing_and_cost.json',{'methods':timings,'new_preview_API_attempts':llm_info['new_API_attempts'],
        'preview_new_API_wall_seconds':float(wall) if wall is not None else None,'preview_observed_tokens':llm_info['observed_tokens'],
        'previous_interrupted_full_API_attempts':previous,'reused_decisions':llm_info['reused_decisions'],
        'money_cost':None,'money_cost_reason':'applicable account tariff unknown',
        'comparison_note':'API batch latency, amortized per-case time and full wall time are different quantities; cached decisions retain original batch timing'})
    diagnoses=[]
    for row in read('diagnostic_results.json'):
        if row['kind']=='learning' and row['level']=='1.0':
            diagnoses.append({'method':row['method'],'train_F1':row['train_metrics']['primary_f1'],
                'validation_F1':row['validation_metrics']['primary_f1'],
                'gap':row['train_metrics']['primary_f1']-row['validation_metrics']['primary_f1']})
    probes=read('name_probes.json')['probes'];names=[]
    for method in sorted({row['method'] for row in probes}):
        selected=[row for row in probes if row['method']==method]
        names.append({'method':method,'probes':len(selected),'alarm_flips':sum(row['alarm_changed'] for row in selected),
                      'native_class_flips':sum(row['class_changed'] for row in selected)})
    save_json(OUTPUT/'overfitting_evidence.json',{'train_validation_gaps':diagnoses,'neutral_name_sensitivity':names,
        'transfer':[{'method':row['method'],'direction':row['scope'],'metrics':row['metrics']} for row in transfer],
        'conclusions':['original model name sensitivity and external false alarms are directly measured',
            'name masking is not a guarantee of cross-application transfer','reduced data, fixed thresholds and domain shift can also explain bad transfer',
            'preview does not establish that overfitting is eliminated']})
    all_runs=local+llm+transfer+read('diagnostic_results.json')
    if (OUTPUT/'legacy_results.json').exists():all_runs+=read('legacy_results.json')['grouped_holdout']
    identifiers=sorted({row['run_id'] for row in all_runs})
    checked=[audit_run(store,run) for run in identifiers]
    plan=read('llm_plan.json')
    with store.connect() as connection:
        attempts=connection.execute('SELECT count(*) AS n FROM nirs.llm_requests WHERE origin_pilot_id=%s',(plan['pilot_id'],)).fetchone()['n']
    assert attempts<=plan['max_api_attempts']<=36
    save_json(OUTPUT/'audit_report.json',{'purpose':'preliminary_preview','checked_runs':checked,'new_API_attempts':attempts,
        'new_API_ceiling':plan['max_api_attempts'],'all_common_ID_sets_identical':True,'metrics_checked_with_sklearn':True,
        'historical_reference_label_space':'source_five_class','checks':['pinned_annotations','frozen_split_parts',
            'database_prediction_rows','independent_confusion_and_F1','referenced_artifact_checksums']})
    save_json(OUTPUT/'run_manifest.json',{'profile':'preliminary_preview','stages':[10,11,12],
        'common_observations':30,'runs':identifiers,'main_results':'analysis.json','audit':'audit_report.json',
        'full_run_deferred_to_stage':13})
    print(json.dumps({'audited_runs':len(identifiers),'new_API_calls':attempts,'common_observations':30,'errors':len(errors)}))


if __name__=='__main__':main()
