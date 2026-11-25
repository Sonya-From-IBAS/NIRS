"""Small, explicitly preliminary common test; reuse completed GPT outputs by observation ID."""

import argparse
from collections import defaultdict,Counter
import json

from detector_llm import MODELS,build_specification,prompt_observation,request_key
from experiment_store import save_json
from llm_store import prepare_pilot
import main_llm
from stage10_store import persist_test
from storage import ROOT,Storage,canonical,digest,file_hash

OUTPUT=ROOT/'results/preview'
SOURCES=(*main_llm.SOURCES,'preview_llm.py')


def prepare(store):
    original=json.loads((ROOT/'results/stage10/llm/main_plan.json').read_text(encoding='utf-8'))
    full=json.loads((ROOT/'results/stage10/llm/main_inputs.json').read_text(encoding='utf-8'))
    # Prefixes of frozen metadata ranks, selected without inspecting predictions.
    samples=[row for row in full['samples'] if row['rank_in_stratum']<5]
    if len(samples)!=30 or Counter((row['task'],row['target']) for row in samples)!=Counter({(task,label):5 for task in ('sqli','privilege','exfiltration') for label in (0,1)}):
        raise ValueError('Preview must contain five examples of each frozen task/target stratum')
    selected={row['observation_id'] for row in samples};shots=full['few_shot_candidates']
    with store.connect() as connection:
        calls=connection.execute("SELECT * FROM nirs.llm_requests WHERE origin_pilot_id=%s AND state='complete'",(original['pilot_id'],)).fetchall()
    reused=[]
    for call in calls:
        spec=call['specification'];result=call['result']
        if result['status']!='ok' or spec['evaluation_input_mode']!='sql_context':
            continue
        saved=json.loads(store.get_artifact(call['response_artifact']).read_text(encoding='utf-8'))
        if saved!={'request_key':call['request_key'],'result':result}:
            raise ValueError('Original cached response differs')
        name=next(row['name'] for row in MODELS if row['model']==spec['model'] and row['provider']==spec['provider'])
        for oid,label in zip(spec['ordered_observation_ids'],result['labels']):
            if oid in selected:
                reused.append({'model_name':name,'mode':spec['mode'],'observation_id':oid,'prediction':label,
                    'status':'ok','request_key':call['request_key'],'origin_main_id':original['pilot_id']})
    known={(row['model_name'],row['mode'],row['observation_id']) for row in reused}
    models=[next(row for row in MODELS if row['name']==name) for name in ('gptoss120','glm53','kimi31pro')]
    requests=[]
    for model in models:
        for mode in ('zero_shot','few_shot'):
            missing=[row for row in samples if (model['name'],mode,row['observation_id']) not in known]
            for index in range(0,len(missing),model['batch_size']):
                batch=missing[index:index+model['batch_size']]
                spec=build_specification(model,mode,[prompt_observation(row['alias'],row['payload']) for row in batch],
                    [(row['input'],row['target']) for row in shots])
                spec.update(evaluation_input_mode='sql_context',ordered_observation_ids=[row['observation_id'] for row in batch],
                    few_shot_observation_ids=[row['observation_id'] for row in shots] if mode=='few_shot' else [])
                requests.append({'model_name':model['name'],'input_mode':'sql_context','mode':mode,'batch':index//model['batch_size'],
                    'observation_ids':spec['ordered_observation_ids'],'specification':spec,'request_key':request_key(spec)})
    if len(requests)>36:
        raise ValueError('The reduced profile must not exceed 36 new API calls')
    sig={name:digest((ROOT/'src'/name).read_bytes().replace(b'\r\n',b'\n')) for name in SOURCES}
    plan={'purpose':'preliminary_preview','stage':10,'observations':30,'models':models,'max_api_attempts':len(requests),
        'automatic_retries':0,'source_main_id':original['pilot_id'],'dataset_version_id':original['dataset_version_id'],
        'split_id':original['split_id'],'code_sha256_lf':sig,'ordered_request_keys':[row['request_key'] for row in requests],
        'selection':'first five frozen ranks per task/target, never prediction-based','cached_decisions':reused,
        'authorization':'user_reduced_scope_to_small_fixed_run_count','full_experiment_deferred_to_stage':13}
    plan['code_artifacts']={name:store.register_artifact(ROOT/'src'/name) for name in SOURCES}
    plan['pilot_id']='preview_'+digest(canonical(plan).encode())
    if (OUTPUT/'llm_plan.json').exists() and json.loads((OUTPUT/'llm_plan.json').read_text(encoding='utf-8'))!=plan:
        raise ValueError('Preview configuration is frozen')
    for name,data in (('llm_plan.json',plan),('common_inputs.json',{'samples':samples,'few_shot_candidates':shots}),('llm_requests.json',requests)):
        save_json(OUTPUT/name,data)
    prepare_pilot(store,plan['pilot_id'],plan,len(requests))
    return plan,samples,requests


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    store=Storage();plan,samples,requests=prepare(store)
    if not args.execute:
        print(json.dumps({'new_API_ceiling':len(requests),'reused_decisions':len(plan['cached_decisions']),'observations':30}));return
    main_llm.OUTPUT=OUTPUT
    outcomes=main_llm.execute_requests(store,plan,requests)
    by_key={(row['model_name'],row['mode'],row['observation_id']):row for row in plan['cached_decisions']}
    for request,outcome in zip(requests,outcomes):
        labels=outcome.get('labels') if outcome['status']=='ok' else [None]*len(request['observation_ids'])
        for oid,label in zip(request['observation_ids'],labels):
            by_key[(request['model_name'],request['mode'],oid)]={'prediction':label,'status':outcome['status'],'request_key':request['request_key']}
    files={'code/'+name:store.get_artifact(aid) for name,aid in plan['code_artifacts'].items()}
    files.update({'llm_plan.json':OUTPUT/'llm_plan.json','common_inputs.json':OUTPUT/'common_inputs.json','llm_requests.json':OUTPUT/'llm_requests.json'})
    runs=[]
    for model in plan['models']:
        for mode in ('zero_shot','few_shot'):
            rows=[]
            for entry in samples:
                result=by_key[(model['name'],mode,entry['observation_id'])]
                rows.append({'observation_id':entry['observation_id'],'split_id':plan['split_id'],'fold':0,'truth':entry['target'],
                    'prediction':result['prediction'],'status':result['status'],'metadata':{'request_key':result['request_key'],
                        'task':entry['task'],'representation_tags':entry['representation_tags']}})
            config={'stage':10,'role':'preliminary_preview','model':model['name'],'mode':mode,'input_mode':'sql_context',
                'preview_id':plan['pilot_id'],'code_sha256_lf':plan['code_sha256_lf'],'not_a_new_blind_test':True}
            runs.append(persist_test(store,plan['dataset_version_id'],'LLM '+model['name']+' '+mode,config,rows,files))
    with store.connect() as connection:
        calls=connection.execute('SELECT * FROM nirs.llm_requests WHERE origin_pilot_id=%s',(plan['pilot_id'],)).fetchall()
    save_json(OUTPUT/'llm_outcomes.json',outcomes)
    save_json(OUTPUT/'llm_results.json',{'purpose':'preliminary_preview','runs':runs,'new_API_attempts':len(calls),
        'new_API_ceiling':len(requests),'reused_decisions':len(plan['cached_decisions']),
        'observed_tokens':{key:sum(((row['result'] or {}).get('usage') or {}).get(key) or 0 for row in calls)
            for key in ('prompt_tokens','completion_tokens','total_tokens')},'cached_origin':plan['source_main_id']})
    print(json.dumps({'new_API_attempts':len(calls),'runs':len(runs),'complete':all(row['metrics']['decision_coverage']==1 for row in runs)}))


if __name__=='__main__':
    main()
