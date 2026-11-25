"""Frozen 300-observation main LLM experiment, two shot modes and context ablation."""

import argparse
from collections import Counter,defaultdict
import gzip
import json
import time

from detector_llm import MODELS,build_specification,call_once,create_client,prompt_observation,request_key
from experiment_store import save_json
from llm_store import prepare_pilot,reserve_request,finish_request
from stage10_store import load_test,persist_test
from storage import ROOT,Storage,canonical,digest,file_hash

OUTPUT=ROOT/'results/stage10/llm'
SOURCES=('main_llm.py','stage10_store.py','detector_llm.py','llm_store.py','data_contract.py','sql_features.py',
         'sql_views.py','normalize.py','storage.py','evaluation_metrics.py','experiment_store.py')


def signature():
    return {name:digest((ROOT/'src'/name).read_bytes().replace(b'\r\n',b'\n')) for name in SOURCES}


def write_requests(path,requests):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as stream:
        for request in requests:
            stream.write((canonical(request)+'\n').encode('utf-8'))


def read_requests(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def build_requests(samples,shots,models):
    requests=[]
    for model in models:
        for input_mode in ('sql_context','sql_only'):
            selected=samples if input_mode=='sql_context' else [row for row in samples if row['task']!='sqli']
            demos=[(row['input'] if input_mode=='sql_context' else row['input']['sql'],row['target']) for row in shots]
            for mode in ('zero_shot','few_shot'):
                for index in range(0,len(selected),model['batch_size']):
                    batch=selected[index:index+model['batch_size']]
                    observations=[prompt_observation(row['alias'],row['payload'] if input_mode=='sql_context' else row['payload']['sql']['text']) for row in batch]
                    spec=build_specification(model,mode,observations,demos)
                    spec.update(evaluation_input_mode=input_mode,ordered_observation_ids=[row['observation_id'] for row in batch],
                        few_shot_observation_ids=[row['observation_id'] for row in shots] if mode=='few_shot' else [])
                    requests.append({'model_name':model['name'],'input_mode':input_mode,'mode':mode,
                        'batch':index//model['batch_size'],'observation_ids':spec['ordered_observation_ids'],
                        'specification':spec,'request_key':request_key(spec)})
    return requests


def prepare(store):
    protocol=json.loads((ROOT/'results/stage10/pooled_protocol.json').read_text(encoding='utf-8'))
    selected=json.loads((ROOT/'results/stage9/main_sample_300.json').read_text(encoding='utf-8'))['observations']
    selected=sorted(selected,key=lambda row:digest(('stage10-order:'+row['observation_id']).encode()))
    if len(selected)!=300 or Counter((row['task'],row['target']) for row in selected)!=Counter({(task,label):50 for task in ('sqli','privilege','exfiltration') for label in (0,1)}):
        raise ValueError('The approved 300-observation sample differs')
    parent=protocol['parent_scope']
    records=load_test(store,protocol['dataset_version_id'],parent['split_id'],0,[row['observation_id'] for row in selected])
    samples=[]
    for index,(entry,record) in enumerate(zip(selected,records)):
        if record['truth']!=entry['target']:
            raise ValueError('Main-sample reference changed')
        samples.append({**entry,'alias':f'q{index+1:03d}','payload':record['payload'],
                        'annotation_id':record['annotation_id'],'representation_tags':record['tags']})
    pilot=json.loads((ROOT/'results/stage9/pilot_inputs.json').read_text(encoding='utf-8'))
    shots=pilot['few_shot_candidates']
    if {row['group_id'] for row in samples}&{row['group_id'] for group in pilot.values() for row in group}:
        raise ValueError('Main test shares groups with development observations or demos')
    models=[next(row for row in MODELS if row['name']==name) for name in ('gptoss120','glm53','kimi31pro')]
    requests=build_requests(samples,shots,models)
    if len(requests)!=768 or len({row['request_key'] for row in requests})!=768:
        raise ValueError('Approved main API ceiling or unique request identity differs')
    plan={'stage':10,'purpose':'frozen_main_test','sample_observations':300,'models':models,'max_api_attempts':768,
        'automatic_retries':0,'unavailable_model_policy':'stop_model_continue_others_without_substitution',
        'dataset_version_id':protocol['dataset_version_id'],'split_id':parent['split_id'],'fold':0,
        'code_sha256_lf':signature(),'sample_manifest_sha256':file_hash(ROOT/'results/stage9/main_sample_300.json'),
        'ordered_request_keys':[row['request_key'] for row in requests],
        'input_modes':{'sql_context':300,'sql_only':200},'few_shot_observation_ids':[row['observation_id'] for row in shots],
        'authorization':'user_approved_stage10_300_observations_768_calls','model_selection_uses_main_test':False}
    plan['code_artifacts']={name:store.register_artifact(ROOT/'src'/name) for name in SOURCES}
    plan['pilot_id']='main_'+digest(canonical(plan).encode())
    if (OUTPUT/'main_plan.json').exists():
        previous=json.loads((OUTPUT/'main_plan.json').read_text(encoding='utf-8'))
        if previous!=plan:
            raise ValueError('Main experiment is frozen; source or configuration changed')
    save_json(OUTPUT/'main_plan.json',plan)
    save_json(OUTPUT/'main_inputs.json',{'samples':samples,'few_shot_candidates':shots})
    write_requests(OUTPUT/'request_plan.jsonl.gz',requests)
    prepare_pilot(store,plan['pilot_id'],plan,768)
    return plan,samples,requests


def execute_requests(store,plan,requests,*,client_factory=create_client,caller=call_once,pacing=True):
    by_name={model['name']:model for model in plan['models']}
    clients={}; stopped_models={}; stopped_providers={}; last_call={}; outcomes=[]
    for request in requests:
        model=by_name[request['model_name']]; provider=model['provider']
        reason=stopped_models.get(model['name']) or stopped_providers.get(provider)
        if reason:
            outcomes.append({'request_key':request['request_key'],'status':'not_run','labels':None,'reason':reason,'new_api_attempt':False})
            continue
        if model['name'] not in clients:
            try:
                clients[model['name']]=client_factory(model)
            except ValueError:
                stopped_models[model['name']]='credential_unavailable'
                outcomes.append({'request_key':request['request_key'],'status':'not_run','labels':None,'reason':'credential_unavailable','new_api_attempt':False})
                continue
        key,result,new=reserve_request(store,plan['pilot_id'],request['specification'])
        if new:
            if pacing and provider in last_call:
                delay=model['sleep_s']-(time.monotonic()-last_call[provider])
                if delay>0:
                    time.sleep(delay)
            result=caller(clients[model['name']],request['specification'])
            finish_request(store,key,result)
            last_call[provider]=time.monotonic()
            print(f"{model['name']}/{request['input_mode']}/{request['mode']}/{request['batch']+1}: {result['status']}",flush=True)
        outcomes.append({'request_key':key,**result,'new_api_attempt':new})
        if result.get('stop_model'):
            stopped_models[model['name']]='model_stopped_after:'+str(result.get('parse_error') or result.get('http_status') or result.get('error_type'))
        if result.get('stop_provider'):
            stopped_providers[provider]='provider_stopped_after:'+str(result.get('http_status'))
        if new:
            save_json(OUTPUT/'progress.json',{'completed_request_positions':len(outcomes),'planned_request_positions':len(requests),
                'new_API_attempts_this_invocation':sum(row['new_api_attempt'] for row in outcomes),
                'stopped_models':stopped_models,'stopped_providers':stopped_providers})
    return outcomes


def persist(store,plan,samples,requests,outcomes):
    by_id={row['observation_id']:row for row in samples}
    groups=defaultdict(list)
    for request,outcome in zip(requests,outcomes):
        if request['request_key']!=outcome['request_key']:
            raise ValueError('Main request/result alignment differs')
        labels=outcome['labels'] if outcome['status']=='ok' else [None]*len(request['observation_ids'])
        if len(labels)!=len(request['observation_ids']):
            raise ValueError('Main batch response length differs')
        for oid,label in zip(request['observation_ids'],labels):
            entry=by_id[oid]
            groups[(request['model_name'],request['input_mode'],request['mode'])].append({
                'observation_id':oid,'split_id':plan['split_id'],'fold':0,'truth':entry['target'],'prediction':label,
                'status':outcome['status'],'metadata':{'request_key':request['request_key'],'task':entry['task'],
                    'representation_tags':entry['representation_tags']}})
    files={'code/'+name:store.get_artifact(aid) for name,aid in plan['code_artifacts'].items()}
    files.update({'main_plan.json':OUTPUT/'main_plan.json','main_inputs.json':OUTPUT/'main_inputs.json',
                  'request_plan.jsonl.gz':OUTPUT/'request_plan.jsonl.gz'})
    reports=[]
    for (model,input_mode,mode),rows in groups.items():
        if len(rows)!=plan['input_modes'][input_mode]:
            raise ValueError('Main comparison does not cover its full frozen sample')
        config={'stage':10,'role':'final_test','main_id':plan['pilot_id'],'model':model,'input_mode':input_mode,
            'shot_mode':mode,'code_sha256_lf':plan['code_sha256_lf'],'ordered_request_keys':plan['ordered_request_keys'],
            'observations_sha256':digest(canonical([row['observation_id'] for row in rows]).encode())}
        reports.append(persist_test(store,plan['dataset_version_id'],'LLM '+model+' '+input_mode+' '+mode,config,rows,files))
    with store.connect() as connection:
        calls=connection.execute('SELECT * FROM nirs.llm_requests WHERE origin_pilot_id=%s',(plan['pilot_id'],)).fetchall()
    attempts=sum(row['api_attempts'] for row in calls)
    if attempts>768:
        raise ValueError('Main API budget exceeded')
    summaries=[]
    for model in plan['models']:
        chosen=[row for row in calls if row['specification']['model']==model['model']]
        summaries.append({'name':model['name'],'api_attempts':len(chosen),
            'statuses':dict(Counter((row['result'] or {}).get('status','interrupted') for row in chosen)),
            'observed_tokens':{key:sum(((row['result'] or {}).get('usage') or {}).get(key) or 0 for row in chosen)
                for key in ('prompt_tokens','completion_tokens','total_tokens')},
            'usage_reported_calls':sum(bool((row['result'] or {}).get('usage')) for row in chosen),
            'elapsed_api_s':sum((row['result'] or {}).get('elapsed_s') or 0 for row in chosen)})
    report={'stage':10,'main_id':plan['pilot_id'],'role':'final_test','actual_API_attempts':attempts,'API_ceiling':768,
        'new_API_attempts_this_invocation':sum(row['new_api_attempt'] for row in outcomes),'models':summaries,'runs':reports,
        'money_cost':None,'money_cost_reason':'account-specific tariff unknown; measured calls/tokens provided'}
    save_json(OUTPUT/'main_report.json',report)
    write_requests(OUTPUT/'request_outcomes.jsonl.gz',outcomes)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); store=Storage()
    plan,samples,requests=prepare(store)
    if not args.execute:
        print(json.dumps({'main_id':plan['pilot_id'],'API_ceiling':768,'classifications':3000,'network_calls':0}))
        return
    with store.connect(autocommit=True) as lease:
        locked=lease.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked',(plan['pilot_id'],)).fetchone()['locked']
        if not locked:
            raise ValueError('The main experiment is already running')
        try:
            outcomes=execute_requests(store,plan,requests)
            if signature()!=plan['code_sha256_lf']:
                raise ValueError('Main LLM source changed while running')
            report=persist(store,plan,samples,requests,outcomes)
        finally:
            lease.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(plan['pilot_id'],))
    print(json.dumps({'actual_API_attempts':report['actual_API_attempts'],'runs':len(report['runs'])}))


if __name__=='__main__':
    main()
