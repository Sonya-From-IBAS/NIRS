"""Frozen 30-observation pilot: at most 46 calls, no retries or substitutions."""

import argparse
from collections import Counter,defaultdict
import gzip
import json
import time
from importlib.metadata import version as package_version

from psycopg.types.json import Jsonb

from detector_llm import (MODELS,SYSTEM_PROMPT,VERSION,build_specification,call_once,create_client,
                          prompt_observation,request_key)
from evaluation_metrics import classification_report
from experiment_store import save_json
from llm_store import finish_request,prepare_pilot,reserve_request
from storage import ROOT,Storage,canonical,digest,file_hash

OUTPUT = ROOT/'results/stage9'
SOURCES = ('detector_llm.py','llm_store.py','llm_pilot.py','data_contract.py','sql_features.py',
           'sql_views.py','normalize.py','storage.py','evaluation_metrics.py','experiment_store.py')


def signature():
    return {name:digest((ROOT/'src'/name).read_bytes().replace(b'\r\n',b'\n')) for name in SOURCES}


def load_pilot(store):
    manifest = json.loads((ROOT/'results/stage6/llm_development.json').read_text(encoding='utf-8'))
    if manifest['purpose'] != 'development_only_not_final_test' or len(manifest['pilot']) != 30 or len(manifest['few_shot_candidates']) != 6:
        raise ValueError('The agreed pilot manifest differs')
    cohorts = {}
    for kind,expected_part in (('pilot','validation'),('few_shot_candidates','train')):
        values = []
        for entry in manifest[kind]:
            with store.connect() as connection:
                row = connection.execute('''SELECT o.sql_text,o.context_json,a.status,a.target_label,a.data,
                    s.part FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
                    JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
                    JOIN nirs.split_members s ON s.dataset_version_id=d.dataset_version_id AND s.observation_id=d.observation_id
                    WHERE d.dataset_version_id=%s AND d.observation_id=%s AND s.split_id=%s AND s.fold=%s''',
                    (entry['dataset_version_id'],entry['observation_id'],entry['split_id'],entry['fold'])).fetchone()
                group = connection.execute('''SELECT m.group_id FROM nirs.group_members m JOIN nirs.sample_groups g USING(group_id)
                    WHERE m.observation_id=%s AND g.kind='stage6_component' ''',(entry['observation_id'],)).fetchall()
            if not row or row['status'] != 'verified' or row['target_label'] != entry['target'] or row['part'] != expected_part or entry['part'] != expected_part:
                raise ValueError('Pilot membership or pinned target differs')
            if [item['group_id'] for item in group] != [entry['group_id']]:
                raise ValueError('Pilot group identity differs')
            payload = {'sql':{'text':row['sql_text']},'context':row['context_json']}
            values.append({**entry,'input':prompt_observation('temporary',payload),
                           'representation_tags':row['data'].get('representation_tags',[])})
        if len({item['observation_id'] for item in values}) != len(values):
            raise ValueError('Repeated observation in development manifest')
        cohorts[kind] = sorted(values,key=lambda item:digest(('stage9-order:'+item['observation_id']).encode()))
    groups = [{row['group_id'] for row in cohorts[kind]} for kind in ('pilot','few_shot_candidates')]
    with gzip.open(ROOT/'results/stage6/common_test_pool.jsonl.gz','rt',encoding='utf-8') as stream:
        test_groups = {json.loads(line)['group_id'] for line in stream}
    if groups[0]&groups[1] or (groups[0]|groups[1])&test_groups:
        raise ValueError('Pilot, training demonstrations or common test groups overlap')
    for index,entry in enumerate(cohorts['pilot']):
        entry['input']['id'] = f'q{index+1:03d}'
    for entry in cohorts['few_shot_candidates']:
        entry['input'].pop('id')
    return cohorts


def prepare(store):
    cohorts = load_pilot(store)
    shots = [(row['input'],row['target']) for row in cohorts['few_shot_candidates']]
    requests = []
    for model in MODELS:
        for mode in ('zero_shot','few_shot'):
            for index in range(0,30,model['batch_size']):
                rows = cohorts['pilot'][index:index+model['batch_size']]
                spec = build_specification(model,mode,[row['input'] for row in rows],shots)
                spec['ordered_observation_ids'] = [row['observation_id'] for row in rows]
                spec['few_shot_observation_ids'] = [row['observation_id'] for row in cohorts['few_shot_candidates']] if mode=='few_shot' else []
                requests.append({'model_name':model['name'],'mode':mode,'batch':index//model['batch_size'],
                                 'observation_ids':[row['observation_id'] for row in rows],
                                 'request_key':request_key(spec),'specification':spec})
    if len(requests) != 46:
        raise ValueError('The agreed API budget differs')
    plan = {'version':VERSION,'purpose':'development_only_not_final_test','models':list(MODELS),
            'max_api_attempts':46,'automatic_retries':0,'unavailable_model_policy':'stop_that_model_continue_others_without_substitution',
            'code_sha256_lf':signature(),'development_manifest_sha256':file_hash(ROOT/'results/stage6/llm_development.json'),
            'pilot_ids':[row['observation_id'] for row in cohorts['pilot']],
            'few_shot_ids':[row['observation_id'] for row in cohorts['few_shot_candidates']],
            'ordered_request_keys':[request['request_key'] for request in requests],
            'packages':{name:package_version(name) for name in ('openai','groq','psycopg','httpx')},
            'money_limit_or_tariff':None,'completion_token_upper_bound':6*512+40*4096}
    plan['code_artifacts'] = {name:store.register_artifact(ROOT/'src'/name) for name in SOURCES}
    pilot_id = 'pilot_'+digest(canonical(plan).encode())
    plan['pilot_id'] = pilot_id
    save_json(OUTPUT/'pilot_plan.json',plan)
    save_json(OUTPUT/'pilot_inputs.json',cohorts)
    save_json(OUTPUT/'request_plan.json',requests)
    save_json(OUTPUT/'prompts.json',{'version':VERSION,'system':SYSTEM_PROMPT,'few_shot_count':6,'output':'exact ID set and integer labels 0/1'})
    prepare_pilot(store,pilot_id,plan,46)
    return plan,cohorts,requests


def request_results(store,plan,requests,client_factory=create_client,caller=call_once,pacing=True):
    outcomes,clients,stopped_models,stopped_providers = [],{},{},{}
    by_name = {model['name']:model for model in MODELS}
    last_call = {}
    for request in requests:
        model = by_name[request['model_name']]
        provider = model['provider']
        blocked = stopped_models.get(model['name']) or stopped_providers.get(provider)
        if blocked:
            outcomes.append({'request_key':request['request_key'],'status':'not_run','labels':None,'reason':blocked,'new_api_attempt':False})
            continue
        if provider not in clients:
            try:
                clients[provider] = client_factory(model)
            except ValueError:
                stopped_providers[provider] = 'credential_unavailable'
                outcomes.append({'request_key':request['request_key'],'status':'not_run','labels':None,'reason':'credential_unavailable','new_api_attempt':False})
                continue
        if pacing and provider in last_call:
            delay = model['sleep_s']-(time.monotonic()-last_call[provider])
            if delay>0:
                time.sleep(delay)
        key,result,new = reserve_request(store,plan['pilot_id'],request['specification'])
        if new:
            result = caller(clients[provider],request['specification'])
            finish_request(store,key,result)
            last_call[provider] = time.monotonic()
            print(f"{model['name']}/{request['mode']}/{request['batch']+1}: {result['status']}",flush=True)
        outcomes.append({'request_key':key,**result,'new_api_attempt':new})
        if result.get('stop_model'):
            stopped_models[model['name']] = 'model_stopped_after:'+str(result.get('error_code') or result.get('http_status'))
        if result.get('stop_provider'):
            stopped_providers[provider] = 'provider_stopped_after:'+str(result.get('http_status'))
    return outcomes


def persist_results(store,plan,cohorts,requests,outcomes):
    by_id = {row['observation_id']:row for row in cohorts['pilot']}
    groups = defaultdict(list)
    for request,result in zip(requests,outcomes):
        if request['request_key'] != result['request_key']:
            raise ValueError('Request/result alignment differs')
        labels = result.get('labels') if result['status']=='ok' else [None]*len(request['observation_ids'])
        if not isinstance(labels,list) or len(labels) != len(request['observation_ids']):
            raise ValueError('Cached response label alignment differs')
        for oid,label in zip(request['observation_ids'],labels):
            if result['status']=='ok' and (type(label) is not int or label not in (0,1)):
                raise ValueError('Invalid cached binary label')
            groups[(request['model_name'],request['mode'])].append({'entry':by_id[oid],'prediction':label,
                'status':result['status'],'request_key':request['request_key']})
    code_artifacts = {'code/'+name:store.register_artifact(ROOT/'src'/name) for name in SOURCES}
    code_artifacts.update({'pilot_plan.json':store.register_artifact(OUTPUT/'pilot_plan.json'),
                          'pilot_inputs.json':store.register_artifact(OUTPUT/'pilot_inputs.json'),
                          'request_plan.json':store.register_artifact(OUTPUT/'request_plan.json'),
                          'prompts.json':store.register_artifact(OUTPUT/'prompts.json')})
    runs,comparisons = [],[]
    for (model,mode),values in groups.items():
        if len(values)!=30 or len({row['entry']['observation_id'] for row in values})!=30:
            raise ValueError('Every comparison must cover exactly the 30 planned IDs')
        metrics = classification_report([row['entry']['target'] for row in values],[row['prediction'] for row in values],statuses=[row['status'] for row in values])
        comparisons.append({'model':model,'mode':mode,'metrics':metrics})
        for dataset in sorted({row['entry']['dataset_version_id'] for row in values}):
            rows = [row for row in values if row['entry']['dataset_version_id']==dataset]
            split_ids = {row['entry']['split_id'] for row in rows}
            if len(split_ids)!=1:
                raise ValueError('Pilot dataset must use a single pinned inner split')
            configuration = {'version':VERSION,'pilot_id':plan['pilot_id'],'model':model,'mode':mode,
                'role':'inner_validation_used_for_selection','purpose':'llm_development_pilot',
                'split_id':split_ids.pop(),'ordered_observation_ids':[row['entry']['observation_id'] for row in rows],
                'code_sha256_lf':plan['code_sha256_lf'],'planned_requests':plan['ordered_request_keys']}
            run_id = 'run_'+digest(canonical({'dataset':dataset,'configuration':configuration}).encode())
            report = classification_report([row['entry']['target'] for row in rows],[row['prediction'] for row in rows],statuses=[row['status'] for row in rows])
            with store.write() as connection:
                existing = connection.execute('SELECT configuration FROM nirs.experiment_runs WHERE run_id=%s',(run_id,)).fetchone()
                if existing:
                    if existing['configuration'] != configuration:
                        raise ValueError('Existing pilot run configuration differs')
                    actual = connection.execute('SELECT observation_id,predicted_label,status FROM nirs.predictions WHERE run_id=%s',(run_id,)).fetchall()
                    if {row['observation_id']:(row['predicted_label'],row['status']) for row in actual} != {row['entry']['observation_id']:(row['prediction'],row['status']) for row in rows}:
                        raise ValueError('Existing pilot predictions differ')
                else:
                    connection.execute("INSERT INTO nirs.experiment_runs VALUES (%s,%s,%s,'binary','prepared',%s)",(run_id,dataset,'LLM '+model+' '+mode,Jsonb(configuration)))
                    store._copy(connection,'predictions',['run_id','dataset_version_id','split_id','observation_id','protocol','fold','true_label','predicted_label','status','metadata'],
                        [(run_id,dataset,row['entry']['split_id'],row['entry']['observation_id'],'inner_validation',row['entry']['fold'],row['entry']['target'],row['prediction'],row['status'],Jsonb({'pilot_id':plan['pilot_id'],'request_key':row['request_key']})) for row in rows])
                    connection.execute("INSERT INTO nirs.metrics VALUES (%s,'inner_validation',0,'classification_report',%s)",(run_id,Jsonb(report)))
                    for role,aid in code_artifacts.items():
                        connection.execute('INSERT INTO nirs.run_artifacts VALUES (%s,%s,%s,%s)',(run_id,aid,role,Jsonb({'stage':9})))
                    connection.execute("UPDATE nirs.experiment_runs SET status='complete' WHERE run_id=%s",(run_id,))
            runs.append({'run_id':run_id,'dataset_version_id':dataset,'model':model,'mode':mode,'metrics':report,'artifacts':code_artifacts})
    with store.connect() as connection:
        calls = connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=ANY(%s)',(plan['ordered_request_keys'],)).fetchall()
    usage = Counter()
    for call in calls:
        details = (call['result'] or {}).get('usage') or {}
        for key in ('prompt_tokens','completion_tokens','total_tokens'):
            if isinstance(details.get(key),int):
                usage[key] += details[key]
    summaries = []
    for model in MODELS:
        model_calls = [row for row in calls if row['specification']['model']==model['model'] and row['specification']['provider']==model['provider']]
        summaries.append({'name':model['name'],'requested_model':model['model'],'provider':model['provider'],
            'api_attempts':sum(row['api_attempts'] for row in model_calls),
            'actual_model_ids':sorted({row['result']['actual_model'] for row in model_calls if row['result'] and row['result'].get('actual_model')}),
            'call_status_counts':dict(Counter((row['result'] or {}).get('status','interrupted') for row in model_calls)),
            'observed_tokens':{key:sum(((row['result'] or {}).get('usage') or {}).get(key) or 0 for row in model_calls) for key in ('prompt_tokens','completion_tokens','total_tokens')},
            'elapsed_api_s':sum((row['result'] or {}).get('elapsed_s') or 0 for row in model_calls)})
    result = {'version':VERSION,'pilot_id':plan['pilot_id'],'role':'development_only_not_final_test',
              'planned_api_ceiling':46,'actual_api_attempts':sum(row['api_attempts'] for row in calls),
              'new_api_attempts_this_invocation':sum(bool(row['new_api_attempt']) for row in outcomes),
              'observed_tokens':dict(usage),'usage_reported_calls':sum(bool((row['result'] or {}).get('usage')) for row in calls),
              'money_cost':None,'money_cost_reason':'account-specific tariff not established; observed tokens and calls reported',
              'models':summaries,'comparisons':comparisons,'runs':runs,'main_experiment_budget':'pending_user_agreement_after_pilot'}
    if result['actual_api_attempts']>46:
        raise ValueError('Pilot exceeded its agreed API ceiling')
    save_json(OUTPUT/'pilot_report.json',result)
    save_json(OUTPUT/'request_outcomes.json',outcomes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Run/resume the agreed pilot; default prepares without network calls')
    args = parser.parse_args()
    store = Storage()
    store.migrate()
    plan,cohorts,requests = prepare(store)
    if not args.execute:
        print(json.dumps({'pilot_id':plan['pilot_id'],'planned_requests':len(requests),'planned_classifications':180,'network_calls':0}))
        return
    with store.connect(autocommit=True) as lease:
        locked = lease.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked',(plan['pilot_id'],)).fetchone()['locked']
        if not locked:
            raise ValueError('This pilot is already running')
        try:
            outcomes = request_results(store,plan,requests)
            if signature()!=plan['code_sha256_lf']:
                raise ValueError('Pilot source changed during execution')
            report = persist_results(store,plan,cohorts,requests,outcomes)
        finally:
            lease.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(plan['pilot_id'],))
    print(json.dumps({'actual_api_attempts':report['actual_api_attempts'],'observed_tokens':report['observed_tokens'],
                      'coverage':{row['model']+'/'+row['mode']:row['metrics']['decision_coverage'] for row in report['comparisons']}}))


if __name__ == '__main__':
    main()
