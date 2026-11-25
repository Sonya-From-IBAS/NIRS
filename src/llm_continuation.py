"""The explicitly authorized GLM-5.3 continuation: eight cached, twelve new batches."""

import argparse
from copy import deepcopy
import json

import llm_pilot
from detector_llm import MODELS,request_key
from experiment_store import save_json
from llm_store import prepare_pilot,reserve_request
from storage import ROOT,Storage,canonical,digest,file_hash

SOURCE = ROOT/'results/stage9/glm53'
OUTPUT = ROOT/'results/stage9/glm53_continued'


def validate_selection(previous,outcomes):
    if len(previous)!=20 or len(outcomes)!=20:
        raise ValueError('Expected the frozen twenty-batch GLM pilot')
    requests,actions = [],[]
    for index,(request,outcome) in enumerate(zip(previous,outcomes)):
        spec=request['specification']
        mode='zero_shot' if index<10 else 'few_shot'
        if (request['model_name']!='glm53' or request['mode']!=mode or request['batch']!=index%10
            or spec['model']!='hf:zai-org/GLM-5.3' or spec['provider']!='synthetic'
            or spec['timeout_s']!=90 or request['request_key']!=request_key(spec)
            or outcome['request_key']!=request['request_key']):
            raise ValueError('Frozen GLM request identity differs')
        changed=deepcopy(request)
        if index<8:
            if outcome['status']!='ok':
                raise ValueError('The eight reused batches must be successful')
            action='reuse_successful_response'
        else:
            if index==8:
                if outcome['status']!='api_error' or outcome.get('error_type')!='APITimeoutError':
                    raise ValueError('Only the original timeout may be retried once')
                action='explicit_timeout_retry'
                changed['specification']['retry_of_request_key']=request['request_key']
            else:
                if outcome['status']!='not_run':
                    raise ValueError('Remaining batches must be previously unattempted')
                action='first_attempt'
                changed['specification']['continues_request_key']=request['request_key']
            changed['specification']['timeout_s']=180
            changed['request_key']=request_key(changed['specification'])
        actions.append({'old_request_key':request['request_key'],'selected_request_key':changed['request_key'],
                        'mode':mode,'batch':request['batch'],'action':action})
        requests.append(changed)
    return requests,actions


def prepare(store):
    if (OUTPUT/'pilot_plan.json').exists():
        read_saved=lambda name:json.loads((OUTPUT/name).read_text(encoding='utf-8'))
        saved=read_saved('pilot_plan.json')
        with store.connect() as connection:
            existing=connection.execute('SELECT configuration FROM nirs.llm_pilots WHERE pilot_id=%s',(saved['pilot_id'],)).fetchone()
        if not existing or existing['configuration']!=saved or read_saved('pilot_inputs.json')!=llm_pilot.load_pilot(store):
            raise ValueError('Saved continuation differs from the immutable database')
        return saved,read_saved('pilot_inputs.json'),read_saved('request_plan.json')
    read=lambda name:json.loads((SOURCE/name).read_text(encoding='utf-8'))
    previous_plan=read('pilot_plan.json')
    previous,outcomes=read('request_plan.json'),read('request_outcomes.json')
    cohorts=read('pilot_inputs.json')
    if cohorts!=llm_pilot.load_pilot(store):
        raise ValueError('Pilot inputs or pinned references changed')
    with store.connect() as connection:
        original=connection.execute('SELECT configuration FROM nirs.llm_pilots WHERE pilot_id=%s',
                                    (previous_plan['pilot_id'],)).fetchone()
        rows=connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=ANY(%s)',
                                 ([row['request_key'] for row in previous],)).fetchall()
    if not original or original['configuration']!=previous_plan or previous_plan['ordered_request_keys']!=[row['request_key'] for row in previous]:
        raise ValueError('Archived GLM configuration differs from the immutable database')
    cached={row['request_key']:row for row in rows}
    if len(cached)!=9:
        raise ValueError('Expected exactly the original nine attempts')
    for index,request in enumerate(previous):
        row=cached.get(request['request_key'])
        if index<9:
            if not row or row['state']!='complete' or row['specification']!=request['specification']:
                raise ValueError('Original cache differs')
            saved=json.loads(store.get_artifact(row['response_artifact']).read_text(encoding='utf-8'))
            if saved!={'request_key':request['request_key'],'result':row['result']}:
                raise ValueError('Original response artifact differs')
            if any(outcomes[index].get(name)!=value for name,value in row['result'].items()):
                raise ValueError('Original outcome differs from stored response')
        elif row:
            raise ValueError('A previously unattempted batch already has an original request')
    requests,actions=validate_selection(previous,outcomes)
    model=dict(next(row for row in MODELS if row['name']=='glm53'),timeout_s=180)
    plan=deepcopy(previous_plan)
    plan.pop('pilot_id')
    plan.update(models=[model],max_api_attempts=12,planned_request_count=20,
        code_sha256_lf=llm_pilot.signature(),ordered_request_keys=[row['request_key'] for row in requests],
        completion_token_upper_bound=12*4096,
        continuation={'authorization':'user_approved_finish_GLM_2026_11_25','source_directory':SOURCE.relative_to(ROOT).as_posix(),
            'source_pilot_id':previous_plan['pilot_id'],'source_request_plan_sha256':file_hash(SOURCE/'request_plan.json'),
            'reused_successful_batches':8,'new_unattempted_batches':11,'explicit_timeout_retries':1,
            'stage_attempts_before':38,'stage_attempts_ceiling':50,'actions':actions})
    plan['code_artifacts']={name:store.register_artifact(ROOT/'src'/name) for name in llm_pilot.SOURCES}
    plan['pilot_id']='pilot_'+digest(canonical(plan).encode())
    for name,data in (('pilot_plan.json',plan),('pilot_inputs.json',cohorts),('request_plan.json',requests),('prompts.json',read('prompts.json'))):
        save_json(OUTPUT/name,data)
    prepare_pilot(store,plan['pilot_id'],plan,12)
    # Reused successes must exist in the cache before the network runner can see them.
    for request in requests[:8]:
        _,result,new=reserve_request(store,plan['pilot_id'],request['specification'])
        if new or result['status']!='ok':
            raise ValueError('Successful batches cannot be reissued')
    return plan,cohorts,requests


def stopped_outcomes(store,plan,requests):
    """Read a stopped iteration without reserving requests or creating API clients."""
    with store.connect() as connection:
        rows=connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=ANY(%s)',(plan['ordered_request_keys'],)).fetchall()
    cached={row['request_key']:row for row in rows}
    outcomes=[]
    for request in requests:
        key=request['request_key']; row=cached.get(key)
        if row:
            if row['state']!='complete' or row['specification']!=request['specification']:
                raise ValueError('Stopped continuation has an unfinished or changed request')
            saved=json.loads(store.get_artifact(row['response_artifact']).read_text(encoding='utf-8'))
            if saved!={'request_key':key,'result':row['result']}:
                raise ValueError('Stopped response artifact differs')
            outcomes.append({'request_key':key,**row['result'],'new_api_attempt':False})
        else:
            outcomes.append({'request_key':key,'status':'not_run','labels':None,
                             'reason':'stopped_after_truncated_response','new_api_attempt':False})
    return outcomes


def validate_repair_selection(previous,outcomes,proposal):
    if len(previous)!=20 or len(outcomes)!=20:
        raise ValueError('Expected the twenty original GLM batches')
    requests,actions,changes=[],[],[]
    for request,outcome in zip(previous,outcomes):
        if (request['model_name']!='glm53' or request['specification']['model']!='hf:zai-org/GLM-5.3'
            or request['request_key']!=request_key(request['specification']) or outcome['request_key']!=request['request_key']):
            raise ValueError('Original repair request identity differs')
        selected=deepcopy(request)
        if outcome['status']=='ok':
            action='reuse_successful_response'
        else:
            if outcome['status']=='parse_error' and outcome.get('finish_reason')=='length':
                action='explicit_truncated_response_retry'
            elif outcome['status']=='not_run':
                action='first_attempt'
            else:
                raise ValueError('Only truncated or unattempted batches were authorized')
            spec=selected['specification']
            spec['parameters']['max_tokens']=8192
            spec['timeout_s']=180
            spec.pop('retry_of_request_key',None)
            spec.pop('continues_request_key',None)
            spec['retry_of_request_key' if action=='explicit_truncated_response_retry' else 'continues_request_key']=request['request_key']
            selected['request_key']=request_key(spec)
            changes.append({'prior_request_key':request['request_key'],'mode':request['mode'],'batch':request['batch'],
                'observation_ids':request['observation_ids'],'action':action,'specification':spec,
                'proposed_request_key':selected['request_key']})
        requests.append(selected)
        actions.append({'old_request_key':request['request_key'],'selected_request_key':selected['request_key'],
                        'mode':request['mode'],'batch':request['batch'],'action':action})
    if (len(changes)!=5 or sum(row['action']=='explicit_truncated_response_retry' for row in changes)!=3
        or proposal['requests']!=changes or proposal['new_API_ceiling']!=5
        or proposal['max_tokens']!=8192 or proposal['timeout_s']!=180):
        raise ValueError('Repair differs from the five-call proposal approved by the user')
    return requests,actions


def prepare_repair(store):
    source=ROOT/'results/stage9/glm53_continued'
    read=lambda name:json.loads((source/name).read_text(encoding='utf-8'))
    prior=read('pilot_plan.json')
    cohorts=read('pilot_inputs.json')
    previous,outcomes,proposal=read('request_plan.json'),read('request_outcomes.json'),read('repair_proposal.json')
    if cohorts!=llm_pilot.load_pilot(store):
        raise ValueError('Pinned pilot inputs changed')
    with store.connect() as connection:
        immutable=connection.execute('SELECT configuration FROM nirs.llm_pilots WHERE pilot_id=%s',(prior['pilot_id'],)).fetchone()
    if not immutable or immutable['configuration']!=prior or prior['ordered_request_keys']!=[row['request_key'] for row in previous]:
        raise ValueError('Repair source differs from the immutable pilot')
    # Only completed cache rows are read; missing rows remain unattempted.
    saved=stopped_outcomes(store,prior,previous)
    if any({key:value for key,value in actual.items() if key!='new_api_attempt'}!=
           {key:value for key,value in expected.items() if key!='new_api_attempt'} for actual,expected in zip(saved,outcomes)):
        raise ValueError('Repair source outcomes differ from the response cache')
    requests,actions=validate_repair_selection(previous,outcomes,proposal)
    if (OUTPUT/'pilot_plan.json').exists():
        saved_plan=json.loads((OUTPUT/'pilot_plan.json').read_text(encoding='utf-8'))
        with store.connect() as connection:
            immutable=connection.execute('SELECT configuration FROM nirs.llm_pilots WHERE pilot_id=%s',(saved_plan['pilot_id'],)).fetchone()
        if not immutable or immutable['configuration']!=saved_plan or saved_plan['ordered_request_keys']!=[row['request_key'] for row in requests]:
            raise ValueError('Saved repair configuration differs')
        return saved_plan,cohorts,requests
    plan=deepcopy(prior)
    plan.pop('pilot_id')
    plan.update(models=[dict(next(row for row in MODELS if row['name']=='glm53'),timeout_s=180)],
        max_api_attempts=5,planned_request_count=20,code_sha256_lf=llm_pilot.signature(),
        ordered_request_keys=[row['request_key'] for row in requests],completion_token_upper_bound=5*8192,
        continuation={'selection_kind':'token_limit_repair','authorization':'user_approved_8192_token_repair_2026_11_25',
            'source_directory':source.relative_to(ROOT).as_posix(),'source_pilot_id':prior['pilot_id'],
            'source_request_plan_sha256':file_hash(source/'request_plan.json'),
            'approved_proposal_sha256':file_hash(source/'repair_proposal.json'),
            'reused_successful_batches':15,'new_unattempted_batches':2,'explicit_truncation_retries':3,
            'stage_attempts_before':48,'stage_attempts_ceiling':53,'actions':actions})
    plan['code_artifacts']={name:store.register_artifact(ROOT/'src'/name) for name in llm_pilot.SOURCES}
    plan['pilot_id']='pilot_'+digest(canonical(plan).encode())
    for name,data in (('pilot_plan.json',plan),('pilot_inputs.json',cohorts),('request_plan.json',requests),('prompts.json',read('prompts.json'))):
        save_json(OUTPUT/name,data)
    prepare_pilot(store,plan['pilot_id'],plan,5)
    return plan,cohorts,requests


def execute_repair(store,plan,requests):
    pending=[request for request,action in zip(requests,plan['continuation']['actions'])
             if action['action']!='reuse_successful_response']
    executed=llm_pilot.request_results(store,plan,pending)
    # A failed new batch stops new calls, while all fifteen old successes stay in the report.
    outcomes=stopped_outcomes(store,plan,requests)
    new_flags={row['request_key']:row['new_api_attempt'] for row in executed}
    for row in outcomes:
        row['new_api_attempt']=new_flags.get(row['request_key'],False)
    errors=[row['request_key'] for row in executed if row['status'] in ('api_error','parse_error')]
    if errors:
        save_json(OUTPUT/'stop_record.json',{'reason':'API_or_parse_error','failed_request_keys':errors,
                                          'automatic_retries':0,'successful_cached_responses_preserved':True})
    return outcomes


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--repair',action='store_true',help='Approved five-call repair with 8192 tokens')
    args=parser.parse_args()
    store=Storage()
    if args.repair:
        OUTPUT=ROOT/'results/stage9/glm53_8192'
        plan,cohorts,requests=prepare_repair(store)
    else:
        plan,cohorts,requests=prepare(store)
    if not args.execute:
        print(json.dumps({'pilot_id':plan['pilot_id'],'cached_batches':plan['continuation']['reused_successful_batches'],
                          'new_API_ceiling':plan['max_api_attempts'],'timeout_s':180,'network_calls':0}))
        return
    with store.connect(autocommit=True) as lease:
        locked=lease.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked',(plan['pilot_id'],)).fetchone()['locked']
        if not locked:
            raise ValueError('This continuation is already running')
        try:
            if (OUTPUT/'stop_record.json').exists():
                outcomes=stopped_outcomes(store,plan,requests)
            else:
                if llm_pilot.signature()!=plan['code_sha256_lf']:
                    raise ValueError('Source changed; authorize a new iteration before API calls')
                outcomes=execute_repair(store,plan,requests) if args.repair else llm_pilot.request_results(store,plan,requests)
                if llm_pilot.signature()!=plan['code_sha256_lf']:
                    raise ValueError('Continuation source changed during execution')
            llm_pilot.OUTPUT=OUTPUT
            report=llm_pilot.persist_results(store,plan,cohorts,requests,outcomes)
        finally:
            lease.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(plan['pilot_id'],))
    print(json.dumps({'continuation_API_attempts':report['origin_pilot_api_attempts'],
        'coverage':{row['mode']:row['metrics']['decision_coverage'] for row in report['comparisons']}}))


if __name__=='__main__':
    main()
