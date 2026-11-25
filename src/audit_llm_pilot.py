"""Offline verification of pilot inputs, request accounting and saved decisions."""

from collections import Counter
import argparse
import json
from pathlib import Path

from detector_llm import parse_batch_response,request_key
from evaluation_metrics import classification_report
from experiment_store import save_json
from storage import ROOT,Storage,canonical,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/stage9')
    args=parser.parse_args()
    store=Storage()
    folder=args.directory
    plan=json.loads((folder/'pilot_plan.json').read_text(encoding='utf-8'))
    requests=json.loads((folder/'request_plan.json').read_text(encoding='utf-8'))
    outcomes=json.loads((folder/'request_outcomes.json').read_text(encoding='utf-8'))
    report=json.loads((folder/'pilot_report.json').read_text(encoding='utf-8'))
    cohort=json.loads((folder/'pilot_inputs.json').read_text(encoding='utf-8'))
    # The archived sources, rather than today's model roster, define old runs.
    ceiling=plan['max_api_attempts']
    request_count=plan.get('planned_request_count',ceiling)
    assert len(requests)==len(outcomes)==request_count
    assert plan['ordered_request_keys']==[item['request_key'] for item in requests]
    with store.connect() as connection:
        job=connection.execute('SELECT * FROM nirs.llm_pilots WHERE pilot_id=%s',(plan['pilot_id'],)).fetchone()
        calls=connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=ANY(%s)',(plan['ordered_request_keys'],)).fetchall()
    assert job['configuration']==plan and job['max_api_attempts']==ceiling
    assert sum(row['api_attempts'] for row in calls)==report['actual_api_attempts']<=request_count
    origin_attempts=sum(row['api_attempts'] for row in calls if row['origin_pilot_id']==plan['pilot_id'])
    assert origin_attempts<=ceiling
    if 'continuation' in plan:
        assert report['origin_pilot_api_attempts']==origin_attempts
        from llm_continuation import validate_selection,validate_repair_selection
        prior_folder=ROOT/plan['continuation']['source_directory']
        previous=json.loads((prior_folder/'request_plan.json').read_text(encoding='utf-8'))
        previous_outcomes=json.loads((prior_folder/'request_outcomes.json').read_text(encoding='utf-8'))
        if plan['continuation'].get('selection_kind')=='token_limit_repair':
            proposal=json.loads((prior_folder/'repair_proposal.json').read_text(encoding='utf-8'))
            assert digest((prior_folder/'repair_proposal.json').read_bytes())==plan['continuation']['approved_proposal_sha256']
            expected,actions=validate_repair_selection(previous,previous_outcomes,proposal)
        else:
            expected,actions=validate_selection(previous,previous_outcomes)
        assert requests==expected and plan['continuation']['actions']==actions
        assert plan['continuation']['stage_attempts_before']+origin_attempts<=plan['continuation']['stage_attempts_ceiling']
    cached={row['request_key']:row for row in calls}
    references={row['observation_id']:row for row in cohort['pilot']}
    aligned={}
    for request,outcome in zip(requests,outcomes):
        key=request_key(request['specification'])
        assert key==request['request_key']==outcome['request_key']
        assert request['specification']['sdk_max_retries']==0
        assert request['specification']['ordered_observation_ids']==request['observation_ids']
        if outcome['status']!='not_run':
            row=cached[key]
            assert row['state']=='complete'
            assert row['specification']==request['specification']
            saved_request=json.loads(store.get_artifact(row['request_artifact']).read_text(encoding='utf-8'))
            saved_response=json.loads(store.get_artifact(row['response_artifact']).read_text(encoding='utf-8'))
            assert saved_request==request['specification']
            assert saved_response=={'request_key':key,'result':row['result']}
            assert all(outcome.get(name)==value for name,value in row['result'].items())
            if outcome['status']=='ok':
                assert outcome['finish_reason']=='stop'
                assert parse_batch_response(outcome['final_content'],request['specification']['batch_aliases'])==outcome['labels']
        labels=outcome.get('labels') if outcome['status']=='ok' else [None]*len(request['observation_ids'])
        for oid,value in zip(request['observation_ids'],labels):
            aligned[(request['model_name'],request['mode'],oid)]=(value,outcome['status'],key)
    assert len(aligned)==len(plan['models'])*2*len(plan['pilot_ids'])
    checked=[]
    for run in report['runs']:
        with store.connect() as connection:
            database=connection.execute('SELECT * FROM nirs.experiment_runs WHERE run_id=%s',(run['run_id'],)).fetchone()
            rows=connection.execute('SELECT * FROM nirs.predictions WHERE run_id=%s ORDER BY observation_id',(run['run_id'],)).fetchall()
            metrics=connection.execute("SELECT value FROM nirs.metrics WHERE run_id=%s AND metric_name='classification_report'",(run['run_id'],)).fetchone()['value']
        assert database['configuration']['purpose']=='llm_development_pilot' and database['status']=='complete'
        expected={row['observation_id'] for row in cohort['pilot'] if row['dataset_version_id']==run['dataset_version_id']}
        assert {row['observation_id'] for row in rows}==expected
        for row in rows:
            expected_value,status,key=aligned[(run['model'],run['mode'],row['observation_id'])]
            assert row['protocol']=='inner_validation'
            assert row['predicted_label']==expected_value and row['status']==status
            assert row['metadata']['request_key']==key
            assert row['true_label']==references[row['observation_id']]['target']
        actual=classification_report([row['true_label'] for row in rows],[row['predicted_label'] for row in rows],statuses=[row['status'] for row in rows])
        assert actual==metrics==run['metrics']
        checked.append(run['run_id'])
    for comparison in report['comparisons']:
        values=[aligned[(comparison['model'],comparison['mode'],oid)] for oid in plan['pilot_ids']]
        assert classification_report([references[oid]['target'] for oid in plan['pilot_ids']],
                                    [value[0] for value in values],statuses=[value[1] for value in values])==comparison['metrics']
    for name,aid in plan['code_artifacts'].items():
        assert digest(store.get_artifact(aid).read_bytes().replace(b'\r\n',b'\n'))==plan['code_sha256_lf'][name]
    integrity=store.verify_artifacts()
    assert not integrity['failures'] and not integrity['not_ready']
    result={'stage':9,'pilot_id':plan['pilot_id'],'checked_runs':checked,'planned_classifications':len(aligned),
            'api_attempts':report['actual_api_attempts'],'origin_pilot_api_attempts':origin_attempts,'ceiling':ceiling,'artifact_integrity':integrity,
            'decision_status_counts':dict(Counter(value[1] for value in aligned.values())),
            'checks':['strict_final_JSON_reparse','ordered_batch_cache_keys','durable_API_budget','pinned_pilot_references',
                      'database_prediction_alignment','metrics_recomputed_from_rows','archived_source','artifact_checksums']}
    save_json(folder/'pilot_audit.json',result)
    print(json.dumps({**result,'checked_runs':len(checked)}))


if __name__=='__main__':
    main()
