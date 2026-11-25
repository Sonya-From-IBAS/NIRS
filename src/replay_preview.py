"""Verify all selected cached LLM answers and rebuild the preview without network calls."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from detector_llm import parse_batch_response
from experiment_store import save_json
import main_llm
import preview_analysis
import preview_report
from storage import ROOT,Storage


def main():
    folder=ROOT/'results/preview';store=Storage()
    plan=json.loads((folder/'llm_plan.json').read_text(encoding='utf-8'))
    requests=json.loads((folder/'llm_requests.json').read_text(encoding='utf-8'))
    previous=json.loads((folder/'llm_outcomes.json').read_text(encoding='utf-8'))
    keys=plan['ordered_request_keys']+list({row['request_key'] for row in plan['cached_decisions']})
    with store.connect() as connection:
        calls=connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=ANY(%s)',(keys,)).fetchall()
        before=connection.execute('SELECT count(*) AS n FROM nirs.llm_requests').fetchone()['n']
    cached={row['request_key']:row for row in calls}
    for key,row in cached.items():
        assert row['state']=='complete'
        result=row['result']
        if result['status']=='ok':
            assert result['labels']==parse_batch_response(result['final_content'],row['specification']['batch_aliases'])
    for decision in plan['cached_decisions']:
        row=cached[decision['request_key']]
        position=row['specification']['ordered_observation_ids'].index(decision['observation_id'])
        assert row['result']['labels'][position]==decision['prediction']
    def forbidden(*args,**kwargs):raise AssertionError('Offline replay must never call an API')
    with TemporaryDirectory(dir=ROOT/'.local') as temporary,patch('main_llm.OUTPUT',Path(temporary)):
        outcomes=main_llm.execute_requests(store,plan,requests,client_factory=lambda model:object(),caller=forbidden,pacing=False)
    assert not any(row['new_api_attempt'] for row in outcomes)
    assert [{key:value for key,value in row.items() if key!='new_api_attempt'} for row in outcomes]==[
        {key:value for key,value in row.items() if key!='new_api_attempt'} for row in previous]
    with store.connect() as connection:
        after=connection.execute('SELECT count(*) AS n FROM nirs.llm_requests').fetchone()['n']
    assert before==after
    names=['analysis.json','summary_table.csv','error_cases.csv','timing_and_cost.json','overfitting_evidence.json','audit_report.json','report.md']
    names += [path.relative_to(folder).as_posix() for path in sorted((folder/'plots').glob('*'))]
    original={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    preview_analysis.main();preview_report.main()
    repeated={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    assert original==repeated,'Rebuilt report differs from the same pinned predictions'
    save_json(folder/'resume_check.json',{'network_caller_replaced_with_raising_stub':True,'new_API_calls':0,
        'LLM_request_count_unchanged':before==after,'cached_primary_decisions_checked':len(plan['cached_decisions']),
        'cached_preview_batches_checked':len(requests),'identical_rebuilt_files':len(names),
        'identical_JSON_CSV_markdown_PNG_SVG':True,'sha256':repeated})
    print('Offline replay passed: no new requests, all tables/reports/plots identical.')


if __name__=='__main__':main()
