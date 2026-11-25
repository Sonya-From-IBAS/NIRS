"""Main-test identity, context ablation, transport isolation and failure policy."""

from collections import Counter,defaultdict
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import main_llm
from detector_llm import MODELS


class MainExperimentTests(unittest.TestCase):
    def fixture(self):
        samples=[{'observation_id':f'obs{i}','alias':f'q{i+1:03d}','task':('sqli','privilege','exfiltration')[i%3],
            'target':i%2,'source':'DO_NOT_SEND','payload':{'sql':{'text':f'SELECT {i}'},'context':{
                'dialect':{'availability':'known','available_before_decision':True,'value':'postgresql'}}}} for i in range(300)]
        shots=[{'observation_id':f'shot{i}','input':{'sql':f'SELECT {i+1000}','context':{
            'dialect':{'availability':'known','value':'postgresql'}}},'target':i%2} for i in range(6)]
        models=[next(row for row in MODELS if row['name']==name) for name in ('gptoss120','glm53','kimi31pro')]
        return samples,shots,models

    def test_exact_768_requests_and_identical_comparison_IDs(self):
        samples,shots,models=self.fixture()
        requests=main_llm.build_requests(samples,shots,models)
        self.assertEqual(len(requests),768)
        self.assertEqual(len({row['request_key'] for row in requests}),768)
        self.assertEqual(Counter(row['model_name'] for row in requests),{'gptoss120':334,'glm53':334,'kimi31pro':100})
        groups=defaultdict(list)
        for row in requests:
            groups[(row['model_name'],row['input_mode'],row['mode'])].extend(row['observation_ids'])
            self.assertNotIn('DO_NOT_SEND',json.dumps(row['specification']['messages']))
        for (_,input_mode,_),ids in groups.items():
            expected=[row['observation_id'] for row in samples if input_mode=='sql_context' or row['task']!='sqli']
            self.assertEqual(ids,expected)

    def test_sql_only_removes_context_from_both_queries_and_demos(self):
        samples,shots,models=self.fixture()
        requests=main_llm.build_requests(samples,shots,models[:1])
        selected=next(row for row in requests if row['input_mode']=='sql_only' and row['mode']=='few_shot')
        messages=selected['specification']['messages']
        demos=json.loads(messages[1]['content'].split('\n',1)[1])
        queries=json.loads(messages[-1]['content'].split('\n',1)[1])
        self.assertTrue(all(row['observation']['context']=={} for row in demos))
        self.assertTrue(all(row['context']=={} for row in queries))

    def test_bad_response_stops_model_and_model_timeouts_do_not_share_a_client(self):
        samples,shots,models=self.fixture()
        requests=[]
        for model,count in ((models[0],2),(models[1],1)):
            requests+=main_llm.build_requests(samples,shots,[model])[:count]
        calls=[]
        def factory(model):
            calls.append((model['name'],model.get('timeout_s',90)))
            return model['name']
        def caller(client,spec):
            return {'status':'parse_error','labels':None,'stop_model':True,'stop_provider':False,'parse_error':'length'} if client=='gptoss120' else {
                'status':'ok','labels':[0]*len(spec['batch_aliases']),'stop_model':False,'stop_provider':False}
        def reserve(store,pid,spec):
            return main_llm.request_key(spec),None,True
        with tempfile.TemporaryDirectory() as folder,patch('main_llm.OUTPUT',Path(folder)), \
             patch('main_llm.reserve_request',side_effect=reserve) as reserved,patch('main_llm.finish_request') as finished:
            results=main_llm.execute_requests(Mock(),{'pilot_id':'test','models':models},requests,
                client_factory=factory,caller=caller,pacing=False)
        self.assertEqual([row['status'] for row in results],['parse_error','not_run','ok'])
        self.assertEqual(calls,[('gptoss120',90),('glm53',180)])
        self.assertEqual(reserved.call_count,2)
        self.assertEqual(finished.call_count,2)


if __name__=='__main__':
    unittest.main()
