"""Strict parsing, final-content isolation and model-independent request identity."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import MagicMock,Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detector_llm import MODELS,build_specification,call_once,create_client,parse_batch_response,prompt_observation,request_key


class LLMTests(unittest.TestCase):
    def repair_fixture(self):
        folder=Path(__file__).resolve().parents[1]/'results/stage9/glm53_continued'
        return tuple(json.loads((folder/name).read_text(encoding='utf-8')) for name in ('request_plan.json','request_outcomes.json','repair_proposal.json'))

    def test_token_repair_preserves_fifteen_successes_and_matches_the_approved_five_requests(self):
        from llm_continuation import validate_repair_selection
        previous,outcomes,proposal=self.repair_fixture()
        requests,actions=validate_repair_selection(previous,outcomes,proposal)
        changed=[]
        for old,new,result,action in zip(previous,requests,outcomes,actions):
            if result['status']=='ok':
                self.assertEqual(old,new)
            else:
                changed.append(new)
                self.assertEqual(new['specification']['messages'],old['specification']['messages'])
                self.assertEqual(new['observation_ids'],old['observation_ids'])
                self.assertEqual(new['specification']['parameters']['max_tokens'],8192)
                self.assertEqual(new['specification']['timeout_s'],180)
        self.assertEqual(len(changed),5)
        self.assertEqual(sum(action['action']=='reuse_successful_response' for action in actions),15)
        bad=deepcopy(proposal);bad['requests'][0]['specification']['parameters']['max_tokens']=16384
        with self.assertRaises(ValueError):
            validate_repair_selection(previous,outcomes,bad)

    def test_token_repair_stops_new_calls_without_dropping_old_successful_batches(self):
        import llm_continuation as continuation
        previous,outcomes,proposal=self.repair_fixture()
        requests,actions=continuation.validate_repair_selection(previous,outcomes,proposal)
        cached=[dict(row,new_api_attempt=False) for row in outcomes]
        pending=[row for row,action in zip(requests,actions) if action['action']!='reuse_successful_response']
        executed=[{'request_key':row['request_key'],'status':'parse_error' if i==0 else 'not_run',
                   'new_api_attempt':i==0} for i,row in enumerate(pending)]
        with tempfile.TemporaryDirectory() as folder,patch('llm_continuation.OUTPUT',Path(folder)), \
             patch('llm_pilot.request_results',return_value=executed) as runner, \
             patch('llm_continuation.stopped_outcomes',return_value=cached):
            result=continuation.execute_repair(Mock(),{'continuation':{'actions':actions}},requests)
            self.assertTrue((Path(folder)/'stop_record.json').exists())
        self.assertEqual(runner.call_args.args[-1],pending)
        self.assertEqual(sum(row['status']=='ok' for row in result),15)

    def continuation_fixture(self):
        folder=Path(__file__).resolve().parents[1]/'results/stage9/glm53'
        return tuple(json.loads((folder/name).read_text(encoding='utf-8')) for name in ('request_plan.json','request_outcomes.json'))

    def test_GLM_continuation_reuses_eight_successes_and_changes_only_authorized_requests(self):
        from llm_continuation import validate_selection
        previous,outcomes=self.continuation_fixture()
        selected,actions=validate_selection(previous,outcomes)
        self.assertEqual(selected[:8],previous[:8])
        self.assertEqual([row['action'] for row in actions],['reuse_successful_response']*8+['explicit_timeout_retry']+['first_attempt']*11)
        for old,new in zip(previous[8:],selected[8:]):
            expected=deepcopy(old['specification'])
            expected['timeout_s']=180
            expected['retry_of_request_key' if old['batch']==8 and old['mode']=='zero_shot' else 'continues_request_key']=old['request_key']
            self.assertEqual(new['specification'],expected)
            self.assertNotEqual(old['request_key'],new['request_key'])
        self.assertEqual(len({row['request_key'] for row in selected}),20)

    def test_GLM_continuation_rejects_unapproved_retry_or_changed_original_identity(self):
        from llm_continuation import validate_selection
        previous,outcomes=self.continuation_fixture()
        for index,patch_value in ((0,{'status':'api_error'}),(8,{'error_type':'PermissionDeniedError'}),(9,{'status':'ok'})):
            changed=deepcopy(outcomes);changed[index].update(patch_value)
            with self.subTest(index=index),self.assertRaises(ValueError):
                validate_selection(previous,changed)
        changed=deepcopy(previous);changed[0]['specification']['messages'][0]['content']='changed'
        with self.assertRaises(ValueError):
            validate_selection(changed,outcomes)

    def test_GLM_continuation_timeout_is_applied_to_transport_with_no_SDK_retries(self):
        model=dict(next(row for row in MODELS if row['name']=='glm53'),timeout_s=180)
        with patch('detector_llm.dotenv_values',return_value={'SYNTHETIC_API_KEY':'test'}),patch('openai.OpenAI') as factory:
            create_client(model)
            self.assertEqual(factory.call_args.kwargs['timeout'],180)
            self.assertEqual(factory.call_args.kwargs['max_retries'],0)
        spec=build_specification(model,'zero_shot',[prompt_observation('q001','SELECT 1')])
        self.assertEqual(spec['timeout_s'],180)

    def test_GLM53_replacement_plan_is_isolated_and_does_not_schedule_GPT_again(self):
        import llm_pilot
        from detector_llm import ACTIVE_MODEL_NAMES
        glm=next(model for model in MODELS if model['name']=='glm53')
        self.assertEqual(glm['model'],'hf:zai-org/GLM-5.3')
        self.assertEqual(set(ACTIVE_MODEL_NAMES),{'glm53','gptoss120','kimi31pro'})
        pilot=[{'input':prompt_observation(f'q{i:03d}','SELECT 1'),'target':i%2,'observation_id':f'obs{i}'} for i in range(30)]
        shots=[{'input':{'sql':'SELECT 1','context':{}},'target':i%2,'observation_id':f'shot{i}'} for i in range(6)]
        store=Mock();store.register_artifact.return_value='a'*64
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.local') as folder, \
             patch('llm_pilot.OUTPUT',Path(folder)),patch('llm_pilot.load_pilot',return_value={'pilot':pilot,'few_shot_candidates':shots}), \
             patch('llm_pilot.prepare_pilot') as register:
            plan,_,requests=llm_pilot.prepare(store,[glm])
        self.assertEqual(len(requests),20)
        self.assertEqual(plan['max_api_attempts'],20)
        self.assertEqual(register.call_args.args[-1],20)
        self.assertEqual({item['model_name'] for item in requests},{'glm53'})
        self.assertEqual({item['mode'] for item in requests},{'zero_shot','few_shot'})

    def test_ID_based_mapping_accepts_reordering_without_shifting_labels(self):
        text='{"predictions":[{"id":"b","label":1},{"id":"a","label":0}]}'
        self.assertEqual(parse_batch_response(text,['a','b']),[0,1])

    def test_AIMLAPI_uses_only_its_own_credential_and_disables_retries(self):
        import os
        model=next(model for model in MODELS if model['name']=='llama33_aimlapi')
        with patch.dict(os.environ,{},clear=True),patch('detector_llm.dotenv_values',return_value={
                'AIMLAPI_API_KEY':'test-aiml','SYNTHETIC_API_KEY':'test-synthetic','GROQ_API_KEY':'test-groq'}), \
             patch('openai.OpenAI') as factory:
            create_client(model)
            self.assertEqual(factory.call_args.kwargs,{'api_key':'test-aiml','base_url':'https://api.aimlapi.com/v1',
                                                      'max_retries':0,'timeout':90})
        with patch.dict(os.environ,{},clear=True),patch('detector_llm.dotenv_values',return_value={'SYNTHETIC_API_KEY':'wrong'}), \
             self.assertRaisesRegex(ValueError,'missing_credential:AIMLAPI_API_KEY'):
            create_client(model)
        with patch('detector_llm.dotenv_values') as load,self.assertRaisesRegex(ValueError,'Unknown provider'):
            create_client(dict(model,provider='unknown'))
        load.assert_not_called()

    def test_AIMLAPI_plan_has_only_six_Llama_batches_on_the_same_pilot(self):
        import llm_pilot
        model=next(model for model in MODELS if model['name']=='llama33_aimlapi')
        pilot=[{'input':prompt_observation(f'q{i:03d}','SELECT 1'),'target':i%2,'observation_id':f'obs{i}'} for i in range(30)]
        shots=[{'input':{'sql':'SELECT 1','context':{}},'target':i%2,'observation_id':f'shot{i}'} for i in range(6)]
        store=Mock();store.register_artifact.return_value='a'*64
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.local') as folder, \
             patch('llm_pilot.OUTPUT',Path(folder)),patch('llm_pilot.load_pilot',return_value={'pilot':pilot,'few_shot_candidates':shots}), \
             patch('llm_pilot.prepare_pilot'):
            plan,_,requests=llm_pilot.prepare(store,[model])
        self.assertEqual(plan['max_api_attempts'],6)
        self.assertEqual({item['model_name'] for item in requests},{'llama33_aimlapi'})
        self.assertEqual([len(item['observation_ids']) for item in requests],[10]*6)
        for mode in ('zero_shot','few_shot'):
            self.assertEqual([oid for request in requests if request['mode']==mode for oid in request['observation_ids']],
                             [row['observation_id'] for row in pilot])

    def test_Kimi_replaces_Llama_without_rescheduling_other_models(self):
        import llm_pilot
        from detector_llm import ACTIVE_MODEL_NAMES
        model=next(model for model in MODELS if model['name']=='kimi31pro')
        self.assertEqual(model['provider'],'synthetic')
        self.assertEqual(model['model'],'hf:moonshotai/Kimi-K3.1-Pro')
        self.assertFalse(any(name.startswith('llama') for name in ACTIVE_MODEL_NAMES))
        pilot=[{'input':prompt_observation(f'q{i:03d}','SELECT 1'),'target':i%2,'observation_id':f'obs{i}'} for i in range(30)]
        shots=[{'input':{'sql':'SELECT 1','context':{}},'target':i%2,'observation_id':f'shot{i}'} for i in range(6)]
        store=Mock();store.register_artifact.return_value='a'*64
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.local') as folder, \
             patch('llm_pilot.OUTPUT',Path(folder)),patch('llm_pilot.load_pilot',return_value={'pilot':pilot,'few_shot_candidates':shots}), \
             patch('llm_pilot.prepare_pilot'):
            plan,_,requests=llm_pilot.prepare(store,[model])
        self.assertEqual(plan['max_api_attempts'],6)
        self.assertEqual({item['model_name'] for item in requests},{'kimi31pro'})
        self.assertEqual([len(item['observation_ids']) for item in requests],[10]*6)

    def test_missing_extra_duplicate_IDs_and_invalid_labels_are_rejected(self):
        values=[[],[{'id':'a','label':0},{'id':'a','label':1}],
                [{'id':'a','label':0},{'id':'x','label':1}],
                [{'id':'a','label':True},{'id':'b','label':1}],
                [{'id':'a','label':'0'},{'id':'b','label':1}],
                [{'id':'a','label':2},{'id':'b','label':1}]]
        for rows in values:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                parse_batch_response(json.dumps({'predictions':rows}),['a','b'])

    def test_extra_prose_fences_duplicate_JSON_keys_and_legacy_labels_are_rejected(self):
        for text in ('["legit"]','```json\n{"predictions":[]}\n```','Here is JSON: {"predictions":[]}',
                     '{"predictions":[],"predictions":[]}'):
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_batch_response(text,['a'])

    def test_reasoning_is_not_used_when_final_content_is_empty(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(id='r',model='actual',created=1,usage=None,
            choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=None,
                reasoning_content='{"predictions":[{"id":"q001","label":0}]}'))])
        result=call_once(client,build_specification(MODELS[0],'zero_shot',[prompt_observation('q001','SELECT 1')]))
        self.assertEqual(result['status'],'parse_error')
        self.assertIsNone(result['labels'])
        self.assertNotIn('reasoning_content',result)

    def test_truncated_but_valid_JSON_is_not_silently_accepted(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(id='r',model='actual',created=1,usage=None,
            choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(content='{"predictions":[{"id":"q001","label":0}]}'))])
        result=call_once(client,build_specification(MODELS[0],'zero_shot',[prompt_observation('q001','SELECT 1')]))
        self.assertEqual(result['status'],'parse_error')
        self.assertIsNone(result['labels'])
        self.assertTrue(result['stop_model'])

    def test_stopped_continuation_never_reserves_unsent_requests(self):
        from llm_continuation import stopped_outcomes
        requests=[{'request_key':'cached','specification':{'test':True}},
                  {'request_key':'unsent','specification':{'test':False}}]
        result={'status':'parse_error','labels':None}
        store=MagicMock()
        store.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value=[{
            'request_key':'cached','specification':{'test':True},'state':'complete','result':result,'response_artifact':'artifact'}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'response.json'
            path.write_text(json.dumps({'request_key':'cached','result':result}),encoding='utf-8')
            store.get_artifact.return_value=path
            with patch('llm_continuation.reserve_request',side_effect=AssertionError('No reservations')), \
                 patch('detector_llm.create_client',side_effect=AssertionError('No clients')):
                outcomes=stopped_outcomes(store,{'ordered_request_keys':['cached','unsent']},requests)
        self.assertEqual([row['status'] for row in outcomes],['parse_error','not_run'])
        self.assertFalse(any(row['new_api_attempt'] for row in outcomes))

    def test_API_failure_is_one_attempt_no_fabricated_legit_or_secret_exception_text(self):
        client=Mock()
        client.chat.completions.create.side_effect=RuntimeError('SECRET_TEST_KEY')
        result=call_once(client,build_specification(MODELS[0],'zero_shot',[prompt_observation('q001','SELECT 1')]))
        self.assertEqual(client.chat.completions.create.call_count,1)
        self.assertEqual(result['status'],'api_error')
        self.assertIsNone(result['labels'])
        self.assertTrue(result['stop_model'])
        self.assertNotIn('SECRET_TEST_KEY',json.dumps(result))

    def test_request_identity_changes_with_context_model_order_settings_and_demonstrations(self):
        rows=[prompt_observation('q001','SELECT 1'),prompt_observation('q002','SELECT 2')]
        spec=build_specification(MODELS[0],'zero_shot',rows)
        variants=[dict(spec,model='different'),dict(spec,messages=list(reversed(spec['messages']))),
                  dict(spec,parameters={'max_completion_tokens':99}),
                  build_specification(MODELS[0],'zero_shot',list(reversed(rows))),
                  build_specification(MODELS[0],'few_shot',rows,[('SELECT 3',0)])]
        changed=deepcopy(rows);changed[0]['context']['dialect']={'availability':'known','value':'mysql'}
        variants.append(build_specification(MODELS[0],'zero_shot',changed))
        self.assertEqual(len({request_key(spec),*(request_key(item) for item in variants)}),len(variants)+1)

    def test_input_projection_removes_source_target_evidence_and_policy_identity(self):
        record={'sql':{'text':'SELECT 1'},'annotations':[{'target_label':1}],
                'source':{'origin':'hidden_origin'},'context':{'policy':{'availability':'known','available_before_decision':True,
                'evidence_ref':'hidden_ref','value':{'policy_id':'hidden_policy','version':'hidden_version',
                'coverage':'complete','rules':[],'scope_resources':['t']}}}}
        text=json.dumps(prompt_observation('q001',record))
        self.assertFalse(any(value in text for value in ('target_label','hidden_origin','hidden_ref','hidden_policy','hidden_version')))

    def test_clients_disable_SDK_retries(self):
        for model,module in ((MODELS[0],'groq.Groq'),(MODELS[1],'openai.OpenAI')):
            with patch('detector_llm.dotenv_values',return_value={'GROQ_API_KEY':'test','SYNTHETIC_API_KEY':'test'}), \
                 patch(module) as factory:
                create_client(model)
                self.assertEqual(factory.call_args.kwargs['max_retries'],0)


if __name__=='__main__':
    unittest.main()
