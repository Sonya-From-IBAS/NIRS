"""Strict parsing, final-content isolation and model-independent request identity."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detector_llm import MODELS,build_specification,call_once,create_client,parse_batch_response,prompt_observation,request_key


class LLMTests(unittest.TestCase):
    def test_ID_based_mapping_accepts_reordering_without_shifting_labels(self):
        text='{"predictions":[{"id":"b","label":1},{"id":"a","label":0}]}'
        self.assertEqual(parse_batch_response(text,['a','b']),[0,1])

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
