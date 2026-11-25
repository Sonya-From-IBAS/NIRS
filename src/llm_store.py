"""Durable single-attempt request cache and explicit API budget accounting."""

import json

from psycopg.types.json import Jsonb

from detector_llm import request_key
from experiment_store import save_json
from storage import ROOT

LOCAL = ROOT/'.local/stage9'


def prepare_pilot(store,pilot_id,configuration,limit):
    with store.write() as connection:
        existing = connection.execute('SELECT * FROM nirs.llm_pilots WHERE pilot_id=%s',(pilot_id,)).fetchone()
        if existing:
            if existing['configuration'] != configuration or existing['max_api_attempts'] != limit:
                raise ValueError('Existing pilot configuration differs')
        else:
            connection.execute('INSERT INTO nirs.llm_pilots(pilot_id,configuration,max_api_attempts) VALUES (%s,%s,%s)',
                               (pilot_id,Jsonb(configuration),limit))


def reserve_request(store,pilot_id,specification):
    key = request_key(specification)
    folder = LOCAL/'requests'/key
    with store.connect() as connection:
        row = connection.execute('SELECT * FROM nirs.llm_requests WHERE request_key=%s',(key,)).fetchone()
    if row:
        if row['specification'] != specification:
            raise ValueError('Request cache identity differs')
        store.get_artifact(row['request_artifact'])
        if row['state'] == 'complete':
            path = store.get_artifact(row['response_artifact'])
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved != {'request_key':key,'result':row['result']}:
                raise ValueError('Response artifact differs from the database')
            return key,row['result'],False
        # A response written before a crash can be committed without an API call.
        response_path = folder/'response.json'
        if row['state'] == 'in_flight' and response_path.exists():
            recovered = json.loads(response_path.read_text(encoding='utf-8'))
            if recovered.get('request_key') != key:
                raise ValueError('Recovered response identity differs')
            finish_request(store,key,recovered['result'])
            return key,recovered['result'],False
        if row['state'] == 'in_flight':
            with store.write() as connection:
                connection.execute("UPDATE nirs.llm_requests SET state='interrupted',finished_at=now() WHERE request_key=%s",(key,))
        return key,{'status':'api_error','labels':None,'error_type':'interrupted_unknown_API_outcome',
                    'elapsed_s':None,'usage':None,'stop_model':False,'stop_provider':False},False
    save_json(folder/'request.json',specification)
    artifact = store.register_artifact(folder/'request.json')
    with store.write() as connection:
        # Budget and immutable identity are also enforced by migration 006.
        connection.execute("""INSERT INTO nirs.llm_requests(request_key,origin_pilot_id,specification,state,api_attempts,request_artifact)
            VALUES (%s,%s,%s,'in_flight',1,%s)""",(key,pilot_id,Jsonb(specification),artifact))
    return key,None,True


def finish_request(store,key,result):
    folder = LOCAL/'requests'/key
    save_json(folder/'response.json',{'request_key':key,'result':result})
    artifact = store.register_artifact(folder/'response.json')
    with store.write() as connection:
        changed = connection.execute("""UPDATE nirs.llm_requests SET state='complete',result=%s,response_artifact=%s,finished_at=now()
            WHERE request_key=%s AND state='in_flight'""",(Jsonb(result),artifact,key))
        if changed.rowcount != 1:
            raise ValueError('Only an in-flight request can be finalized once')
