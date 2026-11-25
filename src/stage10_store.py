"""Pinned final-test loading and immutable, replayable post-row experiment results."""

from collections import defaultdict
import json

from psycopg.types.json import Jsonb

from evaluation_metrics import classification_report
from sql_features import prepare_input
from storage import canonical,digest


def load_test(store,dataset,split_id,fold,ids=None,*,legacy=False):
    with store.connect() as connection:
        rows=connection.execute('''SELECT d.observation_id,d.annotation_id,o.sql_text,o.context_json,o.grouping_json,
            a.target_label,a.status,a.data,c.source_label FROM nirs.split_members s
            JOIN nirs.dataset_members d ON d.dataset_version_id=s.dataset_version_id AND d.observation_id=s.observation_id
            JOIN nirs.observations o ON o.observation_id=d.observation_id JOIN nirs.annotations a USING(annotation_id)
            JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE s.dataset_version_id=%s AND s.split_id=%s AND s.fold=%s AND s.part='test'
            AND (%s::text[] IS NULL OR d.observation_id=ANY(%s)) ORDER BY d.observation_id''',
            (dataset,split_id,fold,ids,ids)).fetchall()
    indexed={row['observation_id']:row for row in rows}
    ordered=sorted(indexed) if ids is None else list(ids)
    if len(indexed)!=len(rows) or set(ordered)!=set(indexed) or len(ordered)!=len(indexed):
        raise ValueError('Final-test membership is missing, repeated or ambiguous')
    records=[]
    for oid in ordered:
        row=indexed[oid]
        truth=int(row['source_label']['label']) if legacy else row['target_label']
        if not legacy and (row['status']!='verified' or truth not in (0,1)):
            raise ValueError('Binary final tests require verified pinned references')
        payload={'sql':{'text':row['sql_text']},'context':row['context_json']}
        records.append({'observation_id':oid,'annotation_id':row['annotation_id'],'truth':truth,
            'payload':payload,'input':prepare_input(payload),'grouping':row['grouping_json'],
            'tags':row['data'].get('representation_tags',[]),'split_id':split_id,'fold':fold,
            'dataset_version_id':dataset})
    return records


def persist_test(store,dataset,method,configuration,rows,files,*,legacy=False):
    coordinates=[(row['split_id'],row['fold'],row['observation_id']) for row in rows]
    if not rows or len(coordinates)!=len(set(coordinates)):
        raise ValueError('Final-test coordinates must be nonempty and unique')
    space='legacy_five_class' if legacy else 'binary'
    scopes=defaultdict(list)
    for row in rows:
        scopes[(row['split_id'],row['fold'])].append(row['observation_id'])
    with store.connect() as connection:
        for (split_id,fold),ids in scopes.items():
            refs=connection.execute('''SELECT s.observation_id,a.status,a.target_label,c.source_label
                FROM nirs.split_members s JOIN nirs.dataset_members d
                ON d.dataset_version_id=s.dataset_version_id AND d.observation_id=s.observation_id
                JOIN nirs.annotations a USING(annotation_id) JOIN nirs.occurrences c
                ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
                WHERE s.dataset_version_id=%s AND s.split_id=%s AND s.fold=%s AND s.part='test'
                AND s.observation_id=ANY(%s)''',(dataset,split_id,fold,ids)).fetchall()
            by_id={row['observation_id']:row for row in refs}
            if set(by_id)!=set(ids):
                raise ValueError('A prediction is not in the frozen test part')
            for row in rows:
                if (row['split_id'],row['fold'])!=(split_id,fold):
                    continue
                ref=by_id[row['observation_id']]
                truth=int(ref['source_label']['label']) if legacy else ref['target_label']
                if truth!=row['truth'] or (not legacy and ref['status']!='verified'):
                    raise ValueError('Final-test reference changed')
    metrics=classification_report([row['truth'] for row in rows],[row['prediction'] for row in rows],
        label_space=space,statuses=[row['status'] for row in rows])
    run_id='run_'+digest(canonical({'dataset':dataset,'method':method,'configuration':configuration,
                                   'coordinates':coordinates}).encode())
    artifacts={role:store.register_artifact(path) for role,path in files.items()}
    with store.write() as connection:
        previous=connection.execute('SELECT configuration,status FROM nirs.experiment_runs WHERE run_id=%s',(run_id,)).fetchone()
        if previous:
            if previous['configuration']!=configuration or previous['status']!='complete':
                raise ValueError('An existing final-test run is inconsistent')
            actual=connection.execute('SELECT split_id,fold,observation_id,true_label,predicted_label,status,metadata FROM nirs.predictions WHERE run_id=%s',(run_id,)).fetchall()
            if {(row['split_id'],row['fold'],row['observation_id']):(row['true_label'],row['predicted_label'],row['status'],row['metadata']) for row in actual}!= {
                (row['split_id'],row['fold'],row['observation_id']):(row['truth'],row['prediction'],row['status'],row.get('metadata',{})) for row in rows}:
                raise ValueError('Replayed final-test predictions differ')
            saved=connection.execute("SELECT value FROM nirs.metrics WHERE run_id=%s AND protocol='test' AND fold=-1 AND metric_name='classification_report'",(run_id,)).fetchone()
            if not saved or saved['value']!=metrics:
                raise ValueError('Replayed final-test metrics differ')
            return {'run_id':run_id,'method':method,'metrics':metrics,'artifacts':artifacts,'configuration':configuration}
        connection.execute("INSERT INTO nirs.experiment_runs VALUES (%s,%s,%s,%s,'prepared',%s)",
            (run_id,dataset,method,space,Jsonb(configuration)))
        store._copy(connection,'predictions',['run_id','dataset_version_id','split_id','observation_id','protocol','fold','true_label','predicted_label','status','metadata'],
            [(run_id,dataset,row['split_id'],row['observation_id'],'test',row['fold'],row['truth'],row['prediction'],row['status'],Jsonb(row.get('metadata',{}))) for row in rows])
        reports={-1:metrics}
        for fold in sorted({row['fold'] for row in rows}):
            values=[row for row in rows if row['fold']==fold]
            reports[fold]=classification_report([row['truth'] for row in values],[row['prediction'] for row in values],
                label_space=space,statuses=[row['status'] for row in values])
        for fold,report in reports.items():
            connection.execute('INSERT INTO nirs.metrics VALUES (%s,\'test\',%s,\'classification_report\',%s)',(run_id,fold,Jsonb(report)))
        for role,aid in artifacts.items():
            connection.execute('INSERT INTO nirs.run_artifacts VALUES (%s,%s,%s,%s)',(run_id,aid,role,Jsonb({'stage':10})))
        connection.execute("UPDATE nirs.experiment_runs SET status='complete' WHERE run_id=%s",(run_id,))
    return {'run_id':run_id,'method':method,'metrics':metrics,'artifacts':artifacts,'configuration':configuration}
