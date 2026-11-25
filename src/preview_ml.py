"""Small fixed ML study: shared benchmark, transfer, name sensitivity and diagnostic curves."""

from collections import Counter,defaultdict
from copy import deepcopy
import json
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import stage10_evaluate as evaluation
from evaluation_metrics import classification_report
from experiment_store import save_json,persist_run
from sql_features import prepare_input
from storage import ROOT,Storage,canonical,digest

OUTPUT=ROOT/'results/preview'
METHODS=('LogisticRegression','LinearSVC','RandomForest')


def label_rows(store,dataset,ids):
    with store.connect() as connection:
        return connection.execute('SELECT d.observation_id,a.target_label FROM nirs.dataset_members d JOIN nirs.annotations a USING(annotation_id) WHERE d.dataset_version_id=%s AND d.observation_id=ANY(%s)',(dataset,ids)).fetchall()


def balanced(store,dataset,ids,per_label,seed):
    groups=defaultdict(list)
    for row in label_rows(store,dataset,ids):
        groups[row['target_label']].append(row['observation_id'])
    if any(len(groups[label])<per_label for label in (0,1)):
        raise ValueError('Insufficient balanced preview rows')
    return sorted(oid for label in (0,1) for oid in sorted(groups[label],key=lambda value:digest((seed+value).encode()))[:per_label])


def load_training(store,dataset,ids):
    with store.connect() as connection:
        rows=connection.execute('''SELECT d.observation_id,o.sql_text,o.context_json,a.target_label FROM nirs.dataset_members d
            JOIN nirs.observations o USING(observation_id) JOIN nirs.annotations a USING(annotation_id)
            WHERE d.dataset_version_id=%s AND d.observation_id=ANY(%s)''',(dataset,ids)).fetchall()
    inputs={row['observation_id']:prepare_input({'sql':{'text':row['sql_text']},'context':row['context_json']}) for row in rows}
    labels={row['observation_id']:row['target_label'] for row in rows}
    if set(inputs)!=set(ids):
        raise ValueError('Preview training inputs missing')
    return inputs,labels


def selection(method,context=True,*,k=1000,strong=False):
    params={'C':.05 if strong else 1.,'class_weight':'balanced'} if method!='RandomForest' else {
        'n_estimators':100,'max_depth':4 if strong else 12,'min_samples_leaf':3,'class_weight':'balanced'}
    feature={'name':'masked_context' if context else 'masked_sql','input_mode':'sql_context' if context else 'sql_only',
        'relations':context,'parameters':{'text_mode':'masked','structure':True,'context':context,'select_k':k}}
    return {'candidate':{'method':method,'parameters':params,'feature':feature},'threshold':.5 if method=='RandomForest' else 0.}


class HistoricalAlarm:
    def __init__(self,pipeline):
        self.pipeline=pipeline
    def predict(self,inputs):
        queries=[item if isinstance(item,str) else item['sql']['text'] if isinstance(item['sql'],dict) else item['sql'] for item in inputs]
        return (self.pipeline.predict(queries)!=0).astype(np.int64)


def original_models(store):
    with store.connect() as connection:
        rows=connection.execute("SELECT r.method,a.artifact_id FROM nirs.experiment_runs r JOIN nirs.run_artifacts a USING(run_id) WHERE r.run_id LIKE 'stage1-control:%' AND a.role LIKE 'models/%'").fetchall()
    models=[]
    for row in rows:
        path=store.get_artifact(row['artifact_id']);raw=joblib.load(path)
        models.append({'name':'Historical '+row['method'],'input_mode':'sql_only','identity':row['artifact_id'],
            'estimator':HistoricalAlarm(raw),'raw_estimator':raw,'files':{'historical_model.joblib':path},'model_bytes':path.stat().st_size})
    return models


def main():
    store=Storage();registry=json.loads((ROOT/'results/stage6/split_registry.json').read_text(encoding='utf-8'))['protocols']
    pooled=json.loads((ROOT/'results/stage10/pooled_protocol.json').read_text(encoding='utf-8'))
    common=json.loads((OUTPUT/'common_inputs.json').read_text(encoding='utf-8'))['samples']
    sql=registry['transfer-web_app-to-wordpress-inner'];ctrl=registry['controlled-inner-0']
    source_inner_train=store.split_ids(sql['split_id'],0,'train')
    source_inner_valid=store.split_ids(sql['split_id'],0,'validation')
    source_train=balanced(store,pooled['dataset_version_id'],source_inner_train,1000,'preview-train:')
    source_valid=balanced(store,pooled['dataset_version_id'],source_inner_valid,200,'preview-valid:')
    ctrl_train=[oid for oid in store.split_ids(pooled['split_id'],0,'train') if oid.startswith('obs_s5_')]
    ctrl_valid=[oid for oid in store.split_ids(pooled['split_id'],0,'validation') if oid.startswith('obs_s5_')]
    ctrl_full=store.split_ids(ctrl['parent_scope']['split_id'],0,'train')
    fit_ids=sorted(source_train+ctrl_full); train=sorted(source_train+ctrl_train);valid=sorted(source_valid+ctrl_valid)
    if set(train)&set(valid) or len(fit_ids)!=2720:
        raise ValueError('Preview train/validation overlap or training size differs')
    inputs,labels=load_training(store,pooled['dataset_version_id'],sorted(set(fit_ids+valid)))
    dataset=store.create_subset(pooled['dataset_version_id'],sorted(train+valid),'Preview train-only diagnostic',{'purpose':'preview_diagnostic'})
    split=store.save_split(dataset,{'purpose':'preview_diagnostic'}, {0:{oid:'train' if oid in set(train) else 'validation' for oid in train+valid}})
    diagnostic={**pooled,'dataset_version_id':dataset,'split_id':split['split_id'],
                'parent_scope':{'split_id':split['split_id'],'fold':0,'part':'train'}}
    sig={**evaluation.signature(),'preview_ml.py':digest((ROOT/'src/preview_ml.py').read_bytes().replace(b'\r\n',b'\n'))}
    evaluation.OUTPUT=OUTPUT;evaluation.LOCAL=ROOT/'.local/preview'
    plan={'purpose':'preliminary_preview','methods':list(METHODS),'common_observations':30,'primary_fit_rows':2720,
        'diagnostic_fit_rows':len(train),'diagnostic_validation_rows':len(valid),'source_SQLi_train_per_label':1000,
        'learning_fractions':[.25,.5,1.0],'complexity_profiles':['baseline','strong_regularization_or_depth4'],
        'compactness_k':[100,1000],'new_classifier_fits_ceiling':27,'code_sha256_lf':sig,'hyperparameter_search':False,
        'full_sweep_deferred_to_stage':13,'fit_ids':fit_ids,'diagnostic_train_ids':train,'diagnostic_validation_ids':valid}
    save_json(OUTPUT/'evaluation_plan.json',plan)
    models=[]; results=[]; curve=[]
    with threadpool_limits(limits=1):
        for method in METHODS:
            for context in (False,True):
                model,_=evaluation.fit_frozen(store,pooled,selection(method,context),fit_ids,inputs,labels,sig,suffix='preview_primary')
                models.append(model)
        common_ids=[row['observation_id'] for row in common]
        results+=evaluation.evaluate_models(store,pooled,models+evaluation.rule_models(),'pooled-preview','common_30',sig,ids=common_ids)
        full_sample=json.loads((ROOT/'results/stage9/main_sample_300.json').read_text(encoding='utf-8'))['observations']
        results+=evaluation.evaluate_models(store,pooled,models,'pooled-preview','ML_holdout_300',sig,ids=[row['observation_id'] for row in full_sample])
        controlled_ids=[row['observation_id'] for row in full_sample if row['task']!='sqli'][:120]
        for transform in ('formatting','consistent_renaming'):
            results+=evaluation.evaluate_models(store,pooled,models,'pooled-preview','robustness',sig,ids=controlled_ids,transform=transform)
        save_json(OUTPUT/'ml_results.json',results)
        with store.connect() as connection:
            groups=connection.execute("SELECT m.observation_id,m.group_id FROM nirs.group_members m JOIN nirs.sample_groups g USING(group_id) WHERE g.kind='stage6_component' AND m.observation_id=ANY(%s)",(train,)).fetchall()
        by_group=defaultdict(list)
        for row in groups:by_group[row['group_id']].append(row['observation_id'])
        order=sorted(by_group,key=lambda group:digest(('preview-learning:'+group).encode()))
        for method in METHODS:
            variants=[]
            for fraction in (.25,.5,1.):
                kept=[]
                for group in order:
                    kept+=by_group[group]
                    if len(kept)>=fraction*len(train):break
                variants.append(('learning',str(fraction),selection(method),sorted(kept)))
            variants.append(('complexity','strong',selection(method,strong=True),train))
            variants.append(('compactness','100',selection(method,k=100),train))
            for kind,level,chosen,kept in variants:
                model,meta=evaluation.fit_frozen(store,diagnostic,chosen,kept,inputs,labels,sig,suffix='preview_diagnostic')
                predicted=model['estimator'].predict([inputs[oid] for oid in valid]).tolist()
                in_sample=model['estimator'].predict([inputs[oid] for oid in kept]).tolist()
                rows=[{'observation_id':oid,'fold':0,'truth':labels[oid],'prediction':int(value),'status':'ok',
                       'metadata':{'kind':kind,'level':level}} for oid,value in zip(valid,predicted)]
                metrics=classification_report([labels[oid] for oid in valid],predicted)
                config={'stage':11,'role':'inner_validation_used_for_selection','purpose':'preview_fixed_diagnostic',
                    'split_id':split['split_id'],'kind':kind,'level':level,'model_identity':model['identity'],'code_sha256_lf':sig}
                run,artifacts=persist_run(store,diagnostic,method,'binary',config,rows,{0:metrics},model['files'])
                curve.append({'run_id':run,'method':method,'kind':kind,'level':level,'fit_rows':len(kept),'columns':meta['columns'],
                    'model_bytes':model['model_bytes'],'train_metrics':classification_report([labels[oid] for oid in kept],in_sample),
                    'validation_metrics':metrics,'artifacts':artifacts})
                save_json(OUTPUT/'diagnostic_results.json',curve)
                print(f'preview diagnostic/{method}/{kind}/{level}: complete',flush=True)
        historical=original_models(store)
        transfer=[]
        for direction in ('transfer-web_app-to-wordpress','transfer-wordpress-to-web_app'):
            proto=registry[direction];entry={**proto,'parent_scope':{'split_id':proto['split_id'],'fold':0,'part':'train'}}
            train_ids=balanced(store,proto['dataset_version_id'],store.split_ids(proto['split_id'],0,'train'),1000,'preview-transfer-train:')
            test_ids=balanced(store,proto['dataset_version_id'],store.split_ids(proto['split_id'],0,'test'),150,'preview-transfer-test:')
            transfer_inputs,transfer_labels=load_training(store,proto['dataset_version_id'],train_ids)
            fixed=[]
            for method in METHODS:
                fitted,_=evaluation.fit_frozen(store,entry,selection(method,False),train_ids,transfer_inputs,transfer_labels,sig,suffix='preview_transfer')
                fixed.append(fitted)
            transfer+=evaluation.evaluate_models(store,entry,fixed+historical,direction,'transfer_300',sig,ids=test_ids)
            save_json(OUTPUT/'transfer_results.json',transfer)
        probes=[]
        for model in historical+[row for row in models if row['input_mode']=='sql_only']:
            for table in ('info.users','app.users','employees','orders','auth.credentials'):
                first=f'SELECT username, user_nickname FROM {table} WHERE record_id = 42'
                second=f'SELECT id, display_name FROM {table} WHERE record_id = 42'
                predictions=model['estimator'].predict([first,second]).tolist()
                classes=model.get('raw_estimator',model['estimator']).predict([first,second]).tolist()
                probes.append({'method':model['name'],'table':table,'queries':[first,second],'alarm_predictions':predictions,
                    'native_predictions':classes,'alarm_changed':predictions[0]!=predictions[1],'class_changed':classes[0]!=classes[1]})
        save_json(OUTPUT/'name_probes.json',{'purpose':'neutral_projection_sensitivity_not_reference_attack_labels','probes':probes})
    save_json(OUTPUT/'ml_completion.json',{'purpose':'preliminary_preview','new_fit_ceiling':27,'common_runs':len(results),
        'diagnostic_runs':len(curve),'transfer_runs':len(transfer),'source_signature':sig})
    print('Reduced ML study complete.')


if __name__=='__main__':main()
