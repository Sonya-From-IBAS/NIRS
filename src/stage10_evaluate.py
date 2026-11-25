"""Predeclared outer-test evaluation, representation controls and train-only diagnostics."""

import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from evaluation_metrics import classification_report
from experiment_store import LOCAL as MATRIX_LOCAL,cache_matrix,load_inputs,persist_run,save_json
from experiment_runner import rank_candidate
from ml_models import FrozenDecisionModel,binary_scores,choose_threshold,new_classifier,new_encoder
from rule_detectors import rule_predictions
from sql_features import prepare_input
from stage10_store import load_test,persist_test
from storage import ROOT,Storage,canonical,digest,file_hash
from study_features import renamed_input,reformatted_input

OUTPUT=ROOT/'results/stage10'
LOCAL=ROOT/'.local/stage10'
SOURCES=('stage10_evaluate.py','stage10_store.py','ml_models.py','sql_features.py','sql_views.py',
         'relational_features.py','normalize.py','rule_detectors.py','evaluation_metrics.py','experiment_store.py','study_features.py')
METHODS=('LogisticRegression','LinearSVC','RandomForest')
ANCHORS=('legacy-inner-0','controlled-inner-0','transfer-web_app-to-wordpress-inner','pooled-inner')


def signature():
    return {name:digest((ROOT/'src'/name).read_bytes().replace(b'\r\n',b'\n')) for name in SOURCES}


def make_plan():
    plan={'stage':10,'version':'main-experiments-v1','code_sha256_lf':signature(),
        'main_sample_sha256':file_hash(ROOT/'results/stage9/main_sample_300.json'),
        'outer_selected_models':'all eleven frozen nested scopes; no test-driven selection',
        'representation_anchors':list(ANCHORS),'representation_selection':'best inner OOF candidate per method and feature',
        'generation_control':'source-only selected SQL feature/parameters fixed; compare source-only vs source+controlled training with own train-only thresholds',
        'learning_fractions':[.25,.5,1.0],'learning_sampling':'deterministic prefix of whole global training groups',
        'complexity_profiles':{'linear_C':[.05,1.,10.],'forest_depth':[4,12,None],'forest_leaf':3},
        'compactness_k':[100,1000,None],'diagnostic_scope':'pooled inner fold zero validation, never final test for tuning',
        'robustness':['formatting','consistent_controlled_renaming'],
        'historical_control':'fixed historical_text C=1 balanced and RF100/leaf3 balanced, both grouped and row CV',
        'cascade':'not included in agreed core stage10'}
    path=OUTPUT/'evaluation_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Evaluation plan already frozen with a different configuration')
    save_json(path,plan)
    return plan


def records_in_chunks(store,entry,ids=None):
    parent=entry['parent_scope']
    ids=store.split_ids(parent['split_id'],parent['fold'],'test') if ids is None else ids
    for index in range(0,len(ids),512):
        yield load_test(store,entry['dataset_version_id'],parent['split_id'],parent['fold'],ids[index:index+512],
                        legacy=entry['cohort']=='legacy')


def task_of(record):
    group=record['grouping'].get('group_id') or ''
    if group.startswith('s5-family:'):
        return 'privilege' if group.startswith('s5-family:p') else 'exfiltration'
    return 'sqli'


def evaluate_models(store,entry,models,scope,phase,sig,*,transform=None,ids=None):
    """Each model has one frozen estimator and one declared input mode, with no task-based routing."""
    base_files={'code/'+name:ROOT/'src'/name for name in SOURCES}
    base_files['evaluation_plan.json']=OUTPUT/'evaluation_plan.json'
    active=[]; results=[]
    for model in models:
        config={'stage':10,'role':'final_test','scope':scope,'phase':phase,'input_mode':model['input_mode'],
            'model_identity':model['identity'],'code_sha256_lf':sig,'selection':model.get('selection'),
            'transform':transform,'sample_ids_sha256':digest(canonical(ids).encode()) if ids is not None else None,
            'known_policy_control_conformance_exposure':model.get('rule') in ('policy_only','regex_policy')}
        cache=LOCAL/'evaluation_runs'/(digest(canonical(config).encode())+'.json')
        if cache.exists():
            result=json.loads(cache.read_text(encoding='utf-8'))
            with store.connect() as connection:
                known=connection.execute('SELECT configuration,status FROM nirs.experiment_runs WHERE run_id=%s',(result['run_id'],)).fetchone()
            if not known or known['configuration']!=config or known['status']!='complete':
                raise ValueError('Evaluation cache differs from database')
            results.append(result)
        else:
            active.append((model,config,cache,[],[0.]))
    if not active:
        return results
    for records in records_in_chunks(store,entry,ids):
        for model,config,cache,rows,elapsed in active:
            payloads=[record['payload'] if model['input_mode']=='sql_context' else record['payload']['sql']['text'] for record in records]
            if transform:
                prepared=[prepare_input(item) for item in payloads]
                changed=[renamed_input(item) if transform=='consistent_renaming' else reformatted_input(item) for item in prepared]
                if transform=='consistent_renaming' and any(a.sql['masked']!=b.sql['masked'] or a.context_text!=b.context_text for a,b in zip(prepared,changed)):
                    raise ValueError('Renaming did not preserve the masked SQL/context relation')
                payloads=[item.projected for item in changed]
            started=time.perf_counter()
            if model.get('rule'):
                predicted,statuses,reasons=rule_predictions(model['rule'],payloads,legacy=entry['cohort']=='legacy')
            else:
                predicted=model['estimator'].predict(payloads).tolist();statuses=['ok']*len(records);reasons=['frozen_model']*len(records)
            elapsed[0]+=time.perf_counter()-started
            for record,value,status,reason in zip(records,predicted,statuses,reasons):
                rows.append({'observation_id':record['observation_id'],'split_id':record['split_id'],'fold':record['fold'],
                    'truth':record['truth'],'prediction':int(value) if value is not None else None,'status':status,
                    'metadata':{'reason':reason,'task':task_of(record),'representation_tags':record['tags']}})
    for model,config,cache,rows,elapsed in active:
        files={**base_files,**model.get('files',{})}
        result=persist_test(store,entry['dataset_version_id'],model['name'],config,rows,files,legacy=entry['cohort']=='legacy')
        result.update(scope=scope,phase=phase,input_mode=model['input_mode'],prediction_seconds=elapsed[0],
            observations=len(rows),model_bytes=model.get('model_bytes'),transform=transform)
        save_json(cache,result);results.append(result)
        print(f'{phase}/{scope}/{model["name"]}: {len(rows)} predictions stored',flush=True)
    return results


def selected_models(store,report):
    models=[]
    for record in report['selected_models']:
        path=store.get_artifact(record['artifacts']['model.joblib'])
        models.append({'name':record['method'],'input_mode':record['input_mode'],'estimator':joblib.load(path),
            'identity':record['artifacts']['model.joblib'],'selection':record['selection'],
            'model_bytes':path.stat().st_size,'files':{'model.joblib':path,'selection.json':store.get_artifact(record['artifacts']['selection.json'])}})
    return models


def rule_models(legacy=False):
    names=['regex_v1'] if legacy else ['regex_v1','regex_v2','policy_only','regex_policy']
    return [{'name':name,'rule':name,'input_mode':'sql_context' if name in ('policy_only','regex_policy') else 'sql_only',
             'identity':name} for name in names]


def fit_frozen(store,entry,selection,fit_ids,inputs,labels,sig,*,suffix='fixed-refit'):
    candidate=selection['candidate']; feature=candidate['feature']; parent=entry['parent_scope']
    config={'entry':entry['split_id'],'candidate':candidate,'threshold':selection['threshold'],
        'fit_ids_sha256':digest(canonical(fit_ids).encode()),'code_sha256_lf':sig,'suffix':suffix}
    folder=LOCAL/'fits'/digest(canonical(config).encode()); path=folder/'model.joblib'; meta=folder/'metadata.json'
    if meta.exists():
        saved=json.loads(meta.read_text(encoding='utf-8'))
        if saved['configuration']!=config or file_hash(path)!=saved['model_sha256']:
            raise ValueError('Frozen diagnostic model cache differs')
        frozen=joblib.load(path)
    else:
        rid,encoder,matrix,_,artifacts=cache_matrix(store,entry['dataset_version_id'],parent['split_id'],parent['fold'],
            feature,fit_ids,[],inputs,labels,sig)
        model=new_classifier(candidate['method'],candidate['parameters']); started=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(matrix,[labels[oid] for oid in fit_ids])
        if any(issubclass(item.category,ConvergenceWarning) for item in caught):
            raise ValueError('Diagnostic fit did not converge')
        frozen=FrozenDecisionModel(Pipeline([('features',encoder),('classifier',model)]),selection['threshold'])
        folder.mkdir(parents=True,exist_ok=True);joblib.dump(frozen,path,compress=3)
        saved={'configuration':config,'model_sha256':file_hash(path),'fit_rows':len(fit_ids),'columns':matrix.shape[1],
            'classifier_fit_seconds':time.perf_counter()-started,'matrix_artifacts':artifacts}
        save_json(meta,saved)
    return {'name':candidate['method'],'input_mode':feature['input_mode'],'estimator':frozen,'identity':store.register_artifact(path),
        'selection':selection,'model_bytes':path.stat().st_size,'files':{'model.joblib':path,'fit_metadata.json':meta}},saved


def primary(store,reports,sig):
    results=[]
    main_ids=[row['observation_id'] for row in json.loads((ROOT/'results/stage9/main_sample_300.json').read_text(encoding='utf-8'))['observations']]
    controlled_ids=[row['observation_id'] for row in json.loads((ROOT/'results/stage9/main_sample_300.json').read_text(encoding='utf-8'))['observations'] if row['task']!='sqli']
    for scope,report in reports.items():
        entry=report['entry']; models=selected_models(store,report)+rule_models(entry['cohort']=='legacy')
        results+=evaluate_models(store,entry,models,scope,'full_outer_test',sig)
        if scope=='pooled-inner':
            results+=evaluate_models(store,entry,models,scope,'common_300',sig,ids=main_ids)
            results+=evaluate_models(store,entry,models,scope,'context_ablation_200',sig,ids=controlled_ids)
        if entry['cohort']=='controlled':
            for transform in ('formatting','consistent_renaming'):
                results+=evaluate_models(store,entry,models,scope,'robustness',sig,transform=transform)
        save_json(OUTPUT/'outer_results.json',results)
    return results


def representation_controls(store,reports,sig):
    results=[]
    for scope in ANCHORS:
        report=reports[scope]; entry=report['entry']
        ids,inputs,labels=load_inputs(store,entry)
        grouped=defaultdict(list)
        for candidate in report['candidates']:
            grouped[(candidate['candidate']['method'],candidate['candidate']['feature']['name'])].append(candidate)
        for (method,feature),choices in sorted(grouped.items()):
            chosen=min(choices,key=rank_candidate)
            existing=next((row for row in report['selected_models'] if row['selection']['candidate']==chosen['candidate']
                and row['selection']['threshold']==chosen['threshold']),None)
            if existing:
                model=next(row for row in selected_models(store,{'selected_models':[existing]}) if row['name']==method)
            else:
                model,_=fit_frozen(store,entry,chosen,ids,inputs,labels,sig,suffix='representation_control')
            model['name']=method+'/'+feature
            results+=evaluate_models(store,entry,[model],scope,'representations',sig)
            save_json(OUTPUT/'representation_results.json',results)
        del inputs
    return results


def generation_controls(store,reports,sig):
    source=reports['transfer-web_app-to-wordpress-inner']; pooled=reports['pooled-inner']; entry=pooled['entry']
    ids,inputs,labels=load_inputs(store,entry)
    results=[]
    main_ids=[row['observation_id'] for row in json.loads((ROOT/'results/stage9/main_sample_300.json').read_text(encoding='utf-8'))['observations']]
    for record in source['selected_models']:
        without=selected_models(store,{'selected_models':[record]})[0]
        without['name']+=' / without controlled training'
        chosen=deepcopy(record['selection']); candidate=chosen['candidate']; outputs=[]; truth=[]
        # Fixed source-selected feature and hyperparameters; only training data and its OOF threshold change.
        for fold in range(3):
            train=store.split_ids(entry['split_id'],fold,'train'); valid=store.split_ids(entry['split_id'],fold,'validation')
            local_selection={'candidate':candidate,'threshold':0.}
            child=deepcopy(entry);child['parent_scope']={'split_id':entry['split_id'],'fold':fold,'part':'train'}
            fitted,_=fit_frozen(store,child,local_selection,train,inputs,labels,sig,suffix='generation_inner')
            pipeline=fitted['estimator'].pipeline
            matrix=pipeline.named_steps['features'].transform([inputs[oid] for oid in valid])
            outputs.extend(binary_scores(pipeline.named_steps['classifier'],matrix).tolist());truth.extend(labels[oid] for oid in valid)
        chosen['threshold']=choose_threshold(truth,outputs)
        chosen['threshold_reference']='pooled train-only OOF, fixed source-selected feature and parameters'
        with_generation,_=fit_frozen(store,entry,chosen,ids,inputs,labels,sig,suffix='generation_refit')
        with_generation['name']+=' / with controlled training'
        for phase,selected_ids in (('generation_full_test',None),('generation_common_300',main_ids)):
            results+=evaluate_models(store,entry,[without,with_generation],'pooled-inner',phase,sig,ids=selected_ids)
        save_json(OUTPUT/'generation_results.json',results)
    return results


def diagnostics(store,reports,sig):
    report=reports['pooled-inner']; entry=report['entry']
    parent_ids,inputs,labels=load_inputs(store,entry)
    train=store.split_ids(entry['split_id'],0,'train'); valid=store.split_ids(entry['split_id'],0,'validation')
    # A single frozen fold is reused only as an explicitly declared development diagnostic.
    valid_set=set(valid)
    split=store.save_split(entry['dataset_version_id'],{'stage':10,'protocol':'fixed_diagnostic_fold',
        'parent_scope':entry['parent_scope'],'original_inner_split':entry['split_id']},
        {0:{oid:'validation' if oid in valid_set else 'train' for oid in parent_ids}})
    child={**entry,'split_id':split['split_id']}
    with store.connect() as connection:
        membership=connection.execute("SELECT m.observation_id,m.group_id FROM nirs.group_members m JOIN nirs.sample_groups g USING(group_id) WHERE g.kind='stage6_component' AND m.observation_id=ANY(%s)",(train,)).fetchall()
    groups=defaultdict(list)
    for row in membership:
        groups[row['group_id']].append(row['observation_id'])
    ordered=sorted(groups,key=lambda group:digest(('stage10-learning:'+group).encode()))
    results=[]
    for record in report['selected_models']:
        if record['input_mode']!='sql_context':
            continue
        base=record['selection']; method=record['method']; configurations=[]
        for fraction in (.25,.5,1.):
            kept=[]
            for group in ordered:
                kept+=groups[group]
                if len(kept)>=fraction*len(train):
                    break
            configurations.append(('learning',str(fraction),deepcopy(base),sorted(kept)))
        for value in ((.05,1.,10.) if method!='RandomForest' else (4,12,None)):
            selected=deepcopy(base)
            if method!='RandomForest':
                selected['candidate']['parameters']={'C':value,'class_weight':'balanced'};selected['threshold']=0.
            else:
                selected['candidate']['parameters']={'n_estimators':100,'max_depth':value,'min_samples_leaf':3,'class_weight':'balanced'};selected['threshold']=.5
            configurations.append(('complexity',str(value),selected,train))
        for count in (100,1000,None):
            selected=deepcopy(base);selected['candidate']['feature']['parameters']['select_k']=count
            configurations.append(('compactness',str(count),selected,train))
        for kind,level,selection,fit_ids in configurations:
            local_entry=deepcopy(entry);local_entry['parent_scope']={'split_id':entry['split_id'],'fold':0,'part':'train'}
            model,metadata=fit_frozen(store,local_entry,selection,fit_ids,inputs,labels,sig,suffix='diagnostic')
            frozen=model['estimator']; y=frozen.predict([inputs[oid] for oid in valid]).tolist()
            train_prediction=frozen.predict([inputs[oid] for oid in fit_ids]).tolist()
            rows=[{'observation_id':oid,'fold':0,'truth':labels[oid],'prediction':int(value),'status':'ok','metadata':{'diagnostic':kind,'level':level}}
                  for oid,value in zip(valid,y)]
            metrics=classification_report([labels[oid] for oid in valid],y)
            config={'stage':10,'role':'inner_validation_used_for_selection','purpose':'fixed_configuration_diagnostic',
                'split_id':split['split_id'],'kind':kind,'level':level,'model_identity':model['identity'],
                'feature_and_parameters':selection['candidate'],'fit_ids_sha256':digest(canonical(fit_ids).encode()),
                'no_outer_test_used':True,'hyperparameters_preselected_on_parent_train':True,'code_sha256_lf':sig}
            run,artifacts=persist_run(store,child,method, 'binary',config,rows,{0:metrics},model['files'])
            results.append({'run_id':run,'method':method,'kind':kind,'level':level,'fit_rows':len(fit_ids),
                'columns':metadata['columns'],'model_bytes':model['model_bytes'],'validation_metrics':metrics,
                'train_metrics':classification_report([labels[oid] for oid in fit_ids],train_prediction),'artifacts':artifacts})
            save_json(OUTPUT/'diagnostic_results.json',results)
            print(f'diagnostic/{method}/{kind}/{level}: saved',flush=True)
    return results


def historical_control(store,reports,sig):
    registry=json.loads((ROOT/'results/stage6/split_registry.json').read_text(encoding='utf-8'))['protocols']
    results=[]
    for protocol in ('legacy-outer','legacy-row-control'):
        outer=registry[protocol]
        for fold in range(5):
            entry={'dataset_version_id':outer['dataset_version_id'],'split_id':outer['split_id'],'cohort':'legacy',
                'parent_scope':{'split_id':outer['split_id'],'fold':fold,'part':'train'}}
            ids,inputs,labels=load_inputs(store,entry)
            for method in METHODS:
                parameters={'C':1.,'class_weight':'balanced'} if method!='RandomForest' else {
                    'n_estimators':100,'max_depth':None,'min_samples_leaf':3,'class_weight':'balanced'}
                selection={'candidate':{'method':method,'parameters':parameters,'feature':{'name':'historical_text',
                    'input_mode':'sql_only','relations':False,'parameters':{'text_mode':'legacy','structure':False}}},'threshold':None}
                model,_=fit_frozen(store,entry,selection,ids,inputs,labels,sig,suffix='historical_control')
                results+=evaluate_models(store,entry,[model],protocol+':'+str(fold),'fixed_historical_control',sig)
                save_json(OUTPUT/'historical_results.json',results)
            del inputs
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--phase',choices=('primary','representations','generation','diagnostics','historical','all'),default='all')
    args=parser.parse_args();plan=make_plan()
    if args.plan_only:
        print('Frozen stage10 evaluation plan; no final-test model predictions made.')
        return
    if not (OUTPUT/'training_summary.json').exists():
        raise ValueError('Finish all frozen inner searches before outer evaluation')
    store=Storage()
    reports={path.stem:json.loads(path.read_text(encoding='utf-8')) for path in sorted((OUTPUT/'selections').glob('*.json'))}
    if len(reports)!=11:
        raise ValueError('Expected all ten original scopes and the pooled scope')
    sig=plan['code_sha256_lf']; phases={'primary':primary,'representations':representation_controls,
        'generation':generation_controls,'diagnostics':diagnostics,'historical':historical_control}
    with threadpool_limits(limits=1):
        for name,function in phases.items():
            if args.phase in ('all',name):
                results=function(store,reports,sig)
                save_json(OUTPUT/(name+'_completion.json'),{'stage':10,'phase':name,'runs':len(results),'code_sha256_lf':sig})
    if signature()!=sig:
        raise ValueError('Evaluation source changed during execution')


if __name__=='__main__':
    main()
