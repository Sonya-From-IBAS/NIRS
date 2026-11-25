"""Complete frozen nested searches and prepare a single pooled binary classifier per method."""

from collections import Counter,defaultdict
import argparse
import json
import os

from threadpoolctl import threadpool_limits

import experiment_runner as runner
from experiment_store import save_json
from ml_models import classifier_candidates,feature_candidates
from storage import ROOT,Storage,canonical,digest,file_hash

OUTPUT=ROOT/'results/stage10'
SOURCES=(*runner.SOURCES,'stage10_training.py')


def pooled_protocol(store,registry):
    sql=registry['transfer-web_app-to-wordpress-inner']
    controlled=registry['controlled-inner-0']
    with store.connect() as connection:
        versions=connection.execute('SELECT DISTINCT source_version_id FROM nirs.dataset_members WHERE dataset_version_id=ANY(%s)',
            ([sql['dataset_version_id'],controlled['dataset_version_id']],)).fetchall()
    dataset=store.create_verified_source_dataset([row['source_version_id'] for row in versions],
        'Stage10 pooled verified SQLi and controlled scenarios',{'stage':10,'role':'pooled_binary','sources':[
            sql['dataset_version_id'],controlled['dataset_version_id']]})
    train_ids=set(); all_ids=set()
    for entry in (sql,controlled):
        parent=entry['parent_scope']
        train_ids.update(store.split_ids(parent['split_id'],parent['fold'],'train'))
        all_ids.update(store.split_ids(parent['split_id'],parent['fold'],'train'))
        all_ids.update(store.split_ids(parent['split_id'],parent['fold'],'test'))
    with store.connect() as connection:
        global_rows=connection.execute("SELECT m.observation_id,m.group_id FROM nirs.group_members m JOIN nirs.sample_groups g USING(group_id) JOIN nirs.dataset_members d USING(observation_id) WHERE g.kind='stage6_component' AND d.dataset_version_id=%s",(dataset,)).fetchall()
    global_groups=defaultdict(set)
    for row in global_rows:
        global_groups[row['group_id']].add(row['observation_id'])
    if {oid for ids in global_groups.values() for oid in ids}!=all_ids:
        raise ValueError('Pooled global groups do not cover the dataset')
    if any(ids&train_ids and ids-train_ids for ids in global_groups.values()):
        raise ValueError('Pooled train and test share a global group')
    outer=store.save_split(dataset,{'stage':10,'protocol':'pooled_source_transfer_and_controlled_outer0',
        'components':[sql['parent_scope'],controlled['parent_scope']]},
        {0:{oid:'train' if oid in train_ids else 'test' for oid in all_ids}})
    ctrl_ids=store.split_ids(controlled['parent_scope']['split_id'],0,'train')
    with store.connect() as connection:
        groups=connection.execute("SELECT m.observation_id,m.group_id FROM nirs.group_members m JOIN nirs.sample_groups g USING(group_id) WHERE g.kind='stage6_component' AND m.observation_id=ANY(%s)",(ctrl_ids,)).fetchall()
    grouped=defaultdict(list)
    for row in groups:
        grouped[row['group_id']].append(row['observation_id'])
    if {oid for ids in grouped.values() for oid in ids}!=set(ctrl_ids):
        raise ValueError('Pooled controlled training groups are missing')
    sizes=[0,0,0]; ctrl_fold={}
    for group,ids in sorted(grouped.items(),key=lambda item:(-len(item[1]),item[0])):
        fold=min(range(3),key=lambda index:(sizes[index],index))
        sizes[fold]+=len(ids)
        ctrl_fold.update({oid:fold for oid in ids})
    assignments={}
    for fold in range(3):
        valid=set(store.split_ids(sql['split_id'],fold,'validation'))|{oid for oid,f in ctrl_fold.items() if f==fold}
        assignments[fold]={oid:'validation' if oid in valid else 'train' for oid in train_ids}
        if any(len({assignments[fold][oid] for oid in ids if oid in train_ids})>1 for ids in global_groups.values()):
            raise ValueError('Pooled inner fold splits a global group')
    inner=store.save_split(dataset,{'stage':10,'protocol':'pooled_inner','parent_scope':{'split_id':outer['split_id'],'fold':0,'part':'train'},
        'SQLi_inner_split':sql['split_id'],'controlled_group_assignment':ctrl_fold},assignments)
    return {'dataset_version_id':dataset,'split_id':inner['split_id'],'cohort':'pooled','role':'inner_tuning',
        'parent_scope':{'split_id':outer['split_id'],'fold':0,'part':'train'},'folds':{str(f):{} for f in range(3)},
        'rows':{'train':len(train_ids),'test':len(all_ids-train_ids)},'controlled_validation_rows':sizes}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true',help='Prepare pooled splits and frozen search plan without fitting any classifier')
    args=parser.parse_args()
    store=Storage(); store.migrate()
    registry=json.loads((ROOT/'results/stage6/split_registry.json').read_text(encoding='utf-8'))['protocols']
    entry=pooled_protocol(store,registry)
    save_json(OUTPUT/'pooled_protocol.json',entry)
    scopes=[name for name,value in registry.items() if value['role']=='inner_tuning']
    signatures=runner.code_signature()
    driver_sha256=file_hash(ROOT/'src/stage10_training.py')
    code={'code/'+name:ROOT/'src'/name for name in SOURCES}
    code.update({'schema/'+path.name:path for path in sorted((ROOT/'migrations').glob('*.sql'))})
    for path in code.values():
        store.register_artifact(path)
    plan={'stage':10,'phase':'all_inner_tuning_before_outer_evaluation','outer_test_used':False,
        'scopes':scopes+['pooled-inner'],'pooled_entry':entry,'code_sha256_lf':signatures,'driver_sha256':driver_sha256,
        'candidate_grids':{cohort:feature_candidates(cohort) for cohort in ('legacy','controlled','sqlstateguard')},
        'pooled_features':feature_candidates('controlled'),'classifiers':{method:classifier_candidates(method) for method in runner.METHODS},
        'selection':'pooled inner OOF F1, then lower FPR, higher recall and frozen complexity order',
        'new_search_threads':1,'reused_stage8_scopes':list(runner.SCOPES),'test_driven_changes':False}
    save_json(OUTPUT/'training_plan.json',plan)
    if args.prepare_only:
        print('Prepared pooled datasets and splits; no classifier fits requested.')
        return
    runner.OUTPUT=OUTPUT/'selections'
    reports=[]
    with threadpool_limits(limits=1):
        for name in scopes:
            target=runner.OUTPUT/(name+'.json')
            original=ROOT/'results/stage8'/(name+'.json')
            if name in runner.SCOPES:
                report=json.loads(original.read_text(encoding='utf-8'))
                for model in report['selected_models']:
                    store.get_artifact(model['artifacts']['model.joblib'])
                save_json(target,report)
                print(name+': reused frozen stage8 selection',flush=True)
            elif target.exists():
                report=json.loads(target.read_text(encoding='utf-8'))
                for model in report['selected_models']:
                    store.get_artifact(model['artifacts']['model.joblib'])
                print(name+': reused completed stage10 selection',flush=True)
            else:
                report=runner.run_scope(store,name,signatures,code)
            reports.append(report)
        name='pooled-inner'; target=runner.OUTPUT/(name+'.json')
        if target.exists():
            report=json.loads(target.read_text(encoding='utf-8'))
            if report['code_sha256_lf']!=signatures:
                raise ValueError('Pooled selection source differs')
        else:
            report=runner.run_scope(store,name,signatures,code,entry=entry,features=feature_candidates('controlled'))
        reports.append(report)
    if signatures!=runner.code_signature() or file_hash(ROOT/'src/stage10_training.py')!=driver_sha256:
        raise ValueError('Training source changed while running')
    save_json(OUTPUT/'training_summary.json',{'stage':10,'scopes':[report['scope'] for report in reports],
        'classifier_fits':sum(report['classifier_fits'] for report in reports),
        'selected_models':sum(len(report['selected_models']) for report in reports),
        'rule_runs':sum(len(report['rules']) for report in reports),'outer_test_used':False,
        'selection_files':{path.name:file_hash(path) for path in sorted(runner.OUTPUT.glob('*.json'))}})


if __name__=='__main__':
    main()
