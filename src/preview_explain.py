import json
import sys
import numpy as np
import joblib

sys.path.insert(0,'src')
from storage import ROOT,Storage
from experiment_store import save_json

store=Storage();folder=ROOT/'results/preview'
probe=next(row for row in json.loads((folder/'name_probes.json').read_text(encoding='utf-8'))['probes']
    if row['method']=='Historical TF-IDF + LinearSVC' and row['class_changed'])
with store.connect() as connection:
    aid=connection.execute("SELECT artifact_id FROM nirs.run_artifacts WHERE run_id='stage1-control:TF-IDF + LinearSVC' AND role='models/linearsvc.joblib'").fetchone()['artifact_id']
model=joblib.load(store.get_artifact(aid));matrix=model[:-1].transform(probe['queries']);classifier=model.steps[-1][1]
scores=classifier.decision_function(matrix);first,second=probe['native_predictions']
a=list(classifier.classes_).index(first);b=list(classifier.classes_).index(second)
weight=classifier.coef_[a]-classifier.coef_[b];bias=classifier.intercept_[a]-classifier.intercept_[b]
reconstructed=np.asarray(matrix@weight).ravel()+bias;actual=scores[:,a]-scores[:,b]
assert np.allclose(reconstructed,actual,rtol=0,atol=1e-10)
difference=matrix[0]-matrix[1];cols=difference.indices;values=difference.data*weight[cols]
names=model.steps[-2][1].get_feature_names_out()
top=sorted(zip(cols,values),key=lambda item:-abs(item[1]))[:15]
save_json(folder/'linear_explanation.json',{'model_artifact':aid,'queries':probe['queries'],'native_predictions':probe['native_predictions'],
    'compared_classes':[int(first),int(second)],'score_difference_actual':actual.tolist(),'score_difference_reconstructed':reconstructed.tolist(),
    'max_absolute_reconstruction_error':float(np.max(np.abs(actual-reconstructed))),
    'top_changed_feature_contributions':[{'feature':str(names[index]),'contribution':float(value)} for index,value in top],
    'interpretation':'exact linear score reconstruction; feature contribution is not proof of malicious intent'})
print('Exact historical SVC score reconstruction verified; saved fifteen changed n-gram contributions.')
