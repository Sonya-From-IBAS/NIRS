"""Independent estimators, frozen candidate grids and validation-only thresholds."""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from relational_features import RelationalFeatureEncoder
from sql_features import SQLFeatureEncoder

VERSION = 'stage8-models-v1'
SEED = 20261125


def feature_candidates(cohort):
    sql = {
        'raw_10k': {'text_mode': 'raw', 'structure': False, 'max_features': 10000},
        'raw_50k': {'text_mode': 'raw', 'structure': False},
        'masked': {'text_mode': 'masked', 'structure': False},
        'masked_structure_1k': {'text_mode': 'masked', 'structure': True, 'select_k': 1000},
    }
    if cohort == 'legacy':
        sql['historical_text'] = {'text_mode': 'legacy', 'structure': False}
    if cohort == 'controlled':
        sql.pop('raw_50k')
    result = [{'name': name, 'input_mode': 'sql_only', 'relations': False, 'parameters': values}
              for name, values in sql.items()]
    if cohort == 'controlled':
        result.extend([
            {'name': 'text_context', 'input_mode': 'sql_context', 'relations': False,
             'parameters': {'text_mode': 'masked', 'structure': False, 'context': True}},
            {'name': 'full_relations', 'input_mode': 'sql_context', 'relations': True,
             'parameters': {'text_mode': 'masked', 'structure': True, 'context': True}},
            {'name': 'structure_context_relations', 'input_mode': 'sql_context', 'relations': True,
             'parameters': {'text_mode': None, 'structure': True, 'context': True}},
        ])
    return result


def classifier_candidates(method):
    if method in ('LogisticRegression', 'LinearSVC'):
        return [{'C': .05, 'class_weight': 'balanced'}, {'C': 1., 'class_weight': 'balanced'},
                {'C': 1., 'class_weight': None}]
    if method == 'RandomForest':
        return [{'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 3, 'class_weight': 'balanced'},
                {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1, 'class_weight': 'balanced'},
                {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 3, 'class_weight': None}]
    raise ValueError('Unknown classifier')


def new_encoder(specification):
    factory = RelationalFeatureEncoder if specification['relations'] else SQLFeatureEncoder
    return factory(**deepcopy(specification['parameters']))


def new_classifier(method, parameters):
    if method == 'LogisticRegression':
        return LogisticRegression(solver='liblinear', max_iter=4000, random_state=SEED, **parameters)
    if method == 'LinearSVC':
        return LinearSVC(dual='auto', max_iter=10000, random_state=SEED, **parameters)
    if method == 'RandomForest':
        return RandomForestClassifier(n_jobs=1, random_state=SEED, **parameters)
    raise ValueError('Unknown classifier')


def binary_scores(model, matrix):
    if list(model.classes_) != [0, 1]:
        raise ValueError('Binary scores require both classes in training')
    values = model.decision_function(matrix) if hasattr(model, 'decision_function') else model.predict_proba(matrix)[:, 1]
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError('Invalid classifier scores')
    return result


def choose_threshold(truth, scores):
    """Exact F1 sweep, then lower FPR, higher recall, higher threshold.

    Equal scores move together. Only inner out-of-fold references are supplied.
    Complexity is O(n log n); no quadratic rescan over possible thresholds.
    """
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or len(truth) != len(scores) or not len(truth) or not set(truth) <= {0, 1}:
        raise ValueError('Invalid binary threshold input')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite')
    order = np.argsort(-scores, kind='stable')
    ranked, sorted_scores = truth[order], scores[order]
    stops = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores)-1]
    positives, negatives = int(truth.sum()), int((truth == 0).sum())
    tp = np.r_[0, np.cumsum(ranked)[stops]]
    fp = np.r_[0, stops + 1 - tp[1:]]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[stops]]
    denominator = positives + tp + fp
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp), dtype=float), where=denominator != 0)
    # FP and TP counts implement the declared FPR/recall order on this fixed set.
    winner = min(range(len(tp)), key=lambda i: (-f1[i], fp[i], -tp[i], -thresholds[i]))
    return float(thresholds[winner])


class FrozenDecisionModel(BaseEstimator, ClassifierMixin):
    """A fitted pipeline plus a validation-chosen binary decision threshold."""
    def __init__(self, pipeline, threshold=None):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, inputs):
        if self.threshold is None:
            return self.pipeline.predict(inputs)
        matrix = self.pipeline.named_steps['features'].transform(inputs)
        return (binary_scores(self.pipeline.named_steps['classifier'], matrix) >= self.threshold).astype(np.int64)
