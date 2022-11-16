import numpy as np
import pandas as pd
import pickle
import copy

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels


class AqClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, num_best=100, quality_index_type=0):
        self.num_best = num_best
        self.quality_index_type = quality_index_type

    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

    def fit(self, X, y, target_class=None):
        X, y = check_X_y(X, y, dtype=None)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        df = pd.DataFrame(X)
        df['class'] = y
        self.rules = AQAlgorithm(X[y == target_class], X[y != target_class])

        return self

    def predict(self, X):
        return predict_table(self.rules, X)


def AQAlgorithm(P, N):
    P1 = P
    R = set()  # ?
    while len(P1) > 1:
        # select random row from P1
        idx = np.random.randint(0, len(P1))
        p = P1[idx]

        # find a rule from p
        r = star(p, N, LEF, maxstar)
        P1 = P1.apply(lambda x: not r(x), axis=1)
        R.add(r)

    return R


def star(p, N, LEF, maxstar):
    r = set()
    for n in N:
        r1 = extension_against(p, n)
        r2 = r1.union(r)
        r2 = LEF(r2, maxstar)
        if mode == 'PD':
            if q(r2) - q(r) > minq:
                r = r2
        else:
            r = r2
    return r

def extension_against(p, n):
    r = set()

    for feature in p:
        pass

    return r

