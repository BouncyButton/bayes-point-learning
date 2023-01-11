import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from bpllib._bpl import OrdinalConstraint, Rule


class BayesPointRandomForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, T=1):
        self.T = T

    def fit(self, X, y, target_class=1):
        self.target_class_ = target_class
        X, y = check_X_y(X, y, dtype=[str, int])
        self.classes_ = unique_labels(y)
        if len(self.classes_) == 2:
            self.other_class_ = (set(self.classes_) - {target_class}).pop()
        else:
            raise NotImplementedError('multiclass not available yet')
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        df = pd.DataFrame(X)
        df['class'] = y

        self.rf_ = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0).fit(X, y)

        self.set_of_rulesets_ = []

        return self


def get_paths(clf, t, node_id, curr_path, curr_rule, target_class=1):
    # print('cp:', curr_path)
    # if node_id is a leaf
    if t.tree_.children_left[node_id] == t.tree_.children_right[node_id] == -1:
        # print(node_id, 'is leaf')
        # print('path found:', curr_path)
        v = t.tree_.value[node_id]
        if clf.classes_[np.argmax(v)] == target_class:
        return [curr_path], [Rule(curr_rule)]

    # print(node_id, 'is not leaf')
    # otherwise, use recursion

    condition_left = OrdinalConstraint(index=t.tree_.feature[node_id], value_max=t.tree_.threshold[node_id])
    condition_right = OrdinalConstraint(index=t.tree_.feature[node_id], value_min=t.tree_.threshold[node_id])
    l_path, l_rules = get_paths(clf, t, t.tree_.children_left[node_id], curr_path + [node_id], curr_rule + [condition_left])
    r_path, r_rules = get_paths(clf, t, t.tree_.children_right[node_id], curr_path + [node_id],
                                curr_rule + [condition_right])
    return l_path + r_path, l_rules + r_rules

