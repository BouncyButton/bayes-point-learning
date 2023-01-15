from collections import Counter

import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# i'd like to suppress the following warning
# C:\Users\<user>\.conda\envs\myenv\lib\site-packages\wittgenstein\base.py:127: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
#   covered = covered.append(rule.covers(df))
# that is present in the wittgenstein lib
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from wittgenstein import RIPPER

from bpllib._bp import callable_rules_bo, callable_rules_bp, alpha_representation
from bpllib._bpl import Rule, DiscreteConstraint


class RIPPERClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, T=1):
        self.T = T

    def fit(self, X, y, target_class=1, find_best_k=False, starting_seed=0):
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
        df['class'] = y.astype(int)

        self.rulesets_ = []

        if self.T == 1:
            self.inner_clf_ = RIPPER()
            self.inner_clf_.fit(df, class_feat='class', pos_class=target_class)

        else:
            self.classifiers_ = []
            for t in range(self.T):
                # permute the df
                perm_df = df.sample(frac=1, random_state=t)

                # create a new classifier
                clf = RIPPER(random_state=starting_seed * self.T + t)  # dovrei?
                clf.fit(perm_df, class_feat='class', pos_class=target_class)
                self.classifiers_.append(clf)
                ruleset = clf.ruleset_
                our_ruleset = []
                # convert the ruleset found in our format
                for rule in ruleset:
                    constraints = dict()
                    for cond in rule.conds:
                        constraints[cond.feature] = DiscreteConstraint(index=cond.feature, value=cond.val)
                    our_rule = Rule(constraints)
                    our_ruleset.append(our_rule)
                self.rulesets_.append(our_ruleset)
            # count the rules
            self.counter_ = alpha_representation(self.rulesets_)

            # suggest k rules using best k
            best_acc = 0
            MAX_PATIENCE = 10
            patience = MAX_PATIENCE
            self.suggested_k_ = 0
            if find_best_k:
                for k in range(1, len(self.counter_)):

                    if patience == 0:
                        break
                    most_freq_rules = self.counter_.most_common(k)
                    y_train_pred = [self.predict_one(x, strategy='best-k', n_rules=k, most_freq_rules=most_freq_rules)
                                    for x in X]
                    acc = np.mean(y_train_pred == y)
                    if acc > best_acc:
                        self.suggested_k_ = k
                    else:
                        patience -= 1
        return self

    def predict(self, X, strategy='bo', n_rules=None):
        if self.T == 1:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self.inner_clf_.predict(X)

        if strategy == 'bo' or strategy == 'bp' or strategy == 'best-k':
            if isinstance(X, pd.DataFrame):
                X = np.array(X.values)
            if strategy == 'best-k':
                most_freq_rules = self.counter_.most_common(n_rules)
                return np.array([self.predict_one(x, most_freq_rules=most_freq_rules,
                                                  strategy=strategy, n_rules=n_rules) for x in X])
            else:
                return np.array([self.predict_one(x, strategy=strategy, n_rules=n_rules) for x in X])
        else:
            raise NotImplementedError('strategy not available')

    def predict_one(self, x, strategy='bo', n_rules=None, most_freq_rules=None):
        if strategy == 'bo':
            rules_bo = callable_rules_bo(self.rulesets_)
            value = sum([ht(x) for ht in rules_bo])
            return self.target_class_ if np.sign(value) > 0 else self.other_class_
        elif strategy == 'bp':
            h_bp = callable_rules_bp(self.rulesets_)
            value = h_bp(x)
            return self.target_class_ if value > self.T / 2 else self.other_class_
        elif strategy == 'best-k':
            if n_rules is None and not hasattr(self, 'suggested_k_'):
                raise ValueError('n_rules must be specified')

            if n_rules is None:
                n_rules = self.suggested_k_
            vote = 0

            alpha_rules = sum(self.counter_.values())
            if alpha_rules == 0:
                return self.other_class_
            alpha_k_rules = sum([alpha for r, alpha in self.counter_.most_common(n_rules)])
            gamma_k = alpha_k_rules / alpha_rules

            if most_freq_rules is None:
                most_freq_rules = self.counter_.most_common(n_rules)

            for rule, alpha in most_freq_rules:
                vote += rule.covers(x) * alpha
            return self.target_class_ if vote > gamma_k * self.T / 2 else self.other_class_
        else:
            raise NotImplementedError('strategy not available')
