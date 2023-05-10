import copy
import multiprocessing
import random
import time
from itertools import repeat
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
from tqdm import tqdm

from bpllib._bp import callable_rules_bo, callable_rules_bp
from .utils import resample
from joblib import Memory


class BayesPointClassifier(ClassifierMixin, BaseEstimator):
    '''
    Abstract class for all bayesian point classifiers.
    To implement new Bayes Point Classifiers, you should subclass this class.
    '''
    description = '(Abstract) Bayes Point Classifier'

    def __init__(self, T=3, verbose=0, threshold_acc=0.99, target_class=None, pool_size='auto', find_best_k=True,
                 random_state=None, encoding='av', to_string=True, cachedir='cachedir', prune_strategy=None,
                 max_rules=None,
                 **kwargs):
        self.max_rules = max_rules
        self.prune_strategy = prune_strategy
        self.encoding = encoding
        self.random_state = random_state
        self.find_best_k = find_best_k
        self.pool_size = pool_size
        self.target_class = target_class
        self.T = T
        self.suggested_k_ = None
        self.counter_ = None
        self.bins_ = None
        self.other_class_ = None
        self.target_class_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.ovr_ = None
        self.rule_sets_ = None
        self.use_bootstrap = False
        self.threshold_acc = threshold_acc
        self.verbose = verbose
        self.kwargs = kwargs
        self.to_string = to_string
        self.cachedir = cachedir

        self.memory = Memory(location=cachedir, verbose=0)

    def base_method(self, X, y, target_class):
        '''
        This method should be implemented by the subclass.

        :param X: the training data
        :param y: the training labels
        :param target_class: the class to be learned

        :return: a rule set.
        '''
        raise NotImplementedError()

    def alpha_representation(self):
        # Check is fit had been called
        check_is_fitted(self, ['rule_sets_'], all_or_any=any)

        from collections import Counter
        cnt = Counter()
        for ruleset in self.rule_sets_:
            curr_counts = Counter(ruleset)
            cnt.update(curr_counts)
        return cnt

    def best_k_rules(self, k=20):
        cnt = self.alpha_representation()
        return cnt.most_common(k)

    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

    def checks_for_base_method(self, X, y):
        '''
        This method checks that the input data is correct.
        You can subclass this method in order to add more checks, implementing binning, etc.
        :param X: the training data
        :param y: the training labels
        :return: the checked data
        '''
        # Check that X and y have correct shape
        # we accept strings
        if np.any(np.array(X).dtype.kind == 'f'):
            warnings.warn("bayes point classifier does not fully support float values yet.")
            X, y = check_X_y(X, y, dtype=None)

        # check for strings
        elif np.any(np.array(X).dtype.kind in ('S', 'U', 'O')):
            X, y = check_X_y(X, y, dtype=[str])

        else:
            X, y = check_X_y(X, y, dtype=[np.int32, np.int64])

        return X, y

    def execute_multiple_runs(self, X, y, target_class, T, pool_size: Union[str, int] = 1, starting_seed=None,
                              **kwargs):
        if starting_seed is None:
            raise ValueError("since i'm debugging, i don't want this")
            # starting_seed = np.random.randint(0, 1000000)

        Xs = []
        ys = []

        for t in range(T):
            if self.use_bootstrap:
                X_resampled, y_resampled = resample(X, y, seed=t + starting_seed)
                X_resampled = X_resampled.copy()
                y_resampled = y_resampled.copy()

                Xs.append(X_resampled)
                ys.append(y_resampled)
            else:
                random_indexes = np.random.RandomState(seed=t + starting_seed).permutation(len(X))
                X_perm = X[random_indexes]  # .copy()
                y_perm = y[random_indexes]  # .copy()
                Xs.append(X_perm)
                ys.append(y_perm)

        @self.memory.cache
        def cached_base_method(X, y, target_class):
            return self.base_method(X, y, target_class)

        method = cached_base_method  # if self.memory is not None else self.base_method

        outputs = []
        if pool_size == 'auto':
            t = time.time()
            outputs = [method(Xs[0], ys[0], target_class)]
            if time.time() - t > 1:
                pool_size = max(multiprocessing.cpu_count() - 1, 1)
                outputs.extend(self.run_with_pool(Xs[1:], ys[1:], target_class, pool_size=pool_size))
            else:
                pool_size = 1
                outputs.extend([method(X, y, target_class) for X, y in zip(Xs[1:], ys[1:])])

        elif pool_size > 1:
            outputs.extend(self.run_with_pool(Xs, ys, target_class, pool_size=pool_size))

        else:
            outputs.extend([method(X, y, target_class) for X, y in zip(Xs, ys)])

        return outputs

    def run_with_pool(self, Xs, ys, target_class, pool_size=2):
        if self.verbose > 5:
            print("Using multiprocessing with {} workers".format(pool_size))

        with Pool(pool_size) as p:
            outputs = p.starmap(self.base_method, zip(Xs, ys, repeat(target_class)))

        return outputs

    # def transform(self, X):
    #     return X, self

    def fit(self, X, y, skip_checks=False):
        if not skip_checks:
            X, y = self.checks_for_base_method(X, y)

        if self.encoding == 'ohe':
            self.enc_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X = self.enc_.fit_transform(X)

        # lazy fix, but should check why ordinal constraints are created
        if self.to_string:
            X = X.astype(str)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        if self.target_class is None:
            self.target_class = max(self.classes_)

        if len(self.classes_) == 1:
            self.target_class_ = self.target_class
            self.rule_sets_ = []
            self.bins_ = []

        if len(self.classes_) == 2:
            self.target_class_ = self.target_class
            self.other_class_ = (set(self.classes_) - {self.target_class_}).pop() if len(self.classes_) > 1 else None

            self.rule_sets_ = self.execute_multiple_runs(X, y, self.target_class, T=self.T, pool_size=self.pool_size,
                                                         starting_seed=self.random_state)

            self.prune_rule_sets()

            self.counter_ = self.alpha_representation()

        else:
            current_subclass = type(self)
            assert current_subclass != BayesPointClassifier, "BayesPointClassifier is an abstract class. "
            clf = current_subclass(verbosity=self.verbose, **self.kwargs)

            self.ovr_ = OneVsRestClassifier(clf).fit(X, y)

        # suggest k rules using best k
        # if self.find_best_k:
        # bp_acc = (self.predict(X, strategy='bp') == y).mean()
        # self.suggested_k_ = None
        if self.find_best_k:
            self.suggested_k_ = self.compute_suggested_k_faster(X, y, already_encoded=True)

        # if self.verbose > 2:
        #     print("Best-k search started. Accuracy threshold: {}".format(self.threshold_acc))
        # for k in range(1, len(self.counter_)):
        #     new_acc = (self.predict(X, 'best-k', n_rules=k) == y).mean()
        #     if self.verbose > 5:
        #         print("Trying k={0} of {1}, acc={2:.4f}, bp_acc={3:.4f}".format(k, len(self.counter_), new_acc, bp_acc))
        #
        #     if new_acc >= self.threshold_acc * bp_acc:
        #         self.suggested_k_ = k
        #         break
        # if self.suggested_k_ is None:
        #     self.suggested_k_ = len(self.counter_)

        return self

    def rule_fire_matrix(self, X):
        # X rows by number of unique rules
        rule_fire = np.zeros((X.shape[0], len(self.counter_)))
        freq_sum = 0
        for i in range(len(X)):
            if self.verbose > 5:
                print("\rcomputing rule fire, {0:.2f}%".format(i / len(X) * 100), end="")
            for j, (rule, freq) in enumerate(self.counter_.most_common(len(self.counter_))):
                rule_fire[i, j] = rule.covers(X[i]) * freq
                freq_sum += freq
        return rule_fire

    def classification_matrix(self, X, rule_fire):
        return np.cumsum(rule_fire, axis=1) > self.T // 2
        # classifications_with_k_rules = np.zeros((X.shape[0], len(self.counter_)))
        # for j in range(rule_fire.shape[1]):
        #     if self.verbose > 5:
        #         print("\rcomputing classification {0:.2f}%".format(j / rule_fire.shape[1] * 100), end="")
        #
        #     classifications_with_k_rules[:, j] = np.sum(rule_fire[:, :j + 1], axis=1) > self.T // 2
        # return classifications_with_k_rules

    def compute_suggested_k_faster(self, X, y: np.ndarray, already_encoded=False):
        if self.verbose > 5:
            print("predict with bp for selecting bp_acc. Accuracy threshold: {}".format(self.threshold_acc))

        # bp_acc = (self.predict(X, strategy='bp', already_encoded=already_encoded) == y).mean()
        self.suggested_k_ = None

        if self.verbose > 2:
            print("Best-k search started. Accuracy threshold: {}".format(self.threshold_acc))

        # compute rule fire
        rule_fire = self.rule_fire_matrix(X)

        # results
        classifications_with_k_rules = self.classification_matrix(X, rule_fire)

        bp_acc = (classifications_with_k_rules[:, -1] == y).mean()

        for k in range(len(self.counter_)):
            new_acc = (classifications_with_k_rules[:, k] == y).mean()
            if self.verbose > 12:
                print(
                    "\rTrying k={0} of {1}, acc={2:.4f}, bp_acc={3:.4f}".format(k, len(self.counter_), new_acc, bp_acc),
                    end='')

            if new_acc >= self.threshold_acc * bp_acc:
                # since k is an index, we need to add 1 to get the number of rules
                return k + 1

        return len(self.counter_)

    def validation(self, X, already_encoded=False):

        # Check is fit had been called
        check_is_fitted(self, ['rule_sets_', 'ovr_'], all_or_any=any)

        # Input validation
        X = check_array(X, dtype=None)
        assert isinstance(X, np.ndarray)

        # convert to ohe if needed
        if self.encoding == 'ohe' and not already_encoded:
            X = self.enc_.transform(X)
        if self.to_string:
            X = X.astype(str)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('the number of features in predict is different from the number of features in fit')
        return X

    def predict(self, X: np.ndarray, strategy: str = 'bp', already_encoded=False, **kwargs) -> np.ndarray:
        """
        Predicts the target class for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        strategy : str
            The strategy to use for prediction. Can be one of the following:
            - 'bp': Use the Bayes Point classifier.
            - 'best-k': Use the best k rules.
            - 'bo': Use the Bayes Optimal classifier.
            - 'single': Use a single classifier (T=1).

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        X = self.validation(X, already_encoded=already_encoded)

        # lazy patch to pass tests
        if len(self.classes_) == 1:
            return np.array([self.target_class_ for _ in X])

        # I'm using the ovr classifier for multiclass
        if len(self.classes_) > 2:
            return self.ovr_.predict(X)

        if len(self.classes_) == 2 and strategy == 'single':
            assert len(self.rule_sets_) >= 1

            return np.array(
                [self.target_class_ if (any([rule.covers(row) for rule in self.rule_sets_[0]])) else self.other_class_
                 for row in X])

        if len(self.classes_) == 2 and strategy == 'bo':
            return self.predict_by_rules(X, 'bo')

        if len(self.classes_) == 2 and strategy == 'old-bo':
            return self.predict_bo_old(X)

        if len(self.classes_) == 2 and strategy == 'bp':
            return self.predict_by_rules(X, 'bp', batch_size=kwargs.get('batch_size', 1000))

            # this is equivalent
            # most_freq_rules = self.counter_.most_common(99999999)
            # alpha_rules = sum(self.counter_.values())
            # alpha_k_rules = sum([alpha for r, alpha in most_freq_rules])
            # gamma_k = alpha_k_rules / alpha_rules
            #
            # return np.array([self.target_class_
            #                  if self.predict_one_with_best_k(x, most_freq_rules, gamma_k, self.T)
            #                  else self.other_class_ for x in X])

        # should be equivalent to below
        if len(self.classes_) == 2 and strategy == 'old-bp':
            return self.predict_bp_old(X)

        if len(self.classes_) == 2 and strategy == 'best-k':
            n_rules = kwargs.get('n_rules', self.suggested_k_)
            if n_rules is None:
                raise ValueError('call compute_suggested_k_faster before using best-k strategy')
            most_freq_rules = self.counter_.most_common(n_rules)
            alpha_rules = sum(self.counter_.values())
            alpha_k_rules = sum([alpha for r, alpha in self.counter_.most_common(n_rules)])
            gamma_k = alpha_k_rules / alpha_rules

            return np.array([self.target_class_
                             if self.predict_one_with_best_k(x, most_freq_rules, gamma_k, self.T)
                             else self.other_class_ for x in X])

    def predict_bp_old(self, X):
        h_bp = callable_rules_bp(self.rule_sets_)
        values = [h_bp(x) for x in X]
        return np.array([self.target_class_ if v > self.T / 2 else self.other_class_ for v in values])

    def predict_bo_old(self, X):
        rules_bo = callable_rules_bo(self.rule_sets_)
        values = [sum([ht(x) for ht in rules_bo]) for x in X]
        return np.array([self.target_class_ if np.sign(v) > 0 else self.other_class_ for v in values])

    def predict_all(self, X):
        # compute rule fire
        rule_fire = self.rule_fire_matrix(X)

        # results
        classifications_with_k_rules = self.classification_matrix(X, rule_fire)

        bp = classifications_with_k_rules[:, -1]
        best_k = classifications_with_k_rules[:, self.suggested_k_ - 1]
        bo = self.predict_by_rules(X, 'bo')
        return bo, bp, best_k

    def predict_by_rules(self, X, strategy='bp', batch_size=1000):
        if strategy != 'bo':
            result = []
            for i in range(0, len(X), batch_size):
                # compute rule fire
                rule_fire = self.rule_fire_matrix(X[i:i + batch_size])

                # results
                classifications_with_k_rules = self.classification_matrix(X, rule_fire)

                if strategy == 'bp':
                    result.extend(classifications_with_k_rules[:, -1])
                elif strategy == 'best-k':
                    result.extend(classifications_with_k_rules[:, self.suggested_k_ - 1])
            return np.array(result)
        elif strategy == 'bo':
            # X rows by number of dnfs
            rule_fire = np.zeros((X.shape[0], len(self.rule_sets_)))

            for i in range(len(X)):
                if self.verbose > 5:
                    print("\rcomputing rule fire, {0:.2f}%".format(i / len(X) * 100), end="")
                for j, rule_set in enumerate(self.rule_sets_):
                    covers = False
                    for rule in rule_set:
                        if rule.covers(X[i]):
                            covers = True
                            break
                    rule_fire[i, j] = 1 if covers else -1

            # compute bo
            classifications = np.sign(np.sum(rule_fire, axis=1)) > 0
            return classifications

    def compute_suggested_k(self, X, y, already_encoded=False):
        bp_acc = (self.predict(X, strategy='bp', already_encoded=already_encoded) == y).mean()
        suggested_k_ = None

        for k in range(1, len(self.counter_)):
            new_acc = (self.predict(X, 'best-k', n_rules=k, already_encoded=already_encoded) == y).mean()
            if new_acc >= self.threshold_acc * bp_acc:
                suggested_k_ = k
                break
        return suggested_k_

    def predict_one_with_best_k(self, x, most_freq_rules, gamma_k, T):
        vote = 0
        for rule, alpha in most_freq_rules:
            vote += rule.covers(x) * alpha
        return vote > gamma_k * T / 2

    # def fit_find_rs_on_rulesets(self, X, y):
    #     # check if fitted
    #     if self.rule_sets_ is None:
    #         raise ValueError("You must fit the model before calling this method.")
    #
    #     from bpllib import FindRsClassifier
    #     clf = FindRsClassifier()
    #     clf.to_string = False
    #     # filter from X only the corresponding target class
    #     # todo this is wrong
    #
    #     N = [x for x, y1 in zip(X, y) if y1 != self.target_class_]
    #
    #     P = list(self.counter_.keys())
    #     X = list(N) + list(P)
    #     y = [self.other_class_] * len(N) + [self.target_class_] * len(P)
    #
    #     data = list(zip(X, y))
    #
    #     random.shuffle(data)
    #
    #     X, y = zip(*data)
    #
    #     rulesets = clf.base_method(X, y, target_class=self.target_class_)
    #     print(rulesets)

    def prune_rule_sets(self):
        if self.max_rules is None:
            return
        if self.max_rules is not None:
            for i, rule_set in enumerate(self.rule_sets_):
                if len(rule_set) > self.max_rules:
                    # possibly, add different pruning strategies
                    self.rule_sets_[i] = rule_set[:self.max_rules]
