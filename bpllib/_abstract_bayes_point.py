import multiprocessing
import time
from itertools import repeat
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
from tqdm import tqdm

from bpllib._bp import callable_rules_bo, callable_rules_bp
from .utils import resample


class BayesPointClassifier(ClassifierMixin, BaseEstimator):
    '''
    Abstract class for all bayesian point classifiers.
    To implement new Bayes Point Classifiers, you should subclass this class.
    '''
    description = '(Abstract) Bayes Point Classifier'

    def __init__(self, T=3, verbose=0, threshold_acc=0.99, target_class=None, pool_size='auto', find_best_k=False,
                 random_state=None, **kwargs):
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
            starting_seed = np.random.randint(0, 1000000)

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

        outputs = []
        if pool_size == 'auto':
            t = time.time()
            outputs = [self.base_method(Xs[0], ys[0], target_class)]
            if time.time() - t > 1:
                pool_size = max(multiprocessing.cpu_count() - 1, 1)
                outputs.extend(self.run_with_pool(Xs[1:], ys[1:], target_class, pool_size=pool_size))
            else:
                pool_size = 1
                outputs.extend([self.base_method(X, y, target_class) for X, y in zip(Xs[1:], ys[1:])])

        elif pool_size > 1:
            outputs.extend(self.run_with_pool(Xs, ys, target_class, pool_size=pool_size))

        else:
            outputs.extend([self.base_method(X, y, target_class) for X, y in zip(Xs, ys)])

        return outputs

    def run_with_pool(self, Xs, ys, target_class, pool_size=2):
        if self.verbose > 5:
            print("Using multiprocessing with {} workers".format(pool_size))

        with Pool(pool_size) as p:
            outputs = p.starmap(self.base_method, zip(Xs, ys, repeat(target_class)))

        return outputs

    def fit(self, X, y):

        X, y = self.checks_for_base_method(X, y)

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

        self.suggested_k_ = self.compute_suggested_k_faster(X, y)

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

    def compute_suggested_k_faster(self, X, y: np.ndarray):
        bp_acc = (self.predict(X, strategy='bp') == y).mean()
        self.suggested_k_ = None

        if self.verbose > 2:
            print("Best-k search started. Accuracy threshold: {}".format(self.threshold_acc))

        # X rows by number of unique rules
        rule_fire = np.zeros((X.shape[0], len(self.counter_)))
        for i in range(len(X)):
            for j, rule in enumerate(self.counter_):
                rule_fire[i, j] = rule.covers(X[i])

        # results
        classifications_with_k_rules = np.zeros((X.shape[0], len(self.counter_)))
        for j in range(rule_fire.shape[1]):
            classifications_with_k_rules[:, j] = np.sign(np.sum(rule_fire[:, :j], axis=1))

        for k in range(1, len(self.counter_)):
            new_acc = (classifications_with_k_rules[:, k] == y).mean()
            if self.verbose > 5:
                print("Trying k={0} of {1}, acc={2:.4f}, bp_acc={3:.4f}".format(k, len(self.counter_), new_acc, bp_acc))

            if new_acc >= self.threshold_acc * bp_acc:
                return k

        if self.suggested_k_ is None:
            return len(self.counter_)
        return self.suggested_k_

    def predict(self, X: np.ndarray, strategy: str = 'bp', **kwargs) -> np.ndarray:
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

        # Check is fit had been called
        check_is_fitted(self, ['rule_sets_', 'ovr_'], all_or_any=any)

        # Input validation
        X = check_array(X, dtype=None)
        assert isinstance(X, np.ndarray)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('the number of features in predict is different from the number of features in fit')

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
            rules_bo = callable_rules_bo(self.rule_sets_)
            values = [sum([ht(x) for ht in rules_bo]) for x in X]
            return np.array([self.target_class_ if np.sign(v) > 0 else self.other_class_ for v in values])

        if len(self.classes_) == 2 and strategy == 'bp':
            h_bp = callable_rules_bp(self.rule_sets_)
            values = [h_bp(x) for x in X]
            return np.array([self.target_class_ if v > self.T / 2 else self.other_class_ for v in values])

        if len(self.classes_) == 2 and strategy == 'best-k':
            n_rules = kwargs.get('n_rules', self.suggested_k_)
            most_freq_rules = self.counter_.most_common(n_rules)
            alpha_rules = sum(self.counter_.values())
            alpha_k_rules = sum([alpha for r, alpha in self.counter_.most_common(n_rules)])
            gamma_k = alpha_k_rules / alpha_rules

            return np.array([self.target_class_
                             if self.predict_one_with_best_k(x, most_freq_rules, gamma_k, self.T)
                             else self.other_class_ for x in X])

    def compute_suggested_k(self, X, y):
        bp_acc = (self.predict(X, strategy='bp') == y).mean()
        suggested_k_ = None

        for k in range(1, len(self.counter_)):
            new_acc = (self.predict(X, 'best-k', n_rules=k) == y).mean()
            if new_acc >= self.threshold_acc * bp_acc:
                suggested_k_ = k
                break
        return suggested_k_

    def predict_one_with_best_k(self, x, most_freq_rules, gamma_k, T):
        vote = 0
        for rule, alpha in most_freq_rules:
            vote += rule.covers(x) * alpha
        return vote > gamma_k * T / 2
