"""
This is a module to be used as a reference for building other modules
"""
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from bpllib._bp import callable_rules_bo, callable_rules_bp, alpha_representation, predict_one_with_bo, \
    predict_one_with_bp, predict_one_with_best_k


class Rule:
    '''
    A rule describes a condition for classifying a data point to the target class.
    It is made up by many constraints, each one considering a single feature.
    '''

    def __init__(self, constraints: dict):
        self.constraints = constraints  # was set()
        self.columns = None

    def __mul__(self, other):
        '''
        Returns the conjunction of two rules
        Parameters
        ----------
        other: the other rule

        -------

        '''
        new_constraints = {}
        for index in set(self.constraints.keys()).union(set(other.constraints.keys())):
            c1 = self.constraints.get(index)
            c2 = other.constraints.get(index)
            if c1 is None:
                new_constraints[index] = c2
            elif c2 is None:
                new_constraints[index] = c1
            else:
                c = c1 & c2
                if c is None:
                    return None
                else:
                    new_constraints[index] = c
        return Rule(new_constraints)

    def generalize(self, x: np.array):
        '''
        Generalizes a rule w.r.t. a given input.
        In practice, we relax constraints by removing them or dilating their bounds.
        Parameters
        ----------
        x np.array which contains a single example

        Returns the generalization of the current rule compared to the current example.
        -------

        '''
        new_constraints = {}
        for idx, constraint in self.constraints.items():
            new_constraint = constraint.generalize(x)
            if new_constraint is not None:
                new_constraints[idx] = new_constraint
        return Rule(new_constraints)

    def covers(self, x: np.array):
        '''
        Checks if the current rule covers an input example
        Parameters
        ----------
        x np.array containing a single example

        Returns True if x is covered
        -------

        '''
        for constraint in self.constraints.values():
            if not constraint.satisfied(x):
                return False
        return True

    def covers_any(self, data: np.array, tol=0):
        '''
        Checks if any data point in the data array is covered by this rule.
        Parameters
        ----------
        data: np.array which contains the data to be processed
        tol: the hyperparameter which enables unpure bins that contain negative examples.

        -------

        '''
        not_covered = []
        for i, data_point in enumerate(data):
            if self.covers(data_point):
                not_covered.append(i)
                if len(not_covered) > tol:
                    return not_covered
        return []

    def covers_all(self, data: np.array):
        '''
        Checks if all data points in the data array are covered by this rule.
        Parameters
        ----------
        data: np.array which contains the data to be processed

        -------

        '''
        for data_point in data:
            if not self.covers(data_point):
                return False
        return True

    def __call__(self, *args, **kwargs):
        return self.covers(*args, **kwargs)

    def examples_covered(self, X):
        return X[[self.covers(x) for x in X]]

    def examples_not_covered(self, X):
        return X[[not self.covers(x) for x in X]]

    def __repr__(self):
        return " ^ ".join(str(c) for c in sorted(self.constraints.values(), key=lambda c: c.index))

    def __len__(self):
        return len(self.constraints)

    def values_count(self):
        return sum(len(c) for c in self.constraints.values())

    def __eq__(self, other):
        return tuple(sorted(self.constraints.items())) == tuple(sorted(other.constraints.items()))

    def __hash__(self):
        return hash(tuple(sorted(self.constraints.items())))

    def str_with_column_names(self, columns):
        self.columns = columns
        reprs = []
        for c in self.constraints.values():
            column = columns[c.index]
            reprs.append(f"{column}={c.value}")
        return " ^ ".join(reprs)


class RuleByExample(Rule):
    '''
    It is useful to instance a rule starting from a given input data point.
    '''

    def __init__(self, example: np.array):
        x = example.flatten()
        constraints = {}

        for i, feature in enumerate(x):
            # we create discrete constraints for strings (for sure)
            # TODO using ints for discrete constraints is very quick and dirty, but we may regret it down the line
            if isinstance(feature, (str, int, np.int32, np.int64)):
                constraints[i] = DiscreteConstraint(value=feature, index=i)
            # we create ordinal constraints for floats (for sure)
            elif isinstance(feature, (float, np.float32, np.float64)):
                constraints[i] = OrdinalConstraint(value_min=feature, value_max=feature, index=i)
            else:
                raise NotImplementedError(f'found {type(feature)} for {feature}')

        super().__init__(constraints)


class Constraint:
    '''
    A constraint always has an index, that is used to pick up the i-th feature which is constrained.
    '''

    def __init__(self, index=None):
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def satisfied(self, x):
        '''
        Checks if a constraint is satisfied.
        Parameters
        ----------
        x np.array data point to check if is satisfied

        Returns True if the constraint is satisfied for x
        -------

        '''
        raise NotImplementedError()

    def generalize(self, x):
        '''
        Returns the generalization of a constraint, that can be None if the constraint would be always satisfied.
        Parameters
        ----------
        x np.array data point to use for the generalization

        Returns a new constraint, or None
        -------

        '''
        raise NotImplementedError()


class AgainstDiscreteConstraint(Constraint):
    '''
    A constraint which is satisfied if the value of the i-th feature is different from the specified value.
    '''

    def __init__(self, value, index):
        super().__init__(index)
        self.value = value

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def __repr__(self):
        return f'X[{self.index}]!={self.value}'

    def satisfied(self, x):
        return x[self.index] != self.value

    def generalize(self, x):
        if self.value != x[self.index]:
            return self
        return None


class DiscreteConstraint(Constraint):
    '''
    A discrete constraint contains a value, that needs to be checked for equality in a constraint check.
    '''

    def __init__(self, value=None, index=None):
        self.value = value
        super().__init__(index=index)

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def __hash__(self):
        return hash((self.index, self.value))

    def __len__(self):
        return 1

    def __and__(self, other):
        if isinstance(other, DiscreteOrConstraint):
            if self.value in other.values:
                return self
            else:
                return None
        if self.value == other.value:
            return self
        return None

    def satisfied(self, x):
        return x[self.index] == self.value

    def generalize(self, x):
        if self.value == x[self.index]:
            return self
        return None

    def __repr__(self):
        return f'X[{self.index}] == {self.value.__repr__()}'


class DiscreteOrConstraint(Constraint):
    def __init__(self, values, index):
        super().__init__(index=index)
        self.values = set(values)

    def __eq__(self, other):
        return super().__eq__(other) and self.values == other.values

    def __len__(self):
        return len(self.values)

    def __and__(self, other):
        if isinstance(other, DiscreteConstraint):
            if other.value in self.values:
                return other
            else:
                return None
        intersection = self.values.intersection(other.values)
        if len(intersection) > 1:
            return DiscreteOrConstraint(values=intersection, index=self.index)
        if len(intersection) == 1:
            return DiscreteConstraint(value=intersection.pop(), index=self.index)
        return None

    def satisfied(self, x):
        return x[self.index] in self.values

    def __repr__(self):
        return f'X[{self.index}] in {self.values}'

    def __hash__(self):
        return hash((self.index, tuple(sorted(self.values))))


class OrdinalConstraint(Constraint):
    '''
    An ordinal constraint contains a value_min and a value_max, that define a bound for continuous values.
    '''

    def __init__(self, value_min=None, value_max=None, index=None):
        self.value_min = value_min
        self.value_max = value_max
        super().__init__(index=index)

    def __repr__(self):
        return f'{self.value_min} <= X[{self.index}] <= {self.value_max}'

    def satisfied(self, x):
        return self.value_min <= x[self.index] <= self.value_max

    def generalize(self, x):
        return OrdinalConstraint(value_min=min(self.value_min, x[self.index]),
                                 value_max=max(self.value_max, x[self.index]),
                                 index=self.index)

    def __and__(self, other):
        if not isinstance(other, OrdinalConstraint):
            raise TypeError("Only ordinal constraints can be combined with an ordinal constraint")
        return OrdinalConstraint(value_min=max(self.value_min, other.value_min),
                                 value_max=min(self.value_max, other.value_max),
                                 index=self.index)


class FindRsClassifier(ClassifierMixin, BaseEstimator):
    """ A classifier which implements Find-RS...

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    tol : int, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, tol=0, T=1, strategy=None, threshold_acc=0.99):
        self.tol = tol
        self.T = T
        self.strategy = strategy
        self.threshold_acc = threshold_acc
        if T > 1 and strategy is None:
            raise ValueError('If T > 1, a strategy must be specified [bo|bp]')

    def alpha_representation(self):
        from collections import Counter
        cnt = Counter()
        for ruleset in self.rulesets_:
            curr_counts = Counter(ruleset)
            cnt.update(curr_counts)
        return cnt

    def best_k_rules(self, k=20):
        cnt = self.alpha_representation()
        return cnt.most_common(k)

    # used to specify to estimator_checks that we accept strings and should not fail, see #11401 of scikit-learn docs
    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

    def fit(self, X, y, target_class=None, pool_size=1, find_best_k=False, starting_seed=0, optimization=True):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        self.suggested_k_ = None

        # Check that X and y have correct shape
        # we accept strings
        if np.any(np.array(X).dtype.kind == 'f'):
            warnings.warn("FindRsClassifier does not fully support float values yet.")
            X, y = check_X_y(X, y, dtype=None)
        # check for strings
        elif np.any(np.array(X).dtype.kind in ('S', 'U', 'O')):
            X, y = check_X_y(X, y, dtype=[str])

        else:
            X, y = check_X_y(X, y, dtype=[np.int32, np.int64])

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        # if multiclass or len(self.classes_) > 2:
        #    raise NotImplementedError('Multiclass implementation not available yet')

        if target_class is None:
            target_class = max(self.classes_)

        if len(self.classes_) == 1:
            self.target_class_ = target_class
            self.rules_, self.bins_ = FindRsClassifier.find_rs(X, y, target_class)

        if len(self.classes_) == 2:
            self.target_class_ = target_class
            self.other_class_ = (set(self.classes_) - {self.target_class_}).pop() if len(self.classes_) > 1 else None

            if self.T == 1:
                self.rules_, self.bins_ = FindRsClassifier.find_rs(X, y, target_class, optimization=optimization)
            else:
                outputs = FindRsClassifier.find_rs_with_multiple_runs(X, y, target_class, T=self.T, pool_size=pool_size,
                                                                      starting_seed=starting_seed)
                self.rulesets_ = [D for D, B in outputs]
                self.counter_ = alpha_representation(self.rulesets_)


        else:
            self.ovr_ = OneVsRestClassifier(FindRsClassifier(tol=self.tol, T=self.T)).fit(X, y)

        # suggest k rules using best k
        if find_best_k:
            bp_acc = (self.predict(X, strategy='bp') == y).mean()
            self.suggested_k_ = None

            for k in range(1, len(self.counter_)):
                new_acc = (self.predict(X, 'best-k', n_rules=k) == y).mean()
                if new_acc >= self.threshold_acc * bp_acc:
                    self.suggested_k_ = k
                    break

        # best_acc = 0
        # MAX_PATIENCE = 10
        # patience = MAX_PATIENCE
        #
        # self.suggested_k_ = 0
        # if find_best_k:
        #     for k in range(1, len(self.counter_)):
        #
        #         if patience == 0:
        #             break
        #         most_freq_rules = self.counter_.most_common(k)
        #         y_train_pred = [self.predict_one(x, strategy='best-k', n_rules=k, most_freq_rules=most_freq_rules)
        #                         for x in X]
        #         acc = np.mean(y_train_pred == y)
        #         if acc > best_acc:
        #             self.suggested_k_ = k
        #         else:
        #             patience -= 1

        # Return the classifier
        return self

    def predict(self, X: np.array, strategy=None, **kwargs):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """

        # Check is fit had been called
        check_is_fitted(self, ['rules_', 'bins_', 'ovr_', 'rulesets_'], all_or_any=any)

        if strategy is None:
            strategy = self.strategy

        # Input validation
        X = check_array(X, dtype=None)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError('the number of features in predict is different from the number of features in fit')

        # lazy patch to pass tests
        if len(self.classes_) == 1:
            return np.array([self.target_class_ for _ in X])

        if len(self.classes_) > 2:
            return self.ovr_.predict(X)

        if len(self.classes_) == 2 and strategy is None:
            return np.array(
                [self.target_class_ if (any([rule.covers(row) for rule in self.rules_])) else self.other_class_
                 for row in X])

        if len(self.classes_) == 2 and strategy == 'bo':
            rules_bo = callable_rules_bo(self.rulesets_)
            values = [sum([ht(x) for ht in rules_bo]) for x in X]
            return np.array([self.target_class_ if np.sign(v) > 0 else self.other_class_ for v in values])

        if len(self.classes_) == 2 and strategy == 'bp':
            h_bp = callable_rules_bp(self.rulesets_)
            values = [h_bp(x) for x in X]
            return np.array([self.target_class_ if v > self.T / 2 else self.other_class_ for v in values])

        if len(self.classes_) == 2 and strategy == 'best-k':
            n_rules = kwargs.get('n_rules', 20)
            most_freq_rules = self.counter_.most_common(n_rules)
            return np.array([self.predict_one(x, most_freq_rules=most_freq_rules,
                                              strategy=strategy, n_rules=n_rules) for x in X])

    @staticmethod
    def find_rs(X, y, target_class, tol=0, optimization=None):
        train_p = list(X[y == target_class])
        train_n = list(X[y != target_class])

        D, B, k = [], [], 0

        while len(train_p) > 0:

            first = train_p.pop(0)
            B.append([first])
            D.append(RuleByExample(first))

            incompatibles = []
            while len(train_p) > 0:
                r = D[-1]
                p = train_p.pop(0)

                new_r = r.generalize(p)
                not_covered = new_r.covers_any(train_n, tol=tol)
                if not not_covered:
                    D[-1] = new_r  # RuleByExample(p) ??
                    B[-1].append(p)
                else:
                    incompatibles.append(p)  # occhio all'ordine!

                    if optimization:
                        # todo extend to >1 not_covered
                        train_n = np.insert(train_n, 0, train_n[not_covered[0]], axis=0)
                        train_n = np.delete(train_n, not_covered[0] + 1, axis=0)

            train_p = incompatibles
        D, B = FindRsClassifier._prune(D, B)
        return D, B

    @staticmethod
    def _prune(D, B):
        old_len = len(D)

        i = 0
        while i < len(D):
            current_b = list(B[i].copy())
            for j in range(i + 1, len(D)):
                next_rule = D[j]
                k = 0

                while k < len(current_b):
                    p = current_b[k]
                    if next_rule.covers(p):
                        current_b.pop(k)
                    else:
                        k += 1
            if not current_b:
                B.pop(i)
                D.pop(i)
            else:
                i += 1

        if old_len > len(D):
            pass
            # print("pruned from " + str(old_len) + " to " + str(len(D)) + " rules")
        return D, B

    @staticmethod
    def _find_rs_iteration(X, y, target_class, t, tol=0, starting_seed=0):
        np.random.seed(t + starting_seed)

        random_indexes = np.random.RandomState(seed=t + starting_seed).permutation(len(X))
        X_perm = X[random_indexes].copy()
        y_perm = y[random_indexes].copy()

        Dt, Bt = FindRsClassifier.find_rs(X_perm, y_perm, target_class, tol=tol)

        return Dt, Bt

    @staticmethod
    def find_rs_with_multiple_runs(X, y, target_class, tol=0, pool_size=1, T=1, starting_seed=0):
        if pool_size > 1:
            # TODO why doesn't it work?
            with Pool(pool_size) as p:
                outputs = p.map(partial(FindRsClassifier._find_rs_iteration, X, y, target_class, tol=tol,
                                        starting_seed=starting_seed), range(T))
        else:
            outputs = [FindRsClassifier._find_rs_iteration(X, y, target_class, t, tol=tol, starting_seed=starting_seed)
                       for t in tqdm.tqdm(range(T))]
        return outputs

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['rules_', 'bins_', 'ovr_', 'rulesets_'], all_or_any=any)

        if len(self.classes_) > 2:
            return self.ovr_.predict_proba(X)  # how can i remove the warning?
        elif len(self.classes_) == 2:
            return np.array(
                [[float(y_pred == self.classes_[0]), float(y_pred == self.classes_[1])] for y_pred in self.predict(X)])
        else:
            return np.ones(X).reshape(1, -1)

    @staticmethod
    def process_rulesets(DBs, strategy='bo'):
        '''
        :param DBs: list of tuples (D, B)
        :param strategy: 'bo' or 'bp'
        :return: list of rulesets
        '''

        if strategy == 'bo':
            import itertools
            result = [D for D, B in DBs]
            return list(result)
        else:
            raise NotImplementedError(f'strategy {strategy} not implemented')

    def predict_one(self, x, most_freq_rules=None, strategy='bo', n_rules=None):
        if strategy == 'bo':
            return predict_one_with_bo(self.rulesets_, x, self.target_class_, self.other_class_)
        elif strategy == 'bp':
            return predict_one_with_bp(self.rulesets_, x, self.target_class_, self.other_class_, self.T)
        elif strategy == 'best-k':
            return predict_one_with_best_k(x, n_rules, n_rules, self.counter_, most_freq_rules, self.target_class_,
                                           self.other_class_, self.T)
        else:
            raise NotImplementedError(f'strategy {strategy} not implemented')
