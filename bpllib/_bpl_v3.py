"""
This is a module to be used as a reference for building other modules
"""
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from numba.experimental import jitclass
import numba as nb


@nb.njit
def _nb_covers(constraints, x):
    for constraint in constraints:
        if not constraint.satisfied(x):
            return False
    return True


@nb.njit
def _nb_covers_any(constraints, data, tol=0):
    not_covered = 0
    for i, data_point in enumerate(data):
        if _nb_covers(constraints, data_point):
            not_covered += 1
            if not_covered > tol:
                return not_covered
    return 0


class Constraint:
    index: int
    '''
    A constraint always has an index, that is used to pick up the i-th feature which is constrained.
    '''

    def __init__(self, index=None):
        self.index = index

    def __repr__(self):
        raise NotImplementedError

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

    def __repr__(self):
        return f'X[{self.index}]!={self.value}'

    def satisfied(self, x):
        return x[self.index] != self.value

    def generalize(self, x):
        if self.value != x[self.index]:
            return self
        return None


@jitclass
class DiscreteConstraint:
    value: nb.from_dtype(np.dtype('<U1'))
    index: int
    '''
    A discrete constraint contains a value, that needs to be checked for equality in a constraint check.
    '''

    def __init__(self, value=None, index=None):
        self.value = value
        self.index = index

    def satisfied(self, x):
        return x[self.index] == self.value

    def generalize(self, x):
        if self.value == x[self.index]:
            return self
        return None

    def __repr__(self):
        return f'X[{self.index}] == {self.value}'


@jitclass
class OrdinalConstraint:
    index: int
    value_min: float
    value_max: float
    '''
    An ordinal constraint contains a value_min and a value_max, that define a bound for continuous values.
    '''

    def __init__(self, value_min=0.0, value_max=0.0, index=0):
        self.value_min = value_min
        self.value_max = value_max
        self.index = index

    def __repr__(self):
        return f'{self.value_min} <= X[{self.index}] <= {self.value_max}'

    def satisfied(self, x):
        return self.value_min <= x[self.index] <= self.value_max

    def generalize(self, x):
        return OrdinalConstraint(value_min=min(self.value_min, x[self.index]),
                                 value_max=max(self.value_max, x[self.index]),
                                 index=self.index)


@jitclass
class Rule:
    discrete_constraints: List[DiscreteConstraint]
    ordinal_constraints: List[OrdinalConstraint]
    '''
    A rule describes a condition for classifying a data point to the target class.
    It is made up by many constraints, each one considering a single feature.
    '''

    def __init__(self, discrete_constraints, ordinal_constraints):
        # TODO maybe i should include the whole dataset to remove constraints like [min(X[:,0]), max(X[:,0])]
        self.discrete_constraints = discrete_constraints
        self.ordinal_constraints = ordinal_constraints

    @staticmethod
    def ByExampleDiscrete(example: np.array):
        x = example.flatten()
        discrete_constraints = []

        for i, feature in enumerate(x):
            discrete_constraints.append(DiscreteConstraint(value=feature, index=i))

        return Rule(discrete_constraints, [OrdinalConstraint() for x in range(0)])

    @staticmethod
    def ByExampleOrdinal(example: np.array):
        x = example.flatten()
        ordinal_constraints = []

        for i, feature in enumerate(x):
            ordinal_constraints.append(OrdinalConstraint(value_min=feature, value_max=feature, index=i))

        return Rule([DiscreteConstraint() for x in range(0)], ordinal_constraints)


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
        new_ordinal_constraints = []
        new_discrete_constraints = []

        for constraint in self.ordinal_constraints:
            new_constraint = constraint.generalize(x)
            if new_constraint is not None:
                new_ordinal_constraints.append(new_constraint)

        for constraint in self.discrete_constraints:
            new_constraint = constraint.generalize(x)
            if new_constraint is not None:
                new_discrete_constraints.append(new_constraint)
        return Rule(new_discrete_constraints, new_ordinal_constraints)

    def covers(self, x: np.array):
        '''
        Checks if the current rule covers an input example
        Parameters
        ----------
        x np.array containing a single example

        Returns True if x is covered
        -------

        '''
        return _nb_covers(self.discrete_constraints, x) and _nb_covers(self.ordinal_constraints, x)

    def covers_any(self, data: np.array, tol=0):
        '''
        Checks if any data point in the data array is covered by this rule.
        Parameters
        ----------
        data: np.array which contains the data to be processed
        tol: the hyperparameter which enables unpure bins that contain negative examples.

        -------

        '''
        return _nb_covers_any(self.discrete_constraints, data, tol=tol) and _nb_covers_any(self.ordinal_constraints,
                                                                                           data, tol=tol)

    def __repr__(self):
        return " ".join(str(c) for c in sorted(self.discrete_constraints, key=lambda c: c.index)) + " ".join(
            str(c) for c in sorted(self.ordinal_constraints, key=lambda c: c.index))


# def __hash__(self):
#     pass

class BplClassifierV3(ClassifierMixin, BaseEstimator):
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

    def __init__(self, tol=0, T=1, strategy=None):
        self.tol = tol
        self.T = T
        self.strategy = strategy

    # used to specify to estimator_checks that we accept strings and should not fail, see #11401 of scikit-learn docs
    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

    def fit(self, X, y, target_class=None, pool_size=1):
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
        # Check that X and y have correct shape
        # we accept strings
        X, y = check_X_y(X, y, dtype=[str, int, np.int32, np.int64, float, np.float32, np.float64])
        if np.issubdtype(X.dtype, np.number):
            self.use_ordinal = True
        else:
            self.use_ordinal = False
            X = X.astype(str)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        # if multiclass or len(self.classes_) > 2:
        #    raise NotImplementedError('Multiclass implementation not available yet')

        if target_class is None:
            target_class = max(self.classes_)

        if len(self.classes_) == 1:
            self.target_class_ = target_class
            self.D_, self.B_ = BplClassifierV3.find_rs(X, y, target_class, use_ordinal=self.use_ordinal)

        if len(self.classes_) == 2:
            self.target_class_ = target_class
            self.other_class_ = (set(self.classes_) - {self.target_class_}).pop() if len(self.classes_) > 1 else None

            if self.T == 1:
                self.D_, self.B_ = BplClassifierV3.find_rs(X, y, target_class, use_ordinal=self.use_ordinal)
            else:
                outputs = BplClassifierV3.find_rs_with_multiple_runs(X, y, target_class, T=self.T, pool_size=pool_size)
                self.Ds_ = [D for D, B in outputs]

        else:
            self.ovr_ = OneVsRestClassifier(BplClassifierV3(tol=self.tol, T=self.T)).fit(X, y)

        # Return the classifier
        return self

    def predict(self, X: np.array):
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
        check_is_fitted(self, ['D_', 'B_', 'ovr_', 'Ds_'], all_or_any=any)

        # Input validation
        X = check_array(X, dtype=None)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError('the number of features in predict is different from the number of features in fit')

        # lazy patch to pass tests
        if len(self.classes_) == 1:
            return np.array([self.target_class_ for _ in X])

        if len(self.classes_) > 2:
            return self.ovr_.predict(X)

        if len(self.classes_) == 2 and self.strategy is None:
            return np.array(
                [self.target_class_ if (any([rule.covers(row) for rule in self.D_])) else self.other_class_ for row in
                 X])

        if len(self.classes_) == 2 and self.strategy == 'bo':
            rules_bo = BplClassifierV3.callable_rules_bo(self.Ds_)
            values = [sum([ht(x) for ht in rules_bo]) for x in X]
            return np.array([self.target_class_ if np.sign(v) > 0 else self.other_class_ for v in values])

        if len(self.classes_) == 2 and self.strategy == 'bp':
            h_bp = BplClassifierV3.callable_rules_bp(self.Ds_)
            values = [h_bp(x) for x in X]
            return np.array([self.target_class_ if v > self.T / 2 else self.other_class_ for v in values])

    @staticmethod
    def callable_rules_bo(Ds):
        return [lambda x: 1 if any([rule.covers(x) for rule in D]) else -1 for D in Ds]

    @staticmethod
    def callable_rules_bp(Ds):
        # NO ! return [lambda x: sum([rule.covers(x) for rule in D]) for D in Ds]
        return lambda x: sum([rule.covers(x) for D in Ds for rule in D])

    @staticmethod
    def find_rs(X, y, target_class, tol=0, use_ordinal=False):
        train_p = list(X[y == target_class])
        train_n = list(X[y != target_class])

        D, B, k = [], [], 0

        while len(train_p) > 0:

            first = train_p.pop(0)
            B.append([first])
            if use_ordinal:
                D.append(Rule.ByExampleOrdinal(first))
            else:
                D.append(Rule.ByExampleDiscrete(first))

            incompatibles = []
            while len(train_p) > 0:
                r = D[-1]
                p = train_p.pop(0)

                new_r = r.generalize(p)

                if not new_r.covers_any(train_n, tol=tol):
                    D[-1] = new_r  # RuleByExample(p) ??
                    B[-1].append(p)
                else:
                    incompatibles.append(p)  # occhio all'ordine!
            train_p = incompatibles
        D, B = BplClassifierV3._prune(D, B)
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
    def _find_rs_iteration(X, y, target_class, t, tol=0, use_ordinal=False):
        np.random.seed(t)

        random_indexes = np.random.RandomState(seed=t).permutation(len(X))
        X_perm = X[random_indexes].copy()
        y_perm = y[random_indexes].copy()

        Dt, Bt = BplClassifierV3.find_rs(X_perm, y_perm, target_class, tol=tol, use_ordinal=use_ordinal)

        return Dt, Bt

    @staticmethod
    def find_rs_with_multiple_runs(X, y, target_class, tol=0, pool_size=1, T=1, use_ordinal=False):
        if pool_size > 1:
            # TODO why doesn't it work?
            with Pool(pool_size) as p:
                outputs = p.map(
                    partial(BplClassifierV3._find_rs_iteration, X, y, target_class, tol=tol, use_ordinal=use_ordinal),
                    range(T))
        else:
            outputs = [BplClassifierV3._find_rs_iteration(X, y, target_class, t, tol=tol, use_ordinal=use_ordinal) for t
                       in range(T)]
        return outputs

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['D_', 'B_', 'ovr_'], all_or_any=any)

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
