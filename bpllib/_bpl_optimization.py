"""
This is a module to be used as a reference for building other modules
"""
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm as tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class Rule:
    '''
    A rule describes a condition for classifying a data point to the target class.
    It is made up by many constraints, each one considering a single feature.
    '''

    def __init__(self, constraints):
        # TODO maybe i should include the whole dataset to remove constraints like [min(X[:,0]), max(X[:,0])]
        self.constraints = set(constraints)

    def is_same_as(self, rule):
        for c1, c2 in zip(self.constraints, rule.constraints):
            if c1.index != c2.index or c1.value != c2.value:
                return False
        # print("it's the same!")
        return True

    def generalize_rule(self, rule):
        intersect = rule.constraints.intersection(self.constraints)
        if not intersect:
            return None
        return Rule(intersect)

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

        # TODO se ho abc come valori e in un esempio ho a e nell'altro b, quindi vale not c, io comunque tolgo not c.

        new_constraints = []
        # changed = False
        for constraint in self.constraints:
            new_constraint = constraint.generalize(x)
            if new_constraint is not None:
                new_constraints.append(new_constraint)
                # if constraint != new_constraint:
                # we changed a constraint, the rule is changed
                # (not verified for discrete constraints, but may be for continuous ones)
                #     changed = True
            else:
                pass
                # we removed a constraint, the rule is changed
                # changed = True

        # if not changed:
        # return "not changed"
        if not new_constraints:
            return None

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
        for constraint in self.constraints:
            if not constraint.satisfied(x):
                return False
        return True

    def covers_any(self, data: np.array, indexes, tol=0, optimization=False):
        '''
        Checks if any data point in the data array is covered by this rule.
        Parameters
        ----------
        data: np.array which contains the data to be processed
        tol: the hyperparameter which enables unpure bins that contain negative examples.

        -------

        '''

        covered = []
        i = None
        index = None
        for i, index in enumerate(indexes):
            if self.covers(data[index]):
                covered.append(index)
                if len(covered) > tol:
                    break

        if optimization:
            indexes.insert(0, index)
            del indexes[i + 1]

        return covered

    def __repr__(self):
        ordered_constraints = sorted(self.constraints, key=lambda c: c.index)
        indexes = [constraint.index for constraint in ordered_constraints]
        vals = iter([constraint.value for constraint in ordered_constraints])

        return " ".join(["-" if i not in indexes else str(int(next(vals))) for i in range(max(indexes) + 1)])
        # for c in :
        # return " ".join(str(c) )


# def __hash__(self):
#     pass

class ListRule:
    def __init__(self, x):
        self.constraints = list(x)

    def generalize(self, x):
        new_rule = ListRule(self.constraints)
        for i, (ri, xi) in enumerate(zip(self.constraints, x)):
            if ri is not None and xi != ri:
                new_rule.constraints[i] = None
        if all([value is None for value in new_rule.constraints]):
            return None
        return new_rule

    def covers(self, x):
        for xi, ri in zip(x, self.constraints):
            if ri is not None and xi != ri:
                return False
        return True

    def covers_any(self, data, idxs):
        for idx in idxs:
            if self.covers(data[idx]):
                return True
        return False


class RuleByExample(Rule):
    '''
    It is useful to instance a rule starting from a given input data point.
    '''

    def __init__(self, example: np.array):
        x = example.flatten()
        constraints = []

        for i, feature in enumerate(x):
            # we create discrete constraints for strings (for sure)
            # TODO using ints for discrete constraints is very quick and dirty, but we may regret it down the line
            # if isinstance(feature, (str, np.int32, np.int64)):
            # if feature == 1:
            constraints.append(DiscreteConstraint(value=feature, index=i))
            # we create ordinal constraints for floats (for sure)
            # elif isinstance(feature, (float, np.float32, np.float64)):
            #     constraints.append(OrdinalConstraint(value_min=feature, value_max=feature, index=i))
            # else:
            #     raise NotImplementedError(f'found {type(feature)} for {feature}')

        super().__init__(constraints)


class Constraint:
    '''
    A constraint always has an index, that is used to pick up the i-th feature which is constrained.
    '''

    def __init__(self, index=None):
        self.index = index

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return hash(self.index)

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

    def satisfied(self, x):
        return x[self.index] == self.value

    def generalize(self, x):
        if self.value == x[self.index]:
            return self
        return None

    def __repr__(self):
        return f'X[{self.index}] == {self.value}'


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


class BplClassifierOptimization(ClassifierMixin, BaseEstimator):
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

    def fit(self, X, y, target_class=None, pool_size=1, **kwargs):
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
        X, y = check_X_y(X, y, dtype=None)  # [str, np.int32, np.int64, float, np.float32, np.float64])

        # old hack
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit_transform(X).toarray()
        # self.enc_ = enc

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        if target_class is None:
            target_class = max(self.classes_)

        if len(self.classes_) == 1:
            self.target_class_ = target_class
            self.D_, self.B_ = BplClassifierOptimization.find_rs(X, y, target_class, **kwargs)

        if len(self.classes_) == 2:
            self.target_class_ = target_class
            self.other_class_ = (set(self.classes_) - {self.target_class_}).pop() if len(self.classes_) > 1 else None

            if self.T == 1:
                self.D_, self.B_ = BplClassifierOptimization.find_rs(X, y, target_class, **kwargs)
            else:
                outputs = BplClassifierOptimization.find_rs_with_multiple_runs(X, y, target_class, T=self.T,
                                                                               pool_size=pool_size)
                self.Ds_ = [D for D, B in outputs]

        else:
            self.ovr_ = OneVsRestClassifier(BplClassifierOptimization(tol=self.tol, T=self.T)).fit(X, y)

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

        # old hack
        # X = self.enc_.transform(X).toarray()
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
            rules_bo = BplClassifierOptimization.callable_rules_bo(self.Ds_)
            values = [sum([ht(x) for ht in rules_bo]) for x in X]
            return np.array([self.target_class_ if np.sign(v) > 0 else self.other_class_ for v in values])

        if len(self.classes_) == 2 and self.strategy == 'bp':
            h_bp = BplClassifierOptimization.callable_rules_bp(self.Ds_)
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
    def find_rs_equiv(X, y, target_class, tol=0, optimization=None, **kwargs):
        n_clusters = kwargs.get('n_clusters', 5)
        print(n_clusters)

        # split dataset
        train_p = (X[y == target_class].copy())
        train_n = (X[y != target_class].copy())

        B, C, D, k = [], [], [], 0

        # we get the clusters of the negative examples

        # for feature in range(len(train_n[0])):
        #    C.append({value: set((train_n[:, feature] == value).nonzero()[0])
        #              for value in np.unique(train_n[:, feature])})

        positives_to_check = set(range(len(train_p)))

        while len(positives_to_check) > 0:
            i = positives_to_check.pop()
            p = train_p[i]

            positive_assigned = False
            for i, (bin, rule) in enumerate(zip(B, D)):
                new_rule = rule.generalize(p)
                if new_rule == 'not changed':
                    B[i].append(p)
                    positive_assigned = True
                    break

                elif new_rule is not None and not new_rule.covers_any(train_n, list(range(len(train_n)))):
                    # the rule is ok
                    B[i].append(p)
                    D[i] = new_rule
                    positive_assigned = True
                    break

            if not positive_assigned:
                # create a new bin
                D.append(RuleByExample(p))
                B.append([p])

        D, B = BplClassifierOptimization._prune(D, B)
        return D, B

    @staticmethod
    def find_rs(X, y, target_class, tol=0, optimization=None, **kwargs):
        n_clusters = kwargs.get('n_clusters', 5)
        k_means = kwargs.get('k_means', False)
        print(n_clusters)

        def get_subset(train_n, indexes):
            return train_n[indexes]

        def get_smallest_cluster(new_r, C, notC):
            first_constraint = next(iter(new_r.constraints))
            smallest_cluster = C[first_constraint.index] if first_constraint.value == 1 else notC[
                first_constraint.index]

            for constraint in new_r.constraints:
                cluster_i = C[constraint.index] if constraint.value == 1 else notC[constraint.index]
                if len(cluster_i) < len(smallest_cluster):
                    smallest_cluster = cluster_i
            return list(smallest_cluster)

        train_p = (X[y == target_class].copy())
        train_n = (X[y != target_class].copy())

        # UB(p) # update per un positivo sulla regola / # pos

        # numeri positiv che hanno implicato un cambio di regole
        # diviso il numero di positivi visti
        # funzionava meglio sulla versione precedente!
        # bound su errore falsi negativi.

        if k_means:
            print('start clustering')
            K_MEANS_CLUSTERS = 500  # int(len(train_p) * 0.001
            k_means = MiniBatchKMeans(n_clusters=K_MEANS_CLUSTERS,
                                      random_state=0,
                                      batch_size=1000,
                                      max_iter=10).fit(train_p)

            positive_clusters = k_means.predict(train_p)

            # x1 = [0,0,1]
            # x2 = [0,1,0]

            new_train_p = []
            for cluster_index in range(K_MEANS_CLUSTERS):
                # pick examples that belong to the current cluster
                cluster = train_p[positive_clusters == cluster_index]
                new_train_p.extend(cluster)

            train_p = np.array(new_train_p)

            # train_p = train_p.take(positive_clusters, axis=0)
            print('end clustering')

        # we get the clusters of the negative examples
        C = []

        # C1, v=A
        # C1, v=B
        # C2C

        for feature in range(len(train_n[0])):
            column = [row[feature] for row in train_n]
            if n_clusters == 1:
                C.append({value: list((train_n[:, feature] == value).nonzero()[0])
                          for value in np.unique(train_n[:, feature])})
            elif n_clusters > 1:
                C.append({value: set((train_n[:, feature] == value).nonzero()[0])
                          for value in np.unique(train_n[:, feature])})

            # C.append({value: set([x for x in column if x == value])
            #           for value in np.unique(train_n[:, feature])})

        D, B, k = [], [], 0

        positives_to_check = set(range(len(train_p)))
        negatives_to_check = list(range(len(train_n)))

        rules_lengths = []
        while len(positives_to_check) > 0:
            print("\n", len(positives_to_check), "\n\n")
            intersect_cluster_lengths = []

            # print(len(positives_to_check))
            # we pick a positive example
            i = positives_to_check.pop()
            p = train_p[i]
            r = RuleByExample(p)  # altra implement. ListRule(p)
            # r = ListRule(p)
            B.append([train_p[i]])
            pos_copy = positives_to_check.copy()
            rule_lengths = []

            MAX_PATIENCE = 1000 # min(len(positives_to_check) // 5, 1000)
            patience = MAX_PATIENCE
            for iter_count, other_i in enumerate(pos_copy):
                if patience == 0:
                    for idx in positives_to_check.copy():
                        if r.covers(train_p[idx]):
                            positives_to_check.remove(idx)
                            B[-1].append(train_p[idx])

                    break
                rule_lengths.append(len(r.constraints))

                if r.covers(train_p[other_i]):
                    # rule satisfies the example
                    positives_to_check.remove(other_i)
                    B[-1].append(train_p[other_i])
                    continue

                # attempt to generalize the rule
                new_r = r.generalize(train_p[other_i])
                # print("\r" + str(r), end="")

                if new_r is not None:
                    # print("\r {0:3.2f}% current rule: {1}".format(iter_count / len(pos_copy) * 100, str(r)), end="")

                    # smallest_cluster_idxs = get_smallest_cluster(new_r, C, notC)
                    #
                    # if not new_r.covers_any(train_n, smallest_cluster_idxs):
                    #     r = new_r
                    #     positives_to_check.remove(other_i)
                    #     B[-1].append(train_p[other_i])

                    # build intersection set of clusters

                    # clusters = [C[constraint.index] if constraint.value == 1 else notC[constraint.index] for constraint in new_r.constraints]

                    # C1->5 notC1->4
                    # C2->3 notC2->6
                    # TODO ordinare C1..Cn, notC1..Cn
                    # x2=0 and x1=1


                    # caso base: no negativi coperti (all'inizio quando riempi un bin vuoto)
                    # hai una lista di controesempi. all'inizio ?? vuota, applichi i cluster.
                    # non appena un cluster risulta coperto, peschi un controesempio ed entra
                    # nella lista dei controesempi.

                    # alle iterazioni successive: la lista sar?? sempre pi?? popolata
                    # prima controesempi
                    # poi clusters.

                    # se il boost non c'??, ciao ciao clusters.
                    # reinizializza i controesempi per ogni bin.

                    if n_clusters == 1:
                        clusters = [C[constr.index].get(constr.value, set()) for constr in new_r.constraints]
                        min_cluster = min(clusters, key=len)
                        if not new_r.covers_any(train_n, min_cluster):
                            patience = MAX_PATIENCE
                            r = new_r
                            positives_to_check.remove(other_i)
                            B[-1].append(train_p[other_i])
                        else:
                            patience -= 1

                    elif n_clusters > 1:
                        # altra implementazione
                        clusters = [C[constr.index].get(constr.value, set()) for constr in new_r.constraints]
                        # min_cluster = min(clusters, key=len)
                        clusters = sorted(clusters, key=lambda x: len(x))[:n_clusters]
                        negative_examples_intersection_idxs = set.intersection(*clusters)
                        intersect_cluster_lengths.append(len(negative_examples_intersection_idxs))
                        # intersect_cluster_lengths.append(len(min_cluster))

                        # print("\r {0:.2f} {1:.2f}%".format(np.average(intersect_cluster_lengths[-50:]),
                        #                                   iter_count / len(pos_copy) * 100), end='   ')

                        if not new_r.covers_any(train_n, negative_examples_intersection_idxs):  # min_cluster):
                            patience = MAX_PATIENCE
                            r = new_r
                            positives_to_check.remove(other_i)
                            B[-1].append(train_p[other_i])
                        else:
                            patience -= 1

                    else:
                        assert n_clusters == 0
                        # use the old method
                        negative_examples_covered = new_r.covers_any(train_n, negatives_to_check, optimization=True)

                        if not negative_examples_covered:
                            patience = MAX_PATIENCE
                            r = new_r
                            positives_to_check.remove(other_i)
                            B[-1].append(train_p[other_i])
                        else:
                            patience -= 1
                    # if negative_examples_intersection:  # or len(negative_examples_union) < N:
                    #     # we are covering a negative example, keep old rule
                    #     pass
                    # else:
                    #     # we are not covering a negative example, keep generalized rule
                    #     r = new_r
                    #     positives_to_check.remove(other_i)
                    #     B[-1].append(train_p[other_i])
                    # print(time.time() - start)

            # assert not r.covers_any(train_n)
            D.append(r)
            rules_lengths.append(rule_lengths)
            #
            # import matplotlib.pyplot as plt
            # for i, rl in enumerate(rules_lengths):
            #     plt.plot(rl, label=f'B{i}, len(B{i})={len(B[i])}, len(D{i})={len(D[i].constraints)}')
            #
            # plt.ylim(ymin=0)
            # plt.ylabel('number of constraints')
            # plt.xlabel('number of positive examples checked')
            # plt.legend()
            # plt.show()

        D, B = BplClassifierOptimization._prune(D, B)
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
    def _find_rs_iteration(X, y, target_class, t, tol=0, **kwargs):
        np.random.seed(t)

        random_indexes = np.random.RandomState(seed=t).permutation(len(X))
        X_perm = X[random_indexes].copy()
        y_perm = y[random_indexes].copy()

        Dt, Bt = BplClassifierOptimization.find_rs(X_perm, y_perm, target_class, tol=tol, **kwargs)

        return Dt, Bt

    @staticmethod
    def find_rs_with_multiple_runs(X, y, target_class, tol=0, pool_size=1, T=1, **kwargs):
        if pool_size > 1:
            # TODO why doesn't it work?
            with Pool(pool_size) as p:
                outputs = p.map(
                    partial(BplClassifierOptimization._find_rs_iteration, X, y, target_class, tol=tol, **kwargs),
                    range(T))
        else:
            outputs = [BplClassifierOptimization._find_rs_iteration(X, y, target_class, t, tol=tol, **kwargs) for t in
                       range(T)]
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
