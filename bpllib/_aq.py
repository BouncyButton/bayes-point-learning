import numpy as np
import pandas as pd
import pickle
import copy

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

from bpllib._bpl import AgainstDiscreteConstraint, Rule, DiscreteConstraint, DiscreteOrConstraint


class AqClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, maxstar=5):
        self.maxstar = maxstar

    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

    def fit(self, X, y, target_class=1):
        self.target_class_ = target_class
        X, y = check_X_y(X, y, dtype=[str, int])
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        df = pd.DataFrame(X)
        df['class'] = y
        self.rules = AQAlgorithm(X[y == target_class], X[y != target_class], self.maxstar)

        return self

    def predict(self, X):
        X = check_array(X, dtype=None)
        return np.array([self.predict_one(x) for x in X])

    def predict_one(self, x):
        for rule in self.rules:
            if rule.covers(x):
                return self.target_class_
        return 1 - self.target_class_


def AQAlgorithm(P, N, maxstar=5):
    # get unique values for each feature
    unique_values = [set(u) for u in np.unique(np.concatenate((P, N), axis=0), axis=0).T]

    P1 = P
    R = set()  # rule set
    while len(P1) > 0:
        print(len(P1))
        # select random p from P1
        idx = 0  # np.random.randint(0, len(P1))
        p = P1[idx]

        # find a rule from p
        rule_set = star(p, P, N, unique_values, maxstar)
        for rule in rule_set:
            # filter out positive examples that are covered by the rule
            P1 = rule.examples_not_covered(P1)
        R = R.union(rule_set)

    return R


def LEF(r, P, N, maxstar=1, new_example_threshold=10):
    return filter(r, lambda x: x.covers_all(P) and not x.covers_any(N))[:maxstar]

    # return sorted(r, key=lambda x: len(x.covered_examples(P))/len(P) + (1-len(x.covered_examples(N)/len(N))), reverse=True)[:maxstar]

    # chosen_rules = []
    #
    # # TODO 1. sort the rules in the star according to LEF, from the best to the worst
    # rule_list = sorted(r, key=lambda x: x.quality_index, reverse=True)
    # # 2a. select the first rule and compute the number of examples it covers.
    # examples_covered = rule_list[0].covered_examples(P)
    # n_examples_covered = len(examples_covered)
    # # 4. continue the process until all rules are inspected.
    # for rule in rule_list[1:]:
    #     # 2b. Select the next rule and compute the number of new examples it covers.
    #     next_examples_covered = rule.covers(P)
    #     new_examples_covered = next_examples_covered - examples_covered
    #     n_new_examples_covered = len(new_examples_covered)
    #     # 3a. If the number of new examples covered exceeds a new example threshold, then the rule is selected
    #     if n_new_examples_covered > new_example_threshold:
    #         chosen_rules.append(rule)
    #     # 3b. otherwise it is discarded.
    #
    # # The result of this procedure is a set of rules
    # # selected from a star. The list of positive events to
    # # cover is updated by deleting all those events that are
    # # covered by these rules.
    # return chosen_rules


def Q(x, P, N):
    return len(x.examples_covered(P)) / len(P) + ((1 - len(x.examples_covered(N))) / len(N))


def star(p, P, N: np.array, unique_values, maxstar=5, mode='TF', minq=0.5):
    PS = set()
    for n in N:
        # select a negative example n from N
        # extend the rule against n
        new_PS = extension_against(p, n, unique_values)
        if not PS:
            # TODO it seems we can filter out the rules that are not good enough here already?
            #  (see AQ20 example)
            # new_PS = set(sorted(new_PS, key=lambda r: Q(r, P, N), reverse=True)[:maxstar])
            PS = new_PS
        else:
            # perform logical multiplication of PS and new_PS
            PS = {r1 * r2 for r1 in PS for r2 in new_PS if r1 * r2}
        # keep only maxstar best complexes in PS, according to Q
        PS = set(sorted(PS, key=lambda r: Q(r, P, N), reverse=True)[:maxstar])

        # TODO remove from N all examples covered by PS?
        # AQ20 and Wojtusiak (encyclopedia) show different implementations
        # N = N[np.array([not any([r.covers(x) for r in PS]) for x in N])]

    return PS

    # r = set()  # empty rule
    # for n in N:
    #     r1 = extension_against(p, n, unique_values)
    #     r2 = r1.union(r)
    #     r2 = LEF(r2, P, N, maxstar)
    #     if mode == 'PD':
    #         if q(r2, P, N) - q(r, P, N) > minq:
    #             r = r2
    #     else:
    #         r = r2
    # return r


def q(r, Pdata, Ndata, w=0.5):
    n = len(r.examples_covered(Ndata))
    p = len(r.examples_covered(Pdata))
    P = len(Pdata)
    N = len(Ndata)
    return (p / P) ** w * max(((P + N) / N) * (p / (p + n) - P / (P + N)), 0) ** (1 - w)


def extension_against(p, n, unique_values):
    r = set()  # = Rule(constraints=set())

    for index, (feature_p, feature_n) in enumerate(zip(p, n)):
        if feature_p == feature_n:
            # new rule is impossible
            continue
        else:
            # create a new constraint for this feature
            valid_values = unique_values[index] - {feature_n}
            if len(valid_values) == 0:
                continue

            if len(valid_values) == 1:
                # using a 'discrete' constraint
                r.add(Rule(constraints={index: DiscreteConstraint(valid_values.pop(), index)}))
            else:
                # using an 'or' constraint
                r.add(Rule(constraints={index: DiscreteOrConstraint(valid_values, index)}))

    # the result of the extension-against operation is a
    # disjunction of single condition rules, namely one rule
    # for each nonidentical attribute.
    return r
