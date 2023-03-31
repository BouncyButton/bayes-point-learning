from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pickle
import copy

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

from bpllib._bp import callable_rules_bo, callable_rules_bp, predict_one_with_best_k, alpha_representation
from bpllib._bpl import AgainstDiscreteConstraint, Rule, DiscreteConstraint, DiscreteOrConstraint


class AqClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, maxstar=1, T=1, verbose=0):
        self.maxstar = maxstar
        self.T = T
        self.verbose = verbose
        self.threshold_acc = 0.99

    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

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

    def fit(self, X, y, target_class=1,find_best_k=False):
        self.target_class_ = target_class
        X, y = check_X_y(X, y, dtype=None)  #[str, int])
        self.classes_ = unique_labels(y)
        if len(self.classes_) == 2:
            self.other_class_ = (set(self.classes_) - {target_class}).pop()
        else:
            raise NotImplementedError('multiclass not available yet')
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        df = pd.DataFrame(X)
        df['class'] = y

        if self.T == 1:
            self.rules_ = AQAlgorithm(X[y == target_class], X[y != target_class], self.maxstar, verbose=self.verbose)
        else:
            self.rulesets_ = self.aq_with_multiple_runs(X, y, target_class, T=self.T, maxstar=self.maxstar)
            self.counter_ = alpha_representation(self.rulesets_)

            if find_best_k:
                bp_acc = (self.predict(X, strategy='bp') == y).mean()
                self.suggested_k_ = None

                for k in range(1, len(self.counter_)):
                    new_acc = (self.predict(X, 'best-k', n_rules=k) == y).mean()
                    if new_acc >= self.threshold_acc * bp_acc:
                        self.suggested_k_ = k
                        break

        return self

    def aq_with_multiple_runs(self, X, y, target_class, T=1, pool_size=1, maxstar=5):
        if pool_size > 1:
            with Pool(pool_size) as p:
                outputs = p.map(partial(AQAlgorithm, X[y == target_class], X[y != target_class], maxstar=maxstar),
                                range(T))
        else:
            outputs = [AQAlgorithm(X[y == target_class], X[y != target_class], maxstar=maxstar, seed=t) for t in
                       tqdm(range(T))]
        # returns T sets of rules
        return outputs

    def predict(self, X, strategy='bo', **kwargs):
        X = check_array(X, dtype=None) #[str, int])
        return np.array([self.predict_one(x, strategy=strategy, **kwargs) for x in X])

    def predict_one(self, x, strategy='bo', **kwargs):
        if self.T == 1:
            for rule in self.rules_:
                if rule.covers(x):
                    return self.target_class_
            return 1 - self.target_class_
        elif strategy == 'bo':
            rules_bo = callable_rules_bo(self.rulesets_)
            value = sum([ht(x) for ht in rules_bo])
            return self.target_class_ if np.sign(value) > 0 else self.other_class_
        elif strategy == 'bp':
            h_bp = callable_rules_bp(self.rulesets_)
            value = h_bp(x)
            return self.target_class_ if value > self.T / 2 else self.other_class_
        elif strategy == 'best-k':
            most_freq_rules = kwargs.get('most_freq_rules')
            n_rules = kwargs.get('n_rules')
            return predict_one_with_best_k(x, n_rules, n_rules, self.counter_, most_freq_rules, self.target_class_,
                                           self.other_class_, self.T)

        else:
            raise NotImplementedError('strategy not implemented')


def AQAlgorithm(P, N, maxstar=5, seed=None, verbose=0):
    if seed:
        np.random.seed(seed)

        random_indexes_P = np.random.RandomState(seed=seed).permutation(len(P))
        random_indexes_N = np.random.RandomState(seed=seed).permutation(len(N))
        P = P[random_indexes_P].copy()
        N = N[random_indexes_N].copy()

    # get unique values for each feature
    # unique_values = [set(u) for u in np.unique(list(np.concatenate((P, N), axis=0)), axis=0).T]
    unique_values = [set(u) for u in np.concatenate((P,N), axis=0).T]


    P1 = P
    R = set()  # rule set
    if verbose > 0:
        pbar = tqdm(total=len(P))

    while len(P1) > 0:
        if verbose > 0:
            pbar.n = len(P) - len(P1)
            pbar.refresh()
        # select random p from P1
        idx = 0  # np.random.randint(0, len(P1))
        p = P1[idx]
        # print("pos", p)

        # find a rule from p
        rule = star(p, P1, N, unique_values, maxstar, verbose=verbose)

        # filter out positive examples that are covered by the rule
        P1 = rule.examples_not_covered(P1)
        R = R.union({rule})

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


def Q(r, P, N, unique_values, eps=1e-8):
    F = len(P[0])
    C = len(r)
    min_len = 1 - C / F
    val_count = r.values_count()
    tot_val_count = sum([len(u) for u in unique_values])
    max_val_count = (1 / F - eps) * val_count / tot_val_count
    # print(r)
    # print(len(r.examples_covered(P)), min_len, max_val_count)

    if r.covers_any(N):
        return min_len + max_val_count

    return len(r.examples_covered(P)) + min_len + max_val_count  # / len(P)  # + 1 - len(x.examples_covered(N)) / len(N)


def star(p, P, N: np.array, unique_values, maxstar=5, mode='TF', minq=0.5, verbose=0):
    PS = set()
    for n in N:
        if verbose > 3:
            print("neg", n)
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
            if verbose > 2:
                print("PS", PS)
                print("newps:", new_PS)
            PS = {r1 * r2 for r1 in PS for r2 in new_PS if r1 * r2}
            if verbose > 2:
                print("result: ", PS)
        # keep only maxstar best complexes in PS, according to Q
        PS = set(sorted(PS, key=lambda r: Q(r, P, N, unique_values), reverse=True)[:maxstar])

        # TODO remove from N all examples covered by PS?
        # AQ20 and Wojtusiak (encyclopedia) show different implementations
        # N = N[np.array([not any([r.covers(x) for r in PS]) for x in N])]

    # it seems we need to return one rule at a time, so we return the best rule according to Q
    if verbose > 1:
        print([(r, "q=" + str(Q(r, P, N, unique_values))) for r in PS])
    # you should find a way to break ties
    # here i pick all the rules with max q
    # max_q = max([Q(r2, P, N) for r2 in PS])
    # rules_with_max_q = [r for r in PS if Q(r, P, N) == max_q]
    # if len(rules_with_max_q) > 1:
    #     shortest_len = min([len(r) for r in rules_with_max_q])
    #     rules_with_min_len = [r for r in rules_with_max_q if len(r) == shortest_len]
    #     if len(rules_with_min_len) > 1:
    #         max_values = max([sum([len(constraint) for constraint in r.constraints]) for r in rules_with_min_len])
    #         rules_with_max_values = [r for r in rules_with_min_len if
    #                                  sum([len(constraint) for constraint in r.constraints]) == max_values]
    #         rule_found = rules_with_max_values[0]
    #     else:
    #         rule_found = rules_with_min_len[0]
    # else:
    #     rule_found = rules_with_max_q[0]

    rule_found = max(PS, key=lambda r: Q(r, P, N, unique_values))
    if verbose > 1:
        print(rule_found, '\tp=', len(rule_found.examples_covered(P)), '\tn=',
              len(rule_found.examples_covered(N)))
    return rule_found

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
