from functools import partial
from multiprocessing import Pool

import numpy as np


def callable_rules_bo(Ds):
    return [lambda x: 1 if any([rule.covers(x) for rule in D]) else -1 for D in Ds]


def callable_rules_bp(Ds):
    # NO ! return [lambda x: sum([rule.covers(x) for rule in D]) for D in Ds]
    return lambda x: sum([rule.covers(x) for D in Ds for rule in D])


def algorithm_with_multiple_runs(alg, X, y, target_class, T=1, pool_size=1, **kwargs):
    if pool_size > 1:
        with Pool(pool_size) as p:
            outputs = p.map(partial(alg, X[y == target_class], X[y != target_class], **kwargs), range(T))
    else:
        outputs = [alg(X[y == target_class], X[y != target_class], seed=t, **kwargs) for t in range(T)]

    return outputs


def alpha_representation(rulesets):
    from collections import Counter
    cnt = Counter()
    for ruleset in rulesets:
        curr_counts = Counter(ruleset)
        cnt.update(curr_counts)
    return cnt


def best_k_rules(rulesets, k=20):
    cnt = alpha_representation(rulesets)
    return cnt.most_common(k)


def predict_one_with_bo(rulesets, x, target_class, other_class):
    rules_bo = callable_rules_bo(rulesets)
    value = sum([ht(x) for ht in rules_bo])
    return target_class if np.sign(value) > 0 else other_class


def predict_one_with_bp(rulesets, x, target_class, other_class, T):
    h_bp = callable_rules_bp(rulesets)
    value = h_bp(x)
    return target_class if value > T / 2 else other_class


def predict_one_with_best_k(x, n_rules, suggested_k, counter, most_freq_rules, target_class, other_class, T):
    if n_rules is None:
        n_rules = suggested_k
    vote = 0

    alpha_rules = sum(counter.values())
    alpha_k_rules = sum([alpha for r, alpha in counter.most_common(n_rules)])
    gamma_k = alpha_k_rules / alpha_rules

    if most_freq_rules is None:
        most_freq_rules = counter.most_common(n_rules)

    for rule, alpha in most_freq_rules:
        vote += rule.covers(x) * alpha
    return target_class if vote > gamma_k * T / 2 else other_class
