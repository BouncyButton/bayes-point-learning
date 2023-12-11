import math
from collections import Counter

import numpy as np

from bpllib import FindRsClassifier

from itertools import permutations


def generate_permutations(X, y):
    from itertools import permutations

    # Get the number of elements in X and y
    num_elements = X.shape[0]

    # Generate all possible permutations of the row indices
    row_indices = np.arange(num_elements)
    permutations_indices = list(permutations(row_indices))

    # Create all permutations of rows for X and y
    permutations_X = [X[perm, :] for perm in permutations_indices]
    permutations_y = [y[list(perm)] for perm in permutations_indices]
    return permutations_X, permutations_y


X_orig = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])
y_orig = np.array([1, 1, 1, 0])

X_full = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y_full = np.array([1, 1, 1, 1, 0, 1, 0, 1])

total_counter = Counter()
clf = FindRsClassifier(rule_pruning=False)
D, B = clf.base_method(X_orig, y_orig, 1)
total_counter.update(D)
print('find-rs, no gen, no bp')
print('example: ', total_counter)
# print(np.array(
#    [any([rule.covers(x) for rule, freq in total_counter.items()]) == y for x, y in zip(X_full, y_full)]).mean())

# permute all possible permutations in X
total_counter = Counter()
rule_sets = []
for i, (X, y) in enumerate(zip(*generate_permutations(X_orig, y_orig))):
    clf = FindRsClassifier(rule_pruning=False)
    # clf.permute_constraints = False
    D, B = clf.base_method(X, y, 1)
    total_counter.update(D)
    rule_sets.append(D)

single_results = np.array([
    np.array([any([rule.covers(x) for rule in ruleset]) == y for x, y in zip(X_full, y_full)]).mean()
    for ruleset in rule_sets])
print(single_results.mean(), "+-", single_results.std())

print('find-rs, no gen, bp')
print(total_counter)
print(np.array(
    [(sum([freq * rule.covers(x) for rule, freq in total_counter.items()]) >= math.factorial(len(X_orig)) / 2) == y for
     x, y
     in zip(X_full, y_full)]).mean())

total_counter = Counter()
clf = FindRsClassifier(rule_pruning=True, generalization_probability=1, random_state=2)
D, B = clf.base_method(X_orig, y_orig, 1)
total_counter.update(D)
print('find-rs, gen (p=1.0), no bp')
print('example: ', total_counter)
# print(np.array(
#    [any([rule.covers(x) for rule, freq in total_counter.items()]) == y for x, y in zip(X_full, y_full)]).mean())

# permute all possible permutations in X
total_counter = Counter()
rule_sets = []
for i, (X, y) in enumerate(zip(*generate_permutations(X_orig, y_orig))):
    clf = FindRsClassifier(rule_pruning=True, generalization_probability=1, random_state=i)
    # clf.permute_constraints = False
    D, B = clf.base_method(X, y, 1)
    total_counter.update(D)
    rule_sets.append(D)

single_results = np.array([
    np.array([any([rule.covers(x) for rule in ruleset]) == y for x, y in zip(X_full, y_full)]).mean()
    for ruleset in rule_sets])
print(single_results.mean(), "+-", single_results.std())

print('find-rs, gen (p=1.0), bp')
print(total_counter)
print(np.array(
    [(sum([freq * rule.covers(x) for rule, freq in total_counter.items()]) >= math.factorial(len(X_orig)) / 2) == y for
     x, y
     in zip(X_full, y_full)]).mean())
