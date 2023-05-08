import math

from tqdm import tqdm

from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._rule import Rule
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# random rules
r1 = Rule(constraints={0: DiscreteConstraint(value=0, index=0),
                       1: DiscreteConstraint(value=1, index=1),
                       3: DiscreteConstraint(value=0, index=3)})

r2 = Rule(constraints={2: DiscreteConstraint(value=1, index=2),
                       4: DiscreteConstraint(value=1, index=4)})

# r3 = Rule(constraints={5: DiscreteConstraint(value=1, index=5),
#                       6: DiscreteConstraint(value=0, index=6)})

rule_set = [r1, r2]

X_train = np.random.randint(0, 2, size=(100, 20))
y_train = np.array([any([r.covers(x) for r in rule_set]) for x in X_train])
X_test = np.random.randint(0, 2, size=(100, 20))
y_test = np.array([any([r.covers(x) for r in rule_set]) for x in X_test])

y_train.sum()

import itertools


def generate_rules(m, n_features=7):
    all_features = list(range(n_features))
    combos = itertools.combinations(all_features, m)
    rules = set()
    for combo in combos:
        for combo_values in itertools.product([0, 1], repeat=m):
            rule = Rule(constraints={})
            for i in range(m):
                if combo_values[i] == 0:
                    rule.constraints[combo[i]] = DiscreteConstraint(value=0, index=combo[i])
                else:
                    rule.constraints[combo[i]] = DiscreteConstraint(value=1, index=combo[i])
            rules.add(rule)
    return rules


rules_space = set.union(*[generate_rules(m=m, n_features=5) for m in range(1, 5)])
print(len(rules_space))


# Generate the set of all possible tuples of up to k elements each, chosen from rules_space
def generate_tuples(k):
    all_tuples = []
    for i in range(1, k + 1):
        tuples = tqdm(itertools.combinations(rules_space, i), total=math.comb(len(rules_space), i))

        l = []
        for t in tuples:
            l.append(t)

        all_tuples = all_tuples + l

        print(i, len(all_tuples))
    return all_tuples


x = generate_tuples(4)  # one minute...
print(x)
