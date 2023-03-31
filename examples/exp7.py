import itertools
import pickle
# get the latest result file
import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from tqdm import tqdm

from bpllib import get_dataset
from bpllib._bpl import DiscreteConstraint, Rule


def remove_inconsistent_data(X, y):
    # remove from X and y all rows that are labeled inconsistently
    # i.e. if there are two rows with the same features but different labels -> remove both

    # get indexes
    X = pd.DataFrame(X)
    y = pd.Series(y)
    indexes = list(X.index)

    inconsistent_indexes = set()
    for i, j in tqdm(list(itertools.combinations(indexes, 2))):
        row_i = X.loc[i]
        row_j = X.loc[j]
        if (row_i == row_j).all() and y[i] != y[j]:
            inconsistent_indexes.add(i)
            inconsistent_indexes.add(j)

    X = X.drop(list(inconsistent_indexes))
    y = y.drop(list(inconsistent_indexes))
    return X, y


TARGET_METRIC = 'f1'

# for METHOD in ['find-rs', 'ripper', 'id3', 'aq']:

# DATASET = 'CAR'

test_datasets = [
    'TTT',
    'MONKS1',
    'MONKS2'
]

ground_truths = [
    [
        Rule({0: DiscreteConstraint('x', 0), 1: DiscreteConstraint('x', 1), 2: DiscreteConstraint('x', 2)}),
        Rule({3: DiscreteConstraint('x', 3), 4: DiscreteConstraint('x', 4), 5: DiscreteConstraint('x', 5)}),
        Rule({6: DiscreteConstraint('x', 6), 7: DiscreteConstraint('x', 7), 8: DiscreteConstraint('x', 8)}),
        Rule({0: DiscreteConstraint('x', 0), 3: DiscreteConstraint('x', 3), 6: DiscreteConstraint('x', 6)}),
        Rule({1: DiscreteConstraint('x', 1), 4: DiscreteConstraint('x', 4), 7: DiscreteConstraint('x', 7)}),
        Rule({2: DiscreteConstraint('x', 2), 5: DiscreteConstraint('x', 5), 8: DiscreteConstraint('x', 8)}),
        Rule({0: DiscreteConstraint('x', 0), 4: DiscreteConstraint('x', 4), 8: DiscreteConstraint('x', 8)}),
        Rule({2: DiscreteConstraint('x', 2), 4: DiscreteConstraint('x', 4), 6: DiscreteConstraint('x', 6)})
    ],
    [
        Rule({0: DiscreteConstraint('1', 0), 1: DiscreteConstraint('1', 1)}),
        Rule({0: DiscreteConstraint('2', 0), 1: DiscreteConstraint('2', 1)}),
        Rule({0: DiscreteConstraint('3', 0), 1: DiscreteConstraint('3', 1)}),
        Rule({4: DiscreteConstraint('1', 4)}),
    ],
    [
        Rule({0: DiscreteConstraint('1', 0), 1: DiscreteConstraint('1', 1)}),
        Rule({0: DiscreteConstraint('1', 0), 2: DiscreteConstraint('1', 2)}),
        Rule({0: DiscreteConstraint('1', 0), 3: DiscreteConstraint('1', 3)}),
        Rule({0: DiscreteConstraint('1', 0), 4: DiscreteConstraint('1', 4)}),
        Rule({0: DiscreteConstraint('1', 0), 5: DiscreteConstraint('1', 5)}),
        Rule({1: DiscreteConstraint('1', 1), 2: DiscreteConstraint('1', 2)}),
        Rule({1: DiscreteConstraint('1', 1), 3: DiscreteConstraint('1', 3)}),
        Rule({1: DiscreteConstraint('1', 1), 4: DiscreteConstraint('1', 4)}),
        Rule({1: DiscreteConstraint('1', 1), 5: DiscreteConstraint('1', 5)}),
        Rule({2: DiscreteConstraint('1', 2), 3: DiscreteConstraint('1', 3)}),
        Rule({2: DiscreteConstraint('1', 2), 4: DiscreteConstraint('1', 4)}),
        Rule({2: DiscreteConstraint('1', 2), 5: DiscreteConstraint('1', 5)}),
        Rule({3: DiscreteConstraint('1', 3), 4: DiscreteConstraint('1', 4)}),
        Rule({3: DiscreteConstraint('1', 3), 5: DiscreteConstraint('1', 5)}),
        Rule({4: DiscreteConstraint('1', 4), 5: DiscreteConstraint('1', 5)})
    ]
]


print("""
\\begin{table}[ht]
\\centering
\\small

\\begin{tabular}{ l l | c c c c  }
\\hline
dataset & size & find-rs & ripper & id3 & aq \\\\ 
 \\hline
""")
for DATASET, ground_truth in zip(test_datasets, ground_truths):

    X, y = get_dataset(DATASET)
    # names = X.columns

    if DATASET in ['BREAST', 'PRIMARY']:
        X, y = remove_inconsistent_data(X, y)

    for DATASET_SIZE in [0.25, 0.33, 0.5]:
        print(f'\n{DATASET} & {DATASET_SIZE} & \t', end='')
        for method_name, METHOD in zip(['find-rs', 'ripper', 'id3', 'aq'],
                                       ['FindRsClassifier', 'RIPPERClassifier', 'ID3Classifier', 'AqClassifier']):
            filename = f'multi-iteration-3-{method_name}.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

            # read from the pickle file
            with open(filename, 'rb') as f:
                df = pickle.load(f)

            # [0.5, 0.33, 0.25]:
            rows = df[(df['method'] == METHOD) & (df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (
                    df['strategy'] == 'bp')]
            if len(rows) == 0:
                print('-- \t', end=' & \t' if method_name != 'aq' else ' \\\\\n')
                continue
            row = rows.iloc[0]

            model = row['model']
            seed = row['seed']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                random_state=seed)

            X_test, y_test = check_X_y(X_test, y_test, dtype=[str])


            def similarity(rs1, rs2):
                return len(set(rs1).intersection(set(rs2))) / len(set(rs1).union(set(rs2)))


            similarities = [similarity(ground_truth, ruleset) for ruleset in model.rulesets_]
            print(f'{np.mean(similarities):.4f}', end=' & \t' if method_name != 'aq' else ' \\\\\n')
print("""
\\hline
\\end{tabular}
\\caption{Average similarity of the rulesets found by the BP methods to the ground truth rulesets.}
\\label{tab:bp-similarity}
\\end{table}
""")

