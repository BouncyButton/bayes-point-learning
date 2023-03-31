import itertools
import pickle
# get the latest result file
import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from tqdm import tqdm

from bpllib import get_dataset


def remove_inconsistent_data(X, y):
    # remove from X and y all rows that are labeled inconsistently
    # i.e. if there are two rows with the same features but different labels -> remove both

    # get indexes
    X = pd.DataFrame(X)
    X.index = list(range(len(X)))
    y = pd.Series(y)
    y.index = list(range(len(X)))

    indexes = list(X.index)

    inconsistent_indexes = set()
    for i, j in list(itertools.combinations(indexes, 2)):
        row_i = X.loc[i]
        row_j = X.loc[j]
        if (row_i == row_j).all() and y[i] != y[j]:
            inconsistent_indexes.add(i)
            inconsistent_indexes.add(j)

    X = X.drop(list(inconsistent_indexes))
    y = y.drop(list(inconsistent_indexes))
    return X, y


TARGET_METRIC = 'f1'
DATASET_SIZE = 0.33
print("""
\\begin{table}[ht]
\\centering
\\small

\\begin{tabular}{ l | l l l l  }
\\hline
dataset & find-rs & ripper & id3 \\\\ 
 \\hline
""")

DATASET = 'CAR'

test_datasets = [
    'CAR',
    'TTT',
    'MUSH',
    'MONKS1',
    'MONKS2',
    'MONKS3',
    'KR-VS-KP',
    'VOTE',
    'BREAST',
    'HIV',
    'LYMPHOGRAPHY',
    'PRIMARY'
]
test_datasets.sort()

methods = ['find-rs', 'ripper', 'id3']  # , 'aq']

for DATASET in test_datasets:

    X, y = get_dataset(DATASET)
    # names = X.columns

    if DATASET in ['BREAST', 'PRIMARY', 'MONKS3']:
        X, y = remove_inconsistent_data(X, y)

    avg_similarities = []
    perf_gain = []

    for DATASET_SIZE in [DATASET_SIZE]:  # [0.5, 0.33, 0.25]:
        print("\\texttt{%s}" % DATASET if DATASET != 'LYMPHOGRAPHY' else '\\texttt{LYMPH}', end=" & ")

        values = []
        for METHOD in methods:  # , 'aq']:
            filename = f'final-dat/multi-iteration-3-{METHOD}.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

            # read from the pickle file
            with open(filename, 'rb') as f:
                df = pickle.load(f)

            row = df[(df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (df['strategy'] == 'bp')
                # & (df['method'] == 'ID3Classifier')]
                     ]
            if len(row) == 0:
                values.append(None)
                continue
            row = row.iloc[0]
            model = row['model']
            seed = row['seed']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                random_state=seed)

            X, y = X_test, y_test

            X, y = check_X_y(X, y, dtype=[str])

            # simulate a T=1 run
            baseline_ruleset = model.rulesets_[0]
            y_pred_base = np.array([1 if any(rule.covers(x) for rule in baseline_ruleset) else 0 for x in X])
            baseline_f1 = f1_score(y, y_pred_base) if TARGET_METRIC == 'f1' else accuracy_score(y, y_pred_base)
            y_pred_bp = model.predict(X, strategy="bp")
            bp_f1 = f1_score(y, y_pred_bp) if TARGET_METRIC == 'f1' else accuracy_score(y, y_pred_bp)

            # print(f'BP F1: {bp_f1}')
            # print(f'best-k f1: {f1_score(y, model.predict(X, strategy="best-k"))}')

            max_increment = 1 - baseline_f1
            delta = bp_f1 - baseline_f1
            values.append((bp_f1, delta))

        for i, x in enumerate(values):
            if x is None:
                print('-', end=" & " if i < len(methods) - 1 else ' \\\\\n')
            else:
                bp_f1, delta = x
                # check if current bp_f1 is the best
                best_bp_f1 = max([x[0] for x in values if x is not None])
                if bp_f1 == best_bp_f1:
                    print(f'\\underline{{\\textbf{{{bp_f1:.3f}}}}} ', end='')
                else:
                    print(f'{bp_f1:.3f} ', end='')
                if delta > 0:
                    print(f'\\tiny{{\\textbf{{+{delta:.3f}}}}}', end=' & ' if i < len(methods) - 1 else ' \\\\\n')
                else:
                    print(f'\\tiny{{{delta:.3f}}}', end=' & ' if i < len(methods) - 1 else ' \\\\\n')

print("""
\\hline
\\end{tabular}
\\caption{Performance (%s) of the proposed BP method on the test datasets. 
Small values show the delta in %s compared to the baseline. """ % (TARGET_METRIC, TARGET_METRIC))
print(f"""We used {DATASET_SIZE * 100}\\%""")
print(""" of the examples in the dataset to train each model.
({\\textbf{bold}} values indicate a performance gain over the baseline.)
(\\underline{\\textbf{underlined}} values indicate the best-performing BP method.)}
\\end{table}
""")
