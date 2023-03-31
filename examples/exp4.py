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

for METHOD in ['find-rs', 'ripper', 'id3', 'aq']:
    filename = f'final-dat/multi-iteration-3-{METHOD}.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

    # read from the pickle file
    with open(filename, 'rb') as f:
        df = pickle.load(f)

    # DATASET = 'CAR'

    test_datasets = [
        # 'CAR',
        # 'TTT',
        # 'MUSH',
        # 'MONKS1',
        # 'MONKS2',
        # 'MONKS3',
        # 'KR-VS-KP',
        'VOTE',
        # 'BREAST',
        # 'HIV',
        # 'LYMPHOGRAPHY',
        # 'PRIMARY'
    ]

    for DATASET in test_datasets:

        X, y = get_dataset(DATASET)
        # names = X.columns

        if DATASET in ['BREAST', 'PRIMARY']:
            X, y = remove_inconsistent_data(X, y)

        avg_similarities = []
        perf_gain = []

        for DATASET_SIZE in [0.33]:  # [0.5, 0.33, 0.25]:
            row = df[(df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (df['strategy'] == 'bp')]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            model = row['model']
            seed = row['seed']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                random_state=seed)

            X, y = X_test, y_test

            X, y = check_X_y(X, y, dtype=[str])


            def similarity(rs1, rs2):
                return len(set(rs1).intersection(set(rs2))) / len(set(rs1).union(set(rs2)))


            similarities = []
            for ruleset1, ruleset2 in itertools.combinations(model.rulesets_, 2):
                similarities.append(similarity(ruleset1, ruleset2))

            print(DATASET, DATASET_SIZE)
            print(f'Average similarity: {np.mean(similarities)}')
            # simulate a T=1 run
            baseline_ruleset = model.rulesets_[0]
            y_pred_base = np.array([1 if any(rule.covers(x) for rule in baseline_ruleset) else 0 for x in X])
            baseline_f1 = f1_score(y, y_pred_base)
            print(f'Baseline F1: {baseline_f1}')
            print(f'BO F1: {f1_score(y, model.predict(X, strategy="bo"))}')
            bp_f1 = f1_score(y, model.predict(X, strategy="bp"))
            print(f'BP F1: {bp_f1}')
            print(f'best-k f1: {f1_score(y, model.predict(X, strategy="best-k"))}')
            avg_similarity = np.mean(similarities)
            avg_similarities.append(avg_similarity)
            max_increment = 1 - baseline_f1

            delta = bp_f1 - baseline_f1
            pg = delta / max_increment if max_increment != 0 else 0
            perf_gain.append(pg)
            plt.scatter(avg_similarity, pg, label=(METHOD + str(DATASET_SIZE)))
plt.legend()
plt.xlabel('Average similarity')
plt.ylabel('Performance gain')
plt.show()
