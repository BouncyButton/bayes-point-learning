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

plt.rcParams["font.family"] = "Times New Roman"


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
filename = f'multi-iteration-hiv.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

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
    #'TTT',
    # 'BREAST',
    'HIV',
    # 'LYMPHOGRAPHY',
    # 'PRIMARY'
    #'VOTE'
]

for DATASET in test_datasets:

    X, y = get_dataset(DATASET)
    # names = X.columns

    if DATASET in ['BREAST', 'PRIMARY']:
        X, y = remove_inconsistent_data(X, y)

    DATASET_SIZE = 0.5
    for METHOD in ['FindRsClassifier', 'RIPPERClassifier', 'ID3Classifier']:
        # [0.5, 0.33, 0.25]:
        rows = df[(df['method'] == METHOD) & (df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (
                    df['strategy'] == 'bp')]
        if len(rows) == 0:
            continue

        models = list(rows['model'])
        seeds = list(rows['seed'])

        avg_baseline_f1 = []
        for model, seed in zip(models, seeds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                random_state=seed)

            X_test, y_test = check_X_y(X_test, y_test, dtype=[str])

            # simulate a T=1 run
            baseline_ruleset = model.rulesets_[0]
            y_pred_base = np.array([1 if any(rule.covers(x) for rule in baseline_ruleset) else 0 for x in X_test])
            baseline_f1 = f1_score(y_test, y_pred_base)
            avg_baseline_f1.append(baseline_f1)
        avg_baseline_f1 = np.mean(avg_baseline_f1)
        # print(f'avg baseline f1: {avg_baseline_f1}')
        avg_similarities = []
        perf_gain = []

        for model, seed in tqdm(list(zip(models, seeds))):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                random_state=seed)

            X_test, y_test = check_X_y(X_test, y_test, dtype=[str])


            def similarity(rs1, rs2):
                return len(set(rs1).intersection(set(rs2))) / len(set(rs1).union(set(rs2)))


            similarities = []
            for ruleset1, ruleset2 in itertools.combinations(model.rulesets_, 2):
                similarities.append(similarity(ruleset1, ruleset2))

            # print(DATASET, DATASET_SIZE)
            # print(f'Average similarity: {np.mean(similarities)}')
            # print(f'BO F1: {f1_score(y_test, model.predict(X_test, strategy="bo"))}')
            bp_f1 = f1_score(y_test, model.predict(X_test, strategy="bp"))
            # print(f'BP F1: {bp_f1}')
            # print(f'best-k f1: {f1_score(y_test, model.predict(X_test, strategy="best-k"))}')
            avg_similarity = np.mean(similarities)
            avg_similarities.append(avg_similarity)
            max_increment = 1 - avg_baseline_f1

            delta = bp_f1 - avg_baseline_f1
            pg = delta  # / max_increment if max_increment != 0 else 0
            perf_gain.append(pg)
        plt.scatter(avg_similarities, perf_gain, label=METHOD.replace('Classifier', ''))
plt.legend()
plt.xlabel('Average similarity')
plt.ylabel('$\Delta$ F1')
plt.title(f'Performance gain vs. similarity ({DATASET}, size={DATASET_SIZE})')
plt.savefig(f'performance_gain_vs_similarity_{DATASET}.eps', format='eps')
plt.savefig(f'performance_gain_vs_similarity_{DATASET}.pdf', format='pdf')
plt.savefig(f'performance_gain_vs_similarity_{DATASET}.png', format='png')