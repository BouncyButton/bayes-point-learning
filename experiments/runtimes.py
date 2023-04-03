import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import csv
import sys

from tqdm import tqdm

from bpllib import get_dataset, FindRsClassifier, AqClassifier
from bpllib._id3 import ID3Classifier
from bpllib._ripper import RIPPERClassifier

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

N_SEEDS = 10
T = 1
methods = [FindRsClassifier]
print(methods)
strategies = ['baseline']  # , 'multi-iteration']
dataset_sizes = [0.5]  #, 0.3333, 0.25, 0.1]  # , 0.3333]  # , 0.1]  # , 0.3333, 0.1]  # [0.5, 0.3]  # , 0.1, 0.03]
POOL_SIZE = 1


def data():
    datasets = [(name, get_dataset(name)) for name in test_datasets]

    # sort by descending size
    datasets.sort(key=lambda x: x[1][0].shape[0], reverse=True)

    return [(name, get_dataset(name)) for name in test_datasets]


def remove_inconsistent_data(X, y):
    # remove from X and y all rows that are labeled inconsistently
    # i.e. if there are two rows with the same features but different labels -> remove both

    # get indexes
    X = pd.DataFrame(X)
    y = pd.Series(y)
    indexes = list(X.index)

    inconsistent_indexes = set()
    for i, j in itertools.combinations(indexes, 2):
        row_i = X.loc[i]
        row_j = X.loc[j]
        if (row_i == row_j).all() and y[i] != y[j]:
            inconsistent_indexes.add(i)
            inconsistent_indexes.add(j)

    X = X.drop(list(inconsistent_indexes))
    y = y.drop(list(inconsistent_indexes))
    return X, y


def template_estimator(data, strategy='bp'):
    # take strptime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filename = 'results_' + now + '.csv'

    results = csv.writer(open(filename, 'w'))
    results.writerow(
        ['dataset', 'method', 'dataset_size', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp', 'strategy', 'T', 'pool_size',
         'best_k', 'time_elapsed', 'seed', 'avg_ruleset_len', 'avg_rule_len', 'optimization'])

    import time

    pbar = tqdm(total=len(data) * N_SEEDS * len(methods) * len(strategies) * len(dataset_sizes) * 2)

    for name, (X, y) in data:
        print(name)
        # remove inconsistent data
        if name == 'BREAST' or name == 'PRIMARY':
            prev_len = len(X)
            X, y = remove_inconsistent_data(X, y)
            if prev_len != len(X):
                print('removed', prev_len - len(X), 'inconsistent data points')

        for method in methods:  # add AqClassifier
            for dataset_size in dataset_sizes:
                for optimization in [True, False]:
                # TODO ensure to pass seed to each submethod to ensure reproducibility
                    # TODO find-rs uses as seed the t value in that loop,
                    #  Q: if we run it multiple times, is it ok?
                    use_next_seed = 0

                    for seed in range(N_SEEDS):
                        pbar.update(1)
                        start = time.time()

                        ok = False
                        while not ok:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - dataset_size,
                                                                                random_state=seed + use_next_seed)
                            # check if y has at least 2 classes
                            if len(np.unique(y_train)) > 1:
                                ok = True
                            else:
                                use_next_seed += 1

                        if method == AqClassifier:
                            est = method(maxstar=3)
                        else:
                            est = method()

                        if method == RIPPERClassifier:
                            est.fit(X_train, y_train, starting_seed=seed)  # , pool_size=POOL_SIZE)

                        else:
                            est.fit(X_train, y_train, optimization=optimization)

                        y_pred = est.predict(np.array(X_test))
                        f1 = f1_score(np.array(y_test), y_pred)
                        # assert f1 > 0
                        acc = accuracy_score(y_test, y_pred)
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        # print(name, f1)

                        if method == RIPPERClassifier:
                            avg_ruleset_len = len(est.inner_clf_.ruleset_)
                            avg_rule_len = np.array([len(rule.conds) for rule in est.inner_clf_.ruleset_]).mean()
                        else:
                            avg_ruleset_len = len(est.rules_)  # TODO change ruleset_ to rules_ in code
                            avg_rule_len = np.array([len(rule) for rule in est.rules_]).mean()
                        results.writerow(
                            [name, method.__name__, dataset_size, f1, acc, tn, fp, fn, tp, strategy, T, 1, -1,
                             time.time() - start, seed + use_next_seed, avg_ruleset_len, avg_rule_len, optimization])


if __name__ == '__main__':
    template_estimator(data())

# TOOO fai confronti sistematici facendo T / |D|.


# idea: bpl può gestire nativamente gli unk
