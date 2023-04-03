import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import csv
import sys

from sklearn.utils import check_array, check_X_y
from tqdm import tqdm

from bpllib import get_dataset, FindRsClassifier, AqClassifier
from bpllib._id3 import ID3Classifier
from bpllib._ripper import RIPPERClassifier

test_datasets = [
    'CAR',
    'TTT',
    #'MUSH',
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

N_SEEDS = 1
T = 100
methods = [
    FindRsClassifier,
    # RIPPERClassifier,
    # ID3Classifier,
    # AqClassifier
]
dataset_sizes = [0.5]  # , 0.33, 0.25]

POOL_SIZE = 1


def data():
    datasets = [(name, get_dataset(name)) for name in test_datasets]

    # sort by descending size
    datasets.sort(key=lambda x: x[1][0].shape[0] * x[1][0].shape[1], reverse=True)

    return [(name, get_dataset(name)) for name in test_datasets]


def remove_inconsistent_data(X, y):
    # remove from X and y all rows that are labeled inconsistently
    # i.e. if there are two rows with the same features but different labels -> remove both

    # get indexes
    X = pd.DataFrame(X)  # .reset_index()
    # monks-3 gives error (indexes are nan)
    X.index = list(range(len(X)))

    y = pd.Series(y)  # .reset_index()
    y.index = list(range(len(X)))
    indexes = list(X.index)

    inconsistent_indexes = set()
    print('removing inconsistent data')
    for i, j in tqdm(list(itertools.combinations(indexes, 2))):
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
    filename = 'results_' + now + '.pkl'

    results = pd.DataFrame(
        columns=['dataset', 'method', 'dataset_size', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp', 'strategy', 'T',
                 'time_elapsed', 'seed', 'model'])

    # results = csv.writer(open(filename, 'w'))
    # results.writerow(
    #    ['dataset', 'method', 'dataset_size', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp', 'strategy', 'T', 'pool_size',
    #     'best_k', 'time_elapsed'])

    import time

    pbar = tqdm(total=len(data) * N_SEEDS * len(methods) * len(dataset_sizes))

    for name, (X, y) in data:
        enc = OneHotEncoder(handle_unknown='ignore')
        X = enc.fit_transform(X).toarray().astype(int)

        print(name)
        # remove inconsistent data
        if name == 'BREAST' or name == 'PRIMARY' or name == 'MONKS3':
            prev_len = len(X)
            X, y = remove_inconsistent_data(X, y)
            if prev_len != len(X):
                print('removed', prev_len - len(X), 'inconsistent data points')

        for method in methods:  # add AqClassifier
            for dataset_size in dataset_sizes:
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
                        X_train, y_train = check_X_y(X_train, y_train, dtype=[str])
                        X_test, y_test = check_X_y(X_test, y_test, dtype=[str])

                        # check if y has at least 2 classes
                        if len(np.unique(y_train)) > 1:
                            ok = True
                        else:
                            use_next_seed += 1

                    if method == AqClassifier:
                        est = method(T=T, maxstar=1)
                    else:
                        est = method(T=T)

                    if method == RIPPERClassifier:
                        est.fit(X_train,
                                y_train, starting_seed=seed + use_next_seed, find_best_k=True)  # , pool_size=POOL_SIZE)
                    else:
                        est.fit(X_train, y_train, find_best_k=True, optimization=False)
                    for strategy in ['bp', 'bo', 'best-k']:
                        y_pred = est.predict(X_test, strategy=strategy)

                        f1 = f1_score(y_test, y_pred)

                        print(strategy, f1)
                        # assert f1 > 0
                        if f1 <= 0:
                            print('f1 <= 0')

                        acc = accuracy_score(y_test, y_pred)
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        # print(name, f1)

                        for rule, alpha in est.counter_.most_common(100):
                            assert np.array([rule.covers(x) for x in X_train[y_train == 1]]).mean() > 0

                        results = results.append(
                            {'dataset': name, 'method': method.__name__, 'dataset_size': dataset_size, 'f1': f1,
                             'accuracy': acc, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'strategy': strategy, 'T': T,
                             'time_elapsed': time.time() - start, 'seed': seed + use_next_seed, 'model': est},
                            ignore_index=True)
                    # write to pickle
                results.to_pickle(filename)


if __name__ == '__main__':
    template_estimator(data())

# TOOO fai confronti sistematici facendo T / |D|.


# idea: bpl può gestire nativamente gli unk
