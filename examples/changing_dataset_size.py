from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import csv
import sys

from tqdm import tqdm

from bpllib import get_dataset, BplClassifier, AqClassifier
from bpllib._ripper import RIPPERClassifier

test_datasets = [
    # 'CAR',
    # 'TTT',
    # 'CONNECT-4',
    # 'MUSH',
    # 'MONKS1',
    # 'MONKS2',
    # 'MONKS3',
    # 'KR-VS-KP',
    # 'VOTE'
    'BREAST',
    'HIV'
]

N_SEEDS = 1
T = 20
methods = [RIPPERClassifier, BplClassifier]
strategies = ['baseline', 'multi-iteration']
dataset_sizes = [0.5, 0.3]  # , 0.1, 0.03]

def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def template_estimator(data, strategy='bp'):
    # take strptime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filename = 'results_' + now + '.csv'

    results = csv.writer(open(filename, 'w'))
    results.writerow(['dataset', 'method', 'dataset_size', 'f1', 'strategy', 'T', 'pool_size', 'best_k', 'time_elapsed'])

    import time

    pbar = tqdm(total=len(data) * N_SEEDS * len(methods) * len(strategies) * len(dataset_sizes))

    for name, (X, y) in data:
        for method in methods:  # add AqClassifier
            for strategy in strategies:
                for dataset_size in dataset_sizes:
                    # TODO ensure to pass seed to each submethod to ensure reproducibility
                    # TODO find-rs uses as seed the t value in that loop,
                    #  Q: if we run it multiple times, is it ok?
                    for seed in range(N_SEEDS):
                        pbar.update(1)
                        start = time.time()

                        ok = False
                        use_next_seed = 0
                        while not ok:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - dataset_size,
                                                                            random_state=use_next_seed)
                            # check if y has at least 2 classes
                            if len(np.unique(y_train)) > 1:
                                ok = True
                            else:
                                use_next_seed += 1
                        if strategy == 'baseline':
                            est = method(T=1)
                        else:
                            est = method(T=T)

                        est.fit(X_train, y_train, starting_seed=seed)

                        if strategy == 'baseline':

                            y_pred = est.predict(X_test)
                            f1 = f1_score(y_test, y_pred)
                            # print(name, f1)

                            results.writerow([name, str(method), dataset_size, f1, strategy, T, 1, -1, time.time() - start])
                        else:
                            for bayes_strategy in ['bo', 'bp']: #, 'best-k']:
                                y_pred = est.predict(X_test, strategy=bayes_strategy)
                                f1 = f1_score(y_test, y_pred)
                                # print(name, f1)
                                if bayes_strategy == 'best-k':
                                    best_k = est.suggested_k_
                                else:
                                    best_k = -1
                                results.writerow(
                                    [name, str(method), dataset_size, f1, bayes_strategy, T, 1, best_k, time.time() - start])


if __name__ == '__main__':
    template_estimator(data())

# TOOO fai confronti sistematici facendo T / |D|.


# idea bpl può gestire nativamente gli unk