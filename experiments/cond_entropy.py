import gzip
import math
import pickle
import os
from itertools import product

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm

from bpllib import utils, get_dataset

# connect to sqlite db
import sqlite3

from bpllib.utils import easy_datasets, medium_datasets, hard_datasets

conn = sqlite3.connect('results.sqlite')
cursor = conn.cursor()

MAX_RULES = 50

ENCODING = 'av'
METRIC = 'precision'  # precision
skip_existing = True

METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']  # 'AQ', 'BRS',
STRATEGIES = ['best-k', 'top-k', 'top-k-alpha', 'best-k-no-weights']
for ENCODING, METRIC in product(['av', 'ohe'], ['f1']):
    for dataset in ['CAR', 'HIV', 'VOTE', 'BREAST']:
        max_model_len = 0
        encoding = ENCODING

        for method in METHODS:
            # check if figure exists
            if os.path.isfile(f'images/bp_{dataset}_{method}_{ENCODING}_{METRIC}.png') and skip_existing:
                print(f'figure for {dataset} {method} {ENCODING} {METRIC} exists, skipping...')
                continue
            avg_metric = {m: [] for m in METHODS}
            std_metric = {m: [] for m in METHODS}

            for strategy in STRATEGIES:
                metrics = []
                for seed in range(10):
                    if method == 'BRS':
                        encoding = 'ohe'

                    X, y = get_dataset(dataset)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                        random_state=seed, shuffle=True)

                    print('retrieving model for', dataset, method, encoding, seed)

                    result = cursor.execute(
                        f"SELECT * FROM results WHERE "
                        f"dataset='{dataset}' AND method='{method}' AND encoding='{encoding}' AND seed={seed} AND strategy='bp'").fetchone()

                    # reverse this gzip.compress(pickle.dumps(row['model']))
                    if result is None:
                        print('[!] row is none! :(')
                        continue

                    if result[6] is None:
                        print('[!] model is none! :(')
                        continue

                    model = gzip.decompress(result[6])
                    model = pickle.loads(model)

                    if model.counter_ is None:
                        print('[!] counter is none! :(')
                        continue

                    model.cond_entropy_ = []
                    model.cond_entropy_alpha_ = []
                    model.cond_entropy_dict_ = dict()
                    model.cond_entropy_alpha_dict_ = dict()
                    model.calculate_cond_entropy(model.validation(X_train), y_train)

                    max_model_len = max(max_model_len, len(model.counter_))

                    metric_seed = []
                    for n_rules in tqdm(range(1, min(MAX_RULES + 1, len(model.counter_) + 1))):
                        y_pred = model.predict(X_test, n_rules=n_rules, strategy=strategy)
                        f1 = f1_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        acc = accuracy_score(y_test, y_pred)
                        if METRIC == 'f1':
                            metric = f1
                        elif METRIC == 'precision':
                            metric = prec
                        elif METRIC == 'recall':
                            metric = recall
                        elif METRIC == 'accuracy':
                            metric = acc
                        else:
                            raise ValueError('metric not supported')
                        metric_seed.append(metric)
                    metrics.append(np.array(metric_seed))

                # make f1 a matrix of shape (n_seeds, max(n_rules))
                # initialize from 0
                metric = np.zeros((len(metrics), min(max_model_len, MAX_RULES)))
                # fill with values
                for i, metric_seed in enumerate(metrics):
                    metric[i, :len(metric_seed)] = metric_seed
                # continue with the last value
                for i in range(len(metrics)):
                    metric[i, len(metrics[i]):] = metrics[i][-1]

                avg_metric[strategy] = np.mean(metric, axis=0)
                std_metric[strategy] = np.std(metric, axis=0)

            plt.figure(figsize=(10, 10))

            labels = {
                'best-k': 'use only alpha',
                'top-k': 'use only cond entropy',
                'top-k-alpha': 'use alpha + cond entropy',
                'best-k-no-weights': 'use only alpha, dont reweight'
            }

            for strategy in STRATEGIES:
                plt.plot(range(1, len(avg_metric[strategy]) + 1), avg_metric[strategy], label=labels[strategy])
                plt.fill_between(range(1, len(avg_metric[strategy]) + 1), avg_metric[strategy] - std_metric[strategy],
                                 avg_metric[strategy] + std_metric[strategy], alpha=0.2)

            # pick ymin as the 0.01 quantile
            ymin = np.quantile([a for m in STRATEGIES for a in avg_metric[m] if not math.isnan(a)], 0.01)
            # pick ymax as the max
            ymax = np.max(
                [a + s for m in STRATEGIES for a, s in zip(avg_metric[m], std_metric[m]) if not math.isnan(a + s)])

            plt.ylim(ymin, ymax)
            plt.xlabel('Number of rules')
            plt.ylabel(f'{METRIC}')
            plt.title(f'{dataset}, method={method}, enc={ENCODING}')
            plt.legend()
            plt.savefig(f'images/strategies/{dataset}_{ENCODING}_{METRIC}_{method}.png')
            plt.show()
            plt.close()
