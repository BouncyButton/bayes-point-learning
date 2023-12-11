import gzip
import json
import math
import pickle
import os
from itertools import product

import numpy as np
import pandas as pd
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

MAX_RULES = 250

ENCODING = 'av'
METRIC = 'precision'  # precision
skip_existing = False

METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']  # 'AQ', 'BRS',

df_results = pd.DataFrame(columns=['dataset', 'method', 'encoding', 'avg_metric', 'std_metric'])

for ENCODING, METRIC in product(['av', 'ohe'], ['f1']):
    for dataset in hard_datasets:
        # check if figure exists
        if os.path.isfile(f'images/best-k/bestk_{dataset}_{ENCODING}_{METRIC}.png') and skip_existing:
            print(f'figure for {dataset} {ENCODING} {METRIC} exists, skipping...')
            continue
        avg_metric = {m: [] for m in METHODS}
        std_metric = {m: [] for m in METHODS}

        max_model_len = 0
        encoding = ENCODING

        for method in METHODS:
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

                max_model_len = max(max_model_len, len(model.counter_))

                metric_seed = []
                for n_rules in tqdm(range(1, MAX_RULES + 1)):
                    y_pred = model.predict(X_test, n_rules=min(n_rules, len(model.counter_)), strategy='best-k')
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
            metric = np.zeros((len(metrics), MAX_RULES))
            # fill with values
            for i, metric_seed in enumerate(metrics):
                metric[i, :len(metric_seed)] = metric_seed
            # continue with the last value
            for i in range(len(metrics)):
                metric[i, len(metrics[i]):] = metrics[i][-1]

            avg_metric[method] = np.mean(metric, axis=0)
            std_metric[method] = np.std(metric, axis=0)

        # normalized auc
        auc = {m: np.trapz(avg_metric[m]) / np.trapz(avg_metric['Find-RS']) for m in METHODS}
        decaying_auc = {m: np.trapz(avg_metric[m] / np.arange(1, len(avg_metric[m]) + 1)) / np.trapz(
            avg_metric['Find-RS'] / np.arange(1, len(avg_metric['Find-RS']) + 1)) for m in METHODS}

        all_avg = {m: np.mean(avg_metric[m]) for m in METHODS}
        all_std = {m: np.std(avg_metric[m]) for m in METHODS}

        plt.figure(figsize=(5, 5))

        for method in METHODS:
            plt.plot(range(1, len(avg_metric[method]) + 1), avg_metric[method],
                     label=f'{method}')  # (AUC={auc[method]:.2f}, dAUC={decaying_auc[method]:.2f})')
            plt.fill_between(range(1, len(avg_metric[method]) + 1), avg_metric[method] - std_metric[method],
                             avg_metric[method] + std_metric[method], alpha=0.2)
            # this is a single row dataframe appended.
            # if you have [[1,2,3]] -> the list is saved in a single row. :D
            df_results = pd.concat([df_results,
                                    pd.DataFrame({'dataset': dataset, 'method': method, 'encoding': ENCODING,
                                                  'avg_metric': all_avg[method],
                                                  'std_metric': all_std[method],
                                                  'y': [json.dumps(list(avg_metric[method]))],
                                                  'y_std': [json.dumps(list(std_metric[method]))]})],
                                   ignore_index=True)

        # pick ymin as the 0.01 quantile
        ymin = np.quantile([a for m in METHODS for a in avg_metric[m] if not math.isnan(a)], 0.01)
        # pick ymax as the max
        ymax = np.max([a + s for m in METHODS for a, s in zip(avg_metric[m], std_metric[m]) if not math.isnan(a + s)])

        plt.ylim(ymin, ymax)
        plt.xlabel('Number of rules')
        plt.ylabel(f'{METRIC}')
        plt.title(dataset + f' enc={ENCODING}')
        plt.legend()
        plt.savefig(f'images/best-k/bestk_{dataset}_{ENCODING}_{METRIC}.pdf')
        plt.savefig(f'images/best-k/bestk_{dataset}_{ENCODING}_{METRIC}.png')

df_results.to_csv('images/best-k/bestk_results-hard.csv', index=False)
