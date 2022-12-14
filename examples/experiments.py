from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import csv

from bpllib import get_dataset, BplClassifierOptimization, BplClassifier, BplClassifierSplit

test_datasets = [#'CAR',
                 #'TTT',
                 # 'CONNECT-4',
                 'MUSH',
                 # 'MONKS1',
                #'MONKS2',
                 # 'MONKS3',
                 # 'KR-VS-KP',
                 # 'VOTE'
                 ]


def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def template_estimator(data, strategy='bp'):
    # take strptime
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filename = 'results_' + now + '.csv'

    results = csv.writer(open(filename, 'w'))
    results.writerow(['dataset', 'f1', 'strategy', 'T', 'clusters', 'pool_size', 'tol', 'time_elapsed'])
    for name, (X, y) in data:
        import time
        orig_start = time.time()
        for n_clusters in [1,2,3,4,5,10,15]: #+ [15, 20, 30, 50, 100, 100000]:
            f1s = []
            times = []
            for i in range(3):
                enc = OneHotEncoder(handle_unknown='ignore')
                X = enc.fit_transform(X).toarray().astype(int)
                start = time.time()

                if name == 'MONKS2':
                    # enc = OneHotEncoder(handle_unknown='ignore')
                    # X = enc.fit_transform(X).toarray().astype(int)
                    tol = 1
                else:
                    tol = 0
                est = BplClassifierOptimization(T=1, strategy=None, tol=tol)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
                # enc = OneHotEncoder(handle_unknown='ignore')
                # X_train = enc.fit_transform(X_train).toarray().astype(int)
                # X_test = enc.transform(X_test).toarray().astype(int)
                est.fit(X_train, y_train, pool_size=1, n_clusters=n_clusters)
                y_pred = est.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                print(name, f1)
                # print("tn,fp,fn,tp:")
                # print(confusion_matrix(y_test, y_pred))
                f1s.append(f1)
                times.append(time.time() - start)
            results.writerow([name, np.mean(f1s), '--', 1, n_clusters, 1, tol, np.mean(times)])

            print("time elapsed:", time.time() - orig_start)
            print(np.mean(f1s), "+-", np.std(f1s))


def optimization():
    est = BplClassifierOptimization(T=1, strategy=None, tol=0)
    X_train = np.array([['a', 'Asmall', 0], ['a', 'Bmedium', 1], ['b', 'Asmall', 1], ['c', 'Clarge', 1],  # pos
                        ['a', 'Clarge', 1], ['a', 'Clarge', 0], ['b', 'Bmedium', 1], ['b', 'Bmedium', 1],  # neg
                        ['c', 'Clarge', 0]])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    est.fit(X_train, y_train, pool_size=1)
    y_train_pred = est.predict(X_train)
    print(confusion_matrix(y_train, y_train_pred))


if __name__ == '__main__':
    # optimization()
    # print('==== BO ====')
    template_estimator(data())  # , 'bo')
    # print('==== BP ====')
    # template_estimator(data(), 'bp')


# TOOO fai confronti sistematici facendo T / |D|.