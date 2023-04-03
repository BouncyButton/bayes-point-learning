import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from bpllib import get_dataset, FindRsClassifier
from bpllib._bp import best_k_rules
from bpllib._bpl import Rule, DiscreteConstraint
from bpllib._id3 import ID3Classifier

test_datasets = ['TTT',
                 # 'CAR',
                 # 'MUSH',
                 # 'KR-VS-KP',
                 # 'HIV',
                 # 'BREAST',
                 # 'VOTE'
                 # 'PRIMARY'
                 ]


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


@pytest.fixture
def data():
    datasets = [(name, get_dataset(name)) for name in test_datasets]
    # sort by ascending size
    datasets.sort(key=lambda x: x[1][0].shape[0])
    return datasets


def test_id3_estimator(data):
    assert Rule({1: DiscreteConstraint(index=1, value='a')}) == Rule({1: DiscreteConstraint(index=1, value='a')})

    for name, (X, y) in data:
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit_transform(X).toarray().astype(int)
        print(name, len(X))

        # create data split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        # remove inconsistent data
        if name == 'BREAST':
            prev_len = len(X_train)
            X_train, y_train = remove_inconsistent_data(X_train, y_train)
            if prev_len != len(X_train):
                print('removed', prev_len - len(X_train), 'inconsistent training data points')

        est = ID3Classifier(T=1)
        est.fit(X_train, y_train)
        f1_base = f1_score(y_test, est.predict(X_test))
        print('base: (T=1)', f1_base)
        est = ID3Classifier(T=100)
        est.fit(X_train, y_train)

        y_pred_bo = est.predict(X_test, strategy='bo')
        f1_bo = f1_score(y_test, y_pred_bo)
        print('bo', f1_bo)
        y_pred_bp = est.predict(X_test, strategy='bp')
        f1_bp = f1_score(y_test, y_pred_bp)
        print('bp', f1_bp)
        for n_rules in [8, 16, 24]:
            y_pred_best_k = est.predict(X_test, strategy='best-k', n_rules=n_rules)
            print(est.counter_.most_common(n_rules))
            # print('best k found:', est.suggested_k_)
            f1_best_k = f1_score(y_test, y_pred_best_k)
            print('k=', n_rules, f1_best_k)


if __name__ == '__main__':
    test_id3_estimator(data())
