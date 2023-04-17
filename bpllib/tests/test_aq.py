import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from bpllib import get_dataset
from bpllib._aq import AqClassifier

test_datasets = ['TTT']


@pytest.fixture
def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def test_aq_estimator(data):
    est = AqClassifier(maxstar=5, T=1)

    X_train = np.array([
        [0, 0, 0],
        [2, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [2, 1, 0],
        [0, 1, 1]
    ])
    y_train = np.array([1, 1, 1, 0, 0, 0])
    est.fit(X_train, y_train)
    y_pred_bo = est.predict(X_train, strategy='bo')
    y_pred_bp = est.predict(X_train, strategy='bp')

    assert np.all(f1_score(y_train, y_pred_bo) > 0.4)
    assert np.all(f1_score(y_train, y_pred_bp) > 0.4)

    for name, (X, y) in data:
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit_transform(X).toarray().astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
        est = AqClassifier(maxstar=2, T=1, verbose=1)
        est.fit(X_train, y_train)
        f1_base = f1_score(y_test, est.predict(X_test))
        print('base: (T=1)', f1_base)
        est = AqClassifier(maxstar=1, T=3, verbose=1)
        est.fit(X_train, y_train)
        print('nrules=', len(est.rulesets_))

        y_pred_bo = est.predict(X_test, strategy='bo')
        f1_bo = f1_score(y_test, y_pred_bo)
        print('bo', f1_bo)
        y_pred_bp = est.predict(X_test, strategy='bp')
        f1_bp = f1_score(y_test, y_pred_bp)
        print('bp', f1_bp)
        print('nrules=', len(est.counter_))
        for n_rules in [8, 16, 24]:
            y_pred_best_k = est.predict(X_test, strategy='best-k', n_rules=n_rules)
            print(est.counter_.most_common(n_rules))
            # print('best k found:', est.suggested_k_)
            f1_best_k = f1_score(y_test, y_pred_best_k)
            print('k=', n_rules, f1_best_k)


if __name__ == '__main__':
    test_aq_estimator(data())
