import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from bpllib import get_dataset, AqClassifier

test_datasets = ['CAR']


# @pytest.fixture
def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def aq_estimator(data):
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
    print('bo', f1_score(y_train, y_pred_bo))
    print('bp', f1_score(y_train, y_pred_bp))
    input()
    for name, (X, y) in data:
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit_transform(X).toarray().astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est = AqClassifier(maxstar=2, T=3)
        est.fit(X_train, y_train)
        print(est.best_k_rules())
        y_pred_bo = est.predict(X_test, strategy='bo')
        y_pred_bp = est.predict(X_test, strategy='bp')
        f1_bo = f1_score(y_test, y_pred_bo)
        f1_bp = f1_score(y_test, y_pred_bp)
        print('bo', f1_bo, 'bp', f1_bp)

if __name__ == '__main__':
    aq_estimator(data())
