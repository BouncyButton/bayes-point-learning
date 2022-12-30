import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from bpllib import get_dataset, AqClassifier

test_datasets = ['TTT']


# @pytest.fixture
def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def aq_estimator(data):
    est = AqClassifier(maxstar=1)

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
    y_pred = est.predict(X_train)
    print(f1_score(y_train, y_pred))

    for name, (X, y) in data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
        est = AqClassifier(maxstar=5)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f1)

if __name__ == '__main__':
    aq_estimator(data())
