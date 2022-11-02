import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from bpllib import get_dataset, BplClassifier

test_datasets = [#'CAR',
                 'TTT',
                 # 'CONNECT-4',
                 # 'MUSH', 'MONKS1', 'MONKS2', 'MONKS3', 'KR-VS-KP', 'VOTE'
                 ]


@pytest.fixture
def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def test_template_estimator(data):
    est = BplClassifier()

    for name, (X, y) in data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        print(name, f1_score(y_test, y_pred))


def test_template_estimator_bo(data):
    for name, (X, y) in data:
        est = BplClassifier(T=3, strategy='bo')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train, pool_size=1)
        y_pred = est.predict(X_test)
        print(name, f1_score(y_test, y_pred))


def test_template_estimator_bp(data):
    for name, (X, y) in data:
        est = BplClassifier(T=3, strategy='bp')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        print(name, f1_score(y_test, y_pred))
