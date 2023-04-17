import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from bpllib import get_dataset
from bpllib._dummy_bp import DummyClassifier


@pytest.fixture
def data():
    return data_custom()


def data_custom(test_datasets=None):
    if test_datasets is None:
        test_datasets = [
            'TTT'
        ]
    return [(name, get_dataset(name)) for name in test_datasets]


def run_training(estimator_class, kwargs, data, min_f1_score=0.0):
    for name, (X, y) in data:
        est = estimator_class(**kwargs)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)

        f1 = f1_score(y_test, y_pred)

        assert f1 >= min_f1_score, 'got ' + str(f1) + ' expected >= ' + str(min_f1_score)
