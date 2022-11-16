import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from bpllib import get_dataset, AqClassifier

test_datasets = ['TTT']


#@pytest.fixture
def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def aq_estimator(data):
    est = AqClassifier()

    for name, (X, y) in data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        print(name, f1_score(y_test, y_pred))


if __name__ == '__main__':
    aq_estimator(data())
