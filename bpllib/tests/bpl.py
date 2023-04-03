import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from bpllib import get_dataset, FindRsClassifier
from bpllib._bp import best_k_rules
from bpllib._bpl import Rule, DiscreteConstraint

test_datasets = ['CAR']


# @pytest.fixture
def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def test_bpl_estimator(data):
    assert Rule({1: DiscreteConstraint(index=1, value='a')}) == Rule({1: DiscreteConstraint(index=1, value='a')})

    for name, (X, y) in data:
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit_transform(X).toarray().astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est = FindRsClassifier(T=1)
        est.fit(X_train, y_train)
        f1_base = f1_score(y_test, est.predict(X_test))
        print('base: (T=1)', f1_base)
        est = FindRsClassifier(T=20)
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
    test_bpl_estimator(data())
