import numpy as np
import pytest
from sklearn.metrics import f1_score

from bpllib._id3_bp import Id3Classifier
from bpllib.tests.utils import run_training, data_custom

from .utils import data
from ..utils import remove_inconsistent_data


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_multi(T, pool_size, strategy):
    d = data_custom(['TTT'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(Id3Classifier, kwargs, d, min_f1_score=0.8)


@pytest.mark.parametrize("T", [1])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_1(data, T, pool_size, strategy):
    d = data_custom(['TTT'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(Id3Classifier, kwargs, d, min_f1_score=0.8)


def test_estimator_replicability():
    # create random categorical data
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(100, 10))
    y = np.random.randint(0, 2, size=(100,))

    X, y = remove_inconsistent_data(X, y)

    clf = Id3Classifier(random_state=42, T=3)
    clf.fit(X, y)
    clf2 = Id3Classifier(random_state=42, T=3)
    clf2.fit(X, y)

    assert clf.rule_sets_ == clf2.rule_sets_
    assert clf.counter_ == clf2.counter_
    assert clf.score(X, y) == clf2.score(X, y)


def test_bootstrap():
    # create random categorical data
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(100, 10))
    y = np.random.randint(0, 2, size=(100,))

    X = X.astype(str)

    X, y = remove_inconsistent_data(X, y)

    clf = Id3Classifier(random_state=42, T=100, pool_size='auto')
    clf.fit(X, y)

    assert len(clf.rule_sets_) == 100

    assert f1_score(y, clf.predict(X)) > 0.5

    # the ruleset should not be all equal
    assert any([c != clf.T for c in clf.counter_.values()])

    assert (clf.predict(X) != clf.predict(X, strategy='single')).any()
