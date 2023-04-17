import numpy as np
import pytest
from bpllib._aq_bp import AqClassifier
from bpllib.tests.utils import run_training, data_custom

from .utils import data
from ..utils import remove_inconsistent_data


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_multi(T, pool_size, strategy):
    d = data_custom(['MONKS1'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(AqClassifier, kwargs, d, min_f1_score=0.8)


@pytest.mark.parametrize("T", [1])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_1(data, T, pool_size, strategy):
    d = data_custom(['MONKS1'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(AqClassifier, kwargs, d, min_f1_score=0.8)


def test_estimator_replicability():
    # create random categorical data
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(100, 10))
    y = np.random.randint(0, 2, size=(100,))

    X, y = remove_inconsistent_data(X, y)

    clf = AqClassifier(random_state=42, maxstar=1)
    clf.fit(X, y)
    clf2 = AqClassifier(random_state=42, maxstar=1)
    clf2.fit(X, y)

    # assert clf.rule_sets_ == clf2.rule_sets_
    # assert clf.counter_ == clf2.counter_
    assert clf.score(X, y) == clf2.score(X, y)
