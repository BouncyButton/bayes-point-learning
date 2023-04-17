import numpy as np
import pytest

from bpllib import get_dataset
from bpllib._find_rs_bp import FindRsClassifier
from bpllib.tests.utils import run_training, data_custom

from bpllib.tests.utils import data
import time

from bpllib.utils import remove_inconsistent_data


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_mono(data, T, pool_size, strategy):
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size, 'verbose': 1}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=0.98)


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [5])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_multi(data, T, pool_size, strategy):
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size, 'verbose': 1}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=0.98)


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1, 3])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_multi(data, T, pool_size, strategy):
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size, 'verbose': 1}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=1.0)


@pytest.mark.parametrize("T", [1])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_1(data, T, pool_size, strategy):
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=0.95)


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [3, 'auto'])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_mush(data, T, pool_size, strategy):
    X, y = get_dataset('MUSH')
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size, 'verbosity': 1}
    run_training(FindRsClassifier, kwargs, [('mush', (X, y))], min_f1_score=1.0)


def test_estimator_replicability():
    # create random categorical data
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(100, 10))
    y = np.random.randint(0, 2, size=(100,))

    X, y = remove_inconsistent_data(X, y)

    clf = FindRsClassifier(random_state=42, T=3)
    clf.fit(X, y)
    clf2 = FindRsClassifier(random_state=42, T=3)
    clf2.fit(X, y)

    assert clf.rule_sets_ == clf2.rule_sets_
    assert clf.counter_ == clf2.counter_
    assert clf.score(X, y) == clf2.score(X, y)


if __name__ == '__main__':
    d = data_custom(test_datasets=['CONNECT-4'])
    import time

    start = time.time()
    test_template_estimator_multi(d, 20, 7, 'bp')
    print(time.time() - start)
