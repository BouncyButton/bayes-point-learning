import pytest
from bpllib._corels_bp import CorelsClassifier
from bpllib.tests.utils import run_training, data_custom

from bpllib.tests.utils import data


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_multi(T, pool_size, strategy):
    d = data_custom(['TTT'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(CorelsClassifier, kwargs, d, min_f1_score=0.8)


@pytest.mark.parametrize("T", [1])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_1(data, T, pool_size, strategy):
    d = data_custom(['TTT'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(CorelsClassifier, kwargs, d, min_f1_score=0.8)
