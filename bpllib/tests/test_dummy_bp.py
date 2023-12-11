import pytest
from bpllib._dummy_bp import DummyClassifier
from bpllib.tests.utils import run_training

from bpllib.tests.utils import data


@pytest.mark.parametrize("T", [1, 3])
@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator(data, T, pool_size, strategy):
    est = DummyClassifier
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size}
    run_training(est, kwargs, data, min_f1_score=0.0)
