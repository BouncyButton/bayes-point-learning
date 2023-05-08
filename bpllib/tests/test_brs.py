import numpy as np
import pytest

from bpllib import get_dataset
from bpllib._brs_bp import BayesianRuleSetClassifier
from bpllib.tests.utils import run_training, data_custom

from bpllib.tests.utils import data
import time


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_1(data, T, pool_size, strategy):
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size, 'verbose': True, 'encoding': 'ohe', 'maxlen': 3,
              'num_iterations': 100}
    run_training(BayesianRuleSetClassifier, kwargs, data, min_f1_score=0.98)


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_2(data, T, pool_size, strategy):
    d = data_custom(test_datasets=['MUSH'])
    kwargs = {'T': T, 'strategy': strategy, 'pool_size': pool_size, 'verbose': True, 'encoding': 'ohe', 'maxlen': 3,
              'num_iterations': 100}
    run_training(BayesianRuleSetClassifier, kwargs, d, min_f1_score=0.98)
