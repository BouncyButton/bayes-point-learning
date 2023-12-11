import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from bpllib import get_dataset
from bpllib._find_rs_bp import FindRsClassifier
from bpllib.tests.utils import run_training, data_custom

from .utils import data
import time

from ..utils import remove_inconsistent_data


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['best-k'])
def test_meta(data, T, pool_size, strategy):
    kwargs = {'T': T, 'pool_size': pool_size, 'verbose': 1}
    # run_training(FindRsClassifier, kwargs, data, min_f1_score=0.98)
    min_f1_score = 0.98
    for name, (X, y) in data:
        est = FindRsClassifier(**kwargs)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train)

        est.fit_find_rs_on_rulesets(X_train, y_train)

        y_pred = est.predict(X_test, strategy=strategy)

        f1 = f1_score(y_test, y_pred)

        assert (f1 >= min_f1_score)
    print('ok')
