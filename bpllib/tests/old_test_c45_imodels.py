import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from bpllib import get_dataset
from bpllib._c45_bp import C45Classifier
from bpllib.tests.utils import run_training, data_custom

from bpllib.tests.utils import data
import time


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_mono(data, T, pool_size, strategy):
    d = data_custom(test_datasets=['TTT'])
    clf = C45Classifier()
    X = np.array(d[0][1][0])
    y = np.array(d[0][1][1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    assert clf.fit(X_train, y_train).score(X_test, y_test) > 0.75
