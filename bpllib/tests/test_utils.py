import numpy as np
import pandas as pd
import pytest

from bpllib.utils import get_indexes_of_good_datapoints


def test_get_indexes_of_good_datapoints():
    X = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 1]])
    y = np.array([False, True, True, True, True, False, False])
    y = pd.Series(y)
    X = pd.DataFrame(X)

    good_indexes = get_indexes_of_good_datapoints(X, y, tiebreaker_value=True)
    assert len(good_indexes) == 4
    assert (y[good_indexes] == [True, True, True, False]).all()
    assert (X.iloc[good_indexes].values == X.iloc[[1, 2, 3, 6]].values).all()
