import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm


def resample(X, y, seed=None):
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, n_samples, replace=True)
    X_resampled = X[indices]
    y_resampled = y[indices]
    return X_resampled, y_resampled


def remove_inconsistent_data(X, y):
    # get indexes
    X = pd.DataFrame(X)
    X.index = list(range(len(X)))
    y = pd.Series(y)
    y.index = list(range(len(X)))
    # change column name
    y.name = 'class'

    df = pd.concat([X, y], axis=1)
    df = df.set_axis(list(X.columns) + ['class'], axis=1)

    df = df.groupby(list(X.columns)).agg(lambda x: x.value_counts().index[0]).reset_index()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y
