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


def get_indexes_of_good_datapoints(X, y, tiebreaker_value=True):
    # get indexes
    X = pd.DataFrame(X)
    X.index = list(range(len(X)))
    y = pd.Series(y)
    y.index = list(range(len(X)))
    # change column name
    y.name = 'class'

    df = pd.concat([X, y], axis=1)
    df = df.set_axis(list(X.columns) + ['class'], axis=1)

    # Aggregate the DataFrame to get the most frequent values for each group
    df_agg = df.groupby(list(X.columns)).agg(lambda x: most_frequent_tiebreaker(x, tiebreaker_value)).reset_index()

    return find_indexes(df, df_agg)


def find_indexes(df1, df2):
    df1_dict = {}
    for i, row in df1.iterrows():
        key = tuple(row)
        if key not in df1_dict:
            df1_dict[key] = i

    indexes = []
    for i, row in df2.iterrows():
        key = tuple(row)
        if key in df1_dict:
            indexes.append(df1_dict[key])

    return indexes


def most_frequent_tiebreaker(x, tiebreaker_value):
    value_counts = x.value_counts()
    if len(value_counts) == 1:
        return value_counts.index[0]
    elif value_counts[0] > value_counts[1]:
        return value_counts.index[0]
    elif value_counts[0] == value_counts[1]:
        return tiebreaker_value
    else:
        raise ValueError('This should not happen')


easy_datasets = list(sorted(['MONKS1', 'MONKS2', 'MONKS3', 'TTT', 'CAR', 'COMPAS', 'BREAST', 'SOYBEAN', 'HIV',
                             'LYMPHOGRAPHY', 'PRIMARY', 'VOTE', 'SPECT', 'AUDIO']))
medium_datasets = list(sorted(['KR-VS-KP', 'MUSH']))
hard_datasets = list(sorted(['MARKET', 'ADULT', 'CONNECT-4']))


def get_av_idx_and_val(enc, ohe_idx):
    i = 0
    av_idx = 0
    for av_idx, cat in enumerate(enc.categories_):
        for val in cat:
            if i == ohe_idx:
                return av_idx, val
            i += 1
    raise ValueError('ohe idx too large')
