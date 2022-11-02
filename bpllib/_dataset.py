import zlib

import numpy as np

from bpllib import ROOT_DIR

DATASET_FOLDER = 'dataset'


def _read_and_create(url, names, sep=',', drop_names=None, url_test=None):
    import os
    import pandas as pd

    filename = url.split('/')[-1]
    filepath = os.path.join(ROOT_DIR, DATASET_FOLDER, filename)

    if not os.path.exists(filepath):
        import wget
        wget.download(url, out=os.path.join(ROOT_DIR, DATASET_FOLDER))

    if url.split('.')[-1] == 'Z':
        # os.system(f'uncompress {filename}')
        if not os.path.exists(filepath[:-2]):
            import unlzw3
            from pathlib import Path

            uncompressed_data = unlzw3.unlzw(Path(filepath))
            f = open(filepath[:-2], 'wb')
            f.write(uncompressed_data)
            f.close()

        filepath = filepath[:-2]

    df = pd.read_csv(filepath, names=names, header=0, sep=sep)
    if url_test:
        filename = url_test.split('/')[-1]
        filepath = os.path.join(ROOT_DIR, DATASET_FOLDER, filename)
        if not os.path.exists(filepath):
            import wget
            wget.download(url_test, out=os.path.join(ROOT_DIR, DATASET_FOLDER))

        df_test = pd.read_csv(filepath, names=names, header=0, sep=sep)
        df = pd.concat([df, df_test])
    if drop_names:
        return df.drop(drop_names, axis=1)
    return df.replace('?', 'unk')


def _split_X_y(df, positive_label=None, target_feature='class'):
    '''

    Parameters
    ----------
    df pd.Dataframe data to transform
    positive_label to transform a multiclass problem to a single class problem
    target_feature str name of the column used as y

    Returns X, y
    -------

    '''
    if positive_label is None:
        return df.drop(target_feature, axis=1), df[target_feature]
    # binary classification
    return df.drop(target_feature, axis=1), df[target_feature] == positive_label


def _get_data(url, names, positive_label, target_feature='class', sep=',', drop_names=None, url_test=None):
    dataset = _read_and_create(url, names, sep=sep, drop_names=drop_names, url_test=url_test)
    X, y = _split_X_y(dataset, positive_label, target_feature=target_feature)

    return X, y


def get_dataset(dataset_name='TTT'):
    # TTT
    if dataset_name == 'TTT':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data',
            names=[letter + number for letter in 'a' for number in '123456789'] + ['class'],
            positive_label='positive')

    elif dataset_name == 'VOTE':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
            names=['class', 'handicapped-infants', 'water-project-cost-sharing',
                   'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                   'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                   'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
                   'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
                   'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
            positive_label='republican')

    elif dataset_name == 'KR-VS-KP':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data',
            names=['bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq',
                   'bkxcr', 'bkxwp', 'blxwp', 'bxqsq', 'cntxt', 'dsopp',
                   'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8',
                   'reskd', 'reskr', 'rimmx', 'rkxwp', 'rxmsq', 'simpl',
                   'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk',
                   'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg'] + ['class'], sep=',',
            positive_label='won')

    elif dataset_name == 'CONNECT-4':
        X, y = _get_data(
            url='http://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z',
            names=[letter + number for letter in 'abcdefg' for number in '123456'] + ['class'],
            positive_label='win')

    elif dataset_name == 'CAR':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        positive_label = 'unacc'
        X, y = _get_data(url, names, positive_label)

    # MONKS1
    elif dataset_name == 'MONKS1':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train',
            names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id'], sep=' ',
            positive_label=1, drop_names=['Id'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test')

    # note: monks2 seems to perform better using tol=0, differently as noted in the paper.
    elif dataset_name == 'MONKS2':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train',
            names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id'], sep=' ',
            positive_label=1, drop_names=['Id'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test')

    elif dataset_name == 'MONKS3':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train',
            names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id'], sep=' ',
            positive_label=1, drop_names=['Id'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test')

    elif dataset_name == 'MUSH':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
            names=['class'] + ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
                               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                               'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                               'stalk-surface-below-ring', 'stalk-color-above-ring',
                               'stalk-color-below-ring', 'veil-type', 'veil-color',
                               'ring-number', 'ring-type', 'spore-print-color',
                               'population', 'habitat'],
            positive_label='e')

    elif dataset_name == 'WINE':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
            names=['class',
                   'Alcohol',
                   'malic acid',
                   'ash',
                   'alcalinity of ash',
                   'magnesium',
                   'total phenols',
                   'flavanoids',
                   'nonflavanoid phenols',
                   'proanthocyanins',
                   'color intesntiy',
                   'hue',
                   'od280/od315',
                   'proline'], positive_label=2, sep=','
        )

    elif dataset_name == 'TOY':
        X = np.array([[1.11, 1.12], [2.21, 2.22], [1.13, 2.23], [2.24, 1.14],
                      [3.31, 0.01], [3.32, 3.33], [4.41, -1.11], [0.02, -1.12]])
        y = np.array([1] * 4 + [0] * 4)


    elif dataset_name == 'TOY2':
        X = np.array([['a', 'b'], ['b', 'b'], ['c', 'b'], ['d', 'b']], dtype=object)
        y = np.array([['a', 'c'], ['b', 'c'], ['c', 'c'], ['d', 'c']], dtype=object)

    elif dataset_name == 'MOONS':
        from sklearn import datasets
        X, y = datasets.make_moons(n_samples=1000, noise=0.3)

    elif dataset_name == 'TOY3':

        X = np.array([['a', 1.0, 1.23, 30.0, 123.4], ['a', 2.0, 1.34, 30.0, 234.5],
                      ['a', 3.0, 1.45, 20.0, 345.6], ['a', 3.0, 1.56, 20.0, 456.7],
                      ['a', 3.0, 1.78, 20.0, 567.6], ['a', 3.0, 1.89, 20.0, 678.7]],
                     [['b', 3.0, 2.23, 10.0, 123.4], ['b', 2.0, 2.34, 10.0, 234.5],
                      ['b', 1.0, 2.45, 20.0, 345.6], ['b', 1.0, 2.54, 20.0, 456.7],
                      ['b', 2.0, 2.67, 10.0, 567.6], ['b', 2.0, 2.76, 20.0, 678.7]], dtype=object)
        y = np.array([1] * 6 + [0] * 6)

    else:
        raise NotImplementedError(f'The dataset {dataset_name} was not found.')

    return X, y
