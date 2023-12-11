import zlib

import numpy as np
import pandas
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from bpllib import ROOT_DIR
from bpllib.utils import get_indexes_of_good_datapoints

DATASET_FOLDER = 'dataset'


def _read_and_create(url, names, sep=',', drop_names=None, url_test=None, name_in_zip=None):
    import os
    import pandas as pd

    filename = url.split('/')[-1]

    if not os.path.exists(os.path.join(ROOT_DIR, DATASET_FOLDER)):
        os.makedirs(os.path.join(ROOT_DIR, DATASET_FOLDER))

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

    if url.split('.')[-1] == 'zip':
        import zipfile
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(ROOT_DIR, DATASET_FOLDER))

        filepath = os.path.join(ROOT_DIR, DATASET_FOLDER, name_in_zip)

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


def _get_data(url, names, positive_label, target_feature='class', sep=',', drop_names=None, url_test=None,
              name_in_zip=None):
    dataset = _read_and_create(url, names, sep=sep, drop_names=drop_names, url_test=url_test, name_in_zip=name_in_zip)
    X, y = _split_X_y(dataset, positive_label, target_feature=target_feature)
    assert y.sum() != 0
    # count classes

    return X, y


def get_float_dataset(dataset_name='TTT'):
    if dataset_name == 'ADULT':
        X, y = get_dataset(dataset_name=dataset_name)

        df = pandas.DataFrame()

        # keep numerical data
        df['Age'] = X['Age']
        df['Hours per week'] = X['Hours per week']
        df['Capital Gain'] = X['Capital Gain']
        df['Capital Loss'] = X['Capital Loss']
        df['Education-Num'] = X['Education-Num']
        df['fnlwgt'] = X['fnlwgt']

        # apply standard one hot encoding only to categorical features
        enc = OneHotEncoder(handle_unknown='ignore')
        X = enc.fit_transform(
            X.drop(['Age', 'Hours per week', 'Capital Gain', 'Capital Loss', 'Education-Num', 'fnlwgt'],
                   axis=1)).toarray().astype(int)
        X = np.concatenate((X, df.values), axis=1)

    elif dataset_name == 'MARKET':
        X, y = get_dataset(dataset_name=dataset_name)

        df = pandas.DataFrame()

        # keep numerical data
        df['age'] = X['age']
        df['day'] = X['day']
        df['balance'] = X['balance']
        df['duration'] = X['duration']
        df['campaign'] = X['campaign']
        df['pdays'] = X['pdays']

        # apply standard one hot encoding only to categorical features
        enc = OneHotEncoder(handle_unknown='ignore')
        X = enc.fit_transform(
            X.drop(['age', 'day', 'balance', 'duration', 'campaign', 'pdays'],
                   axis=1)).toarray().astype(int)
        X = np.concatenate((X, df.values), axis=1)

    else:
        # apply standard one hot encoding
        X, y = get_dataset(dataset_name=dataset_name)
        enc = OneHotEncoder(handle_unknown='ignore')
        X = enc.fit_transform(X).toarray().astype(int)

        # X = pd.get_dummies(X)

    return X, y


def get_discrete_dataset(dataset_name='TTT'):
    if dataset_name == 'ADULT':
        X, y = get_dataset(dataset_name=dataset_name)
        # discretize age using 5 bins with quantile strategy
        X['Age'] = pd.qcut(X['Age'], 5, labels=None,
                           duplicates='drop')  # ['very low', 'low', 'medium', 'high', 'very high'])
        # hours per week
        X['Hours per week'] = pd.qcut(X['Hours per week'], 5, labels=None,
                                      duplicates='drop')  # labels=['very low', 'low', 'medium', 'high', 'very high'])
        # capital gain
        X['Capital Gain'] = pd.qcut(X['Capital Gain'], 5, labels=None,
                                    duplicates='drop')  # labels=['very low', 'low', 'medium', 'high', 'very high'])
        # capital loss
        X['Capital Loss'] = pd.qcut(X['Capital Loss'], 5, labels=None,
                                    duplicates='drop')  # labels=['very low', 'low', 'medium', 'high', 'very high'])
        # education num
        X['Education-Num'] = pd.qcut(X['Education-Num'], 5, labels=None,
                                     duplicates='drop')  # labels=['very low', 'low', 'medium', 'high', 'very high'])
        # fnlwgt
        X['fnlwgt'] = pd.qcut(X['fnlwgt'], 5, labels=None,
                              duplicates='drop')  # labels=['very low', 'low', 'medium', 'high', 'very high'])

        # convert X to string
        X = X.astype(str)

    elif dataset_name == 'MARKET':
        X, y = get_dataset(dataset_name=dataset_name)
        # discretize age using 5 bins with quantile strategy
        X['age'] = pd.qcut(X['age'], 5, labels=None, duplicates='drop')
        # day
        X['day'] = pd.qcut(X['day'], 5, labels=None, duplicates='drop')
        # balance
        X['balance'] = pd.qcut(X['balance'], 5, labels=None, duplicates='drop')
        # duration
        X['duration'] = pd.qcut(X['duration'], 5, labels=None, duplicates='drop')
        # campaign
        X['campaign'] = pd.qcut(X['campaign'], 5, labels=None, duplicates='drop')
        # pdays
        X['pdays'] = pd.qcut(X['pdays'], 5, labels=None, duplicates='drop')

        # convert X to string
        X = X.astype(str)

    elif dataset_name == 'CERV':
        X, y = get_dataset(dataset_name=dataset_name)

        # replace 'unk' values with NaN
        X.replace('unk', np.nan, inplace=True)

        # calculate median for each column
        medians = X.median()

        # replace NaN values with corresponding medians
        X.fillna(medians, inplace=True)

        X = X.astype(float)

        # discretize age using 5 bins with quantile strategy
        X['Age'] = pd.qcut(X['Age'], 5, labels=None, duplicates='drop')
        # First sexual intercourse
        X['First sexual intercourse'] = pd.qcut(X['First sexual intercourse'], 5, labels=None, duplicates='drop')
        # Num of pregnancies
        X['Num of pregnancies'] = pd.qcut(X['Num of pregnancies'], 5, labels=None, duplicates='drop')
        # Smokes (years)
        X['Smokes (years)'] = pd.qcut(X['Smokes (years)'], 5, labels=None, duplicates='drop')
        # Smokes (packs/year)
        X['Smokes (packs/year)'] = pd.qcut(X['Smokes (packs/year)'], 5, labels=None, duplicates='drop')
        # Hormonal Contraceptives (years)
        X['Hormonal Contraceptives (years)'] = pd.qcut(X['Hormonal Contraceptives (years)'], 5, labels=None,
                                                       duplicates='drop')
        X = X.astype(str)

    else:
        X, y = get_dataset(dataset_name=dataset_name)
    return X, y


def get_dataset_continue_and_discrete(dataset_name='TTT', verbose=False):
    X_cont, y = get_float_dataset(dataset_name=dataset_name)
    X_disc, _ = get_discrete_dataset(dataset_name=dataset_name)

    if verbose:
        print(f"removing inconsistent data from {dataset_name}, N={len(X_cont)}")
    idx = get_indexes_of_good_datapoints(X_disc, y)
    if verbose:
        print(f"removed {len(X_cont) - len(idx)} inconsistent data from {dataset_name}, N={len(idx)}")
    return X_cont[list(idx)], X_disc.iloc[idx], y[list(idx)]


def get_dataset(dataset_name='TTT', return_extra_info=False, remove_inconsistent=False):
    positive_label = None
    target_feature = None
    # TTT
    if dataset_name == 'TTT':
        positive_label = 'positive'
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data',
            names=[letter + number for letter in 'a' for number in '123456789'] + ['class'],
            positive_label=positive_label)

    elif dataset_name == 'ADULT':
        positive_label = ' >50K'
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            names=['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital Status', 'Occupation',
                   'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week',
                   'Native Country', 'class'],
            positive_label=positive_label, target_feature=target_feature
        )

    elif dataset_name == 'MARKET':
        positive_label = 'yes'
        target_feature = 'y'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip',
            names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                   'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'],
            positive_label=positive_label, target_feature=target_feature, name_in_zip='bank-full.csv', sep=';')

    elif dataset_name == 'COMPAS':
        positive_label = 1
        target_feature = 'Recidivate-Within-Two-Years'
        X, y = _get_data(
            url='https://raw.githubusercontent.com/fingoldin/pycorels/master/examples/data/compas.csv',
            names=['Age=18-20', 'Age=18-22', 'Age=18-25', 'Age=24-30', 'Age=24-40', 'Age>=30', 'Age<=40', 'Age<=45',
                   'Gender=Male', 'Race=African-American', 'Race=Caucasian', 'Race=Asian', 'Race=Hispanic',
                   'Race=Native-American', 'Race=Other', 'Juvenile-Felonies=0', 'Juvenile-Felonies=1-3',
                   'Juvenile-Felonies>3', 'Juvenile-Crimes=0', 'Juvenile-Crimes=1-3', 'Juvenile-Crimes>3',
                   'Juvenile-Crimes>5', 'Prior-Crimes=0', 'Prior-Crimes=1-3', 'Prior-Crimes>3', 'Prior-Crimes>5',
                   'Current-Charge-Degree=Misdemeanor', 'Recidivate-Within-Two-Years'],
            positive_label=positive_label, target_feature=target_feature)

    elif dataset_name == 'CERV':
        positive_label = 1
        target_feature = 'Biopsy'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv',
            names=['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
                   'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives',
                   'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)',
                   'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
                   'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
                   'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
                   'STDs:HPV', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
                   'STDs: Time since last diagnosis', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
                   'Citology', 'Biopsy'],
            positive_label=positive_label, target_feature=target_feature)


    elif dataset_name == 'HIV':
        positive_label = 1
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00330/newHIV-1_data.zip',
            names=['amino', 'class'],
            positive_label=positive_label,
            name_in_zip='746Data.txt'
        )
        X = pd.DataFrame([np.array([c for c in row[0]]) for row in X.values])
        assert y.sum() != 0

    elif dataset_name == 'BREAST':
        positive_label = 'recurrence-events'
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
            names=['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                   'breast-quad', 'irradiat'],
            positive_label=positive_label, target_feature=target_feature
        )

    elif dataset_name == 'BALANCE':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
            names=['class', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
            positive_label='B'
        )

    elif dataset_name == 'BALLOONS1':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data',
            names=['color', 'size', 'act', 'age', 'inflated'],
            positive_label='T')

    elif dataset_name == 'BALLOONS2':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult-stretch.data',
            names=['color', 'size', 'act', 'age', 'inflated'],
            positive_label='T')

    elif dataset_name == 'BALLOONS3':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small+adult-stretch.data',
            names=['color', 'size', 'act', 'age', 'inflated'],
            positive_label='T')

    elif dataset_name == 'BALLOONS4':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small.data',
            names=['color', 'size', 'act', 'age', 'inflated'],
            positive_label='T')

    elif dataset_name == 'LENSES':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data',
            names=['id', 'age', 'spectacle-prescrip', 'astigmatic', 'tear-prod-rate', 'class'],
            positive_label='1'
        )

    elif dataset_name == 'NURSERY':
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data',
            names=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class'],
            positive_label='not_recom'
        )
        y = 1 - y
        # rules found are X[7] == recommended and x[7] == priority, this means that health is very correlated to class
        # and should be removed (otherwise the problem is trivial)

    elif dataset_name == 'PRIMARY':
        positive_label = 1
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/primary-tumor/primary-tumor.data',
            names=['class', 'age', 'sex', 'histologic-type', 'degree-of-diffe',
                   'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain', 'skin', 'neck',
                   'supraclavicular',
                   'axillar', 'mediastinum', 'abdominal'],
            positive_label=positive_label, target_feature=target_feature
        )  # check the most frequent class

    elif dataset_name == 'LYMPHOGRAPHY':
        positive_label = 2
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data',
            names=['class', 'lymphatics', 'block-of-afferent', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass',
                   'extravasates',
                   'regeneration of', 'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar', 'changes in lym.',
                   'defect in node',
                   'changes in node', 'changes in stru', 'special forms', 'dislocation of', 'exclusion of no',
                   'no. of nodes in'],
            positive_label=positive_label, target_feature=target_feature
        )

    elif dataset_name == 'SOYBEAN':
        positive_label = 'frog-eye-leaf-spot'
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data',
            names=['class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity',
                   'seed-tmt',
                   'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size',
                   'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion',
                   'fruiting-bodies',
                   'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed',
                   'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots'],
            positive_label=positive_label, target_feature=target_feature
        )

    elif dataset_name == 'VOTE':
        positive_label = 'republican'
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
            names=['class', 'handicapped-infants', 'water-project-cost-sharing',
                   'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                   'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                   'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
                   'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
                   'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
            positive_label=positive_label, target_feature=target_feature)

    elif dataset_name == 'KR-VS-KP':
        positive_label = 'won'
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data',
            names=['bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq',
                   'bkxcr', 'bkxwp', 'blxwp', 'bxqsq', 'cntxt', 'dsopp',
                   'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8',
                   'reskd', 'reskr', 'rimmx', 'rkxwp', 'rxmsq', 'simpl',
                   'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk',
                   'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg'] + ['class'], sep=',',
            positive_label=positive_label, target_feature=target_feature)

    elif dataset_name == 'CONNECT-4':
        positive_label = 'win'
        target_feature = 'class'
        X, y = _get_data(
            url='http://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z',
            names=[letter + number for letter in 'abcdefg' for number in '123456'] + ['class'],
            positive_label=positive_label, target_feature=target_feature)

    elif dataset_name == 'CAR':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        positive_label = 'unacc'
        target_feature = 'class'
        X, y = _get_data(url, names, positive_label=positive_label, target_feature=target_feature)

    elif dataset_name == 'SPECT':
        positive_label = 1
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train',
            names=['class'] + [f'F{i}' for i in range(1, 23)], sep=',',
            positive_label=positive_label, target_feature=target_feature,
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test')

    elif dataset_name == 'MONKS1':
        positive_label = 1
        target_feature = 'class'
        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train',
            names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id'], sep=' ',
            positive_label=positive_label, target_feature=target_feature,
            drop_names=['Id'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test')

    # note: monks2 seems to perform better using tol=0, differently as noted in the paper.
    elif dataset_name == 'MONKS2':
        positive_label = 1
        target_feature = 'class'

        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train',
            names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id'], sep=' ',
            positive_label=positive_label, target_feature=target_feature,
            drop_names=['Id'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test')

    elif dataset_name == 'MONKS3':
        positive_label = 1
        target_feature = 'class'

        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train',
            names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id'], sep=' ',
            positive_label=positive_label, target_feature=target_feature,
            drop_names=['Id'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test')

    elif dataset_name == 'MUSH':
        positive_label = 'e'
        target_feature = 'class'

        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
            names=['class'] + ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
                               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                               'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                               'stalk-surface-below-ring', 'stalk-color-above-ring',
                               'stalk-color-below-ring', 'veil-type', 'veil-color',
                               'ring-number', 'ring-type', 'spore-print-color',
                               'population', 'habitat'],
            positive_label=positive_label, target_feature=target_feature)

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

    elif dataset_name == 'LED':
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(data_id=40496, return_X_y=True)

        # make y as binary (select "0" as positive class)
        y = y == '4'

    elif dataset_name == 'AUDIO':
        positive_label = 'cochlear_age'
        target_feature = 'class'

        X, y = _get_data(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.data',
            names=[
                'age_gt_60', 'air', 'airBoneGap', 'ar_c', 'ar_u', 'bone', 'boneAbnormal',
                'bser', 'history_buzzing', 'history_dizziness', 'history_fluctuating', 'history_fullness',
                'history_heredity', 'history_nausea', 'history_noise', 'history_recruitment', 'history_ringing',
                'history_roaring', 'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k', 'm_gt_1k',
                'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k', 'm_m_sn_gt_500', 'm_p_sn_gt_2k',
                'm_s_gt_500', 'm_s_sn', 'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k',
                'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k', 'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k',
                'm_sn_lt_1k', 'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k', 'mod_mixed', 'mod_s_mixed',
                'mod_s_sn_gt_500',
                'mod_sn', 'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k', 'mod_sn_gt_500',
                'notch_4k', 'notch_at_4k', 'o_ar_c', 'o_ar_u', 's_sn_gt_1k', 's_sn_gt_2k',
                's_sn_gt_4k', 'speech', 'static_normal', 'tymp', 'viith_nerve_signs', 'wave_V_delayed',
                'waveform_ItoV_prolonged', 'indentifier', 'class'
            ], sep=',',
            positive_label=positive_label, target_feature=target_feature,
            drop_names=['indentifier'],
            url_test='https://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.test')

    else:
        raise NotImplementedError(f'The dataset {dataset_name} was not found.')

    if remove_inconsistent:
        idx = get_indexes_of_good_datapoints(X, y)
        X = X.iloc[idx]
        y = y[idx]

    if return_extra_info:
        return X, y, positive_label, target_feature
    else:
        return X, y
