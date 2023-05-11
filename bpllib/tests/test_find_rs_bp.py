import numpy as np
import pytest
from joblib import Memory
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from bpllib import get_dataset
from bpllib._dataset import get_dataset_continue_and_discrete
from bpllib._find_rs_bp import FindRsClassifier
from bpllib.tests.utils import run_training, data_custom

from .utils import data
import time

from ..utils import remove_inconsistent_data


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['best-k'])
def test_template_estimator_mono(data, T, pool_size, strategy):
    kwargs = {'T': T, 'pool_size': pool_size, 'verbose': 1}
    # run_training(FindRsClassifier, kwargs, data, min_f1_score=0.98)
    min_f1_score = 0.98
    for name, (X, y) in data:
        est = FindRsClassifier(**kwargs)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test, strategy=strategy)

        f1 = f1_score(y_test, y_pred)

        assert (f1 >= min_f1_score)
    print('ok')


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [5])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_multi(data, T, pool_size, strategy):
    kwargs = {'T': T, 'pool_size': pool_size, 'verbose': 1}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=0.98, strategy=strategy)


@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("pool_size", [1, 3])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_multi(data, T, pool_size, strategy):
    kwargs = {'T': T, 'pool_size': pool_size, 'verbose': 1}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=1.0, strategy=strategy)


@pytest.mark.parametrize("T", [1])
@pytest.mark.parametrize("pool_size", [1])
@pytest.mark.parametrize("strategy", ['bo', 'bp', 'single', 'best-k'])
def test_template_estimator_1(data, T, pool_size, strategy):
    kwargs = {'T': T, 'pool_size': pool_size}
    run_training(FindRsClassifier, kwargs, data, min_f1_score=0.95, strategy=strategy)


@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("pool_size", [3, 'auto'])
@pytest.mark.parametrize("strategy", ['bp'])
def test_template_estimator_mush(data, T, pool_size, strategy):
    X, y = get_dataset('MUSH')
    kwargs = {'T': T, 'pool_size': pool_size, 'verbose': 1}
    run_training(FindRsClassifier, kwargs, [('mush', (X, y))], min_f1_score=1.0, strategy=strategy)


def test_estimator_replicability():
    # create random categorical data
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(100, 10))
    y = np.random.randint(0, 2, size=(100,))

    X, y = remove_inconsistent_data(X, y)

    clf = FindRsClassifier(random_state=42, T=3)
    clf.fit(X, y)
    clf2 = FindRsClassifier(random_state=42, T=3)
    clf2.fit(X, y)

    assert clf.rule_sets_ == clf2.rule_sets_
    assert clf.counter_ == clf2.counter_
    assert clf.score(X, y) == clf2.score(X, y)


def FindRSGridSearch(**kwargs):
    param_grid_find_rs = {
        'tol': [0],  # , 1, 2],
        # 'n_bins': [3, 10, 30, 100],
        'max_rules': [5, 10, 20, 40, 80]  # , 20, 40] #, 80],
        # 'random_state': [42],
    }

    import numpy as np
    # Set up the k-fold cross-validation
    cv = KFold(n_splits=6, shuffle=False)  # True, random_state=kwargs['random_state'])
    grid_search_find_rs = GridSearchCV(
        estimator=FindRsClassifier(**kwargs),
        param_grid=param_grid_find_rs, cv=cv, n_jobs=1, error_score='raise',
        verbose=10,
        refit=True)
    return grid_search_find_rs


def test_reproducibility_cv():
    memory = Memory('cachedir', verbose=0)
    memory.clear(warn=False)

    _, X, y = get_dataset_continue_and_discrete('TTT')
    seed = 8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=seed, shuffle=True)
    clf = FindRsClassifier(random_state=seed, T=100)
    clf.fit(X_train, y_train)

    clf_cv = FindRSGridSearch(random_state=seed, T=100)
    clf_cv.fit(X_train, y_train)

    print(clf_cv.best_params_)

    y_pred = clf.predict(X_test, strategy='single')
    y_pred_cv = clf_cv.best_estimator_.predict(X_test, strategy='single')

    assert np.all(y_pred == y_pred_cv)


if __name__ == '__main__':
    test_reproducibility_cv()
