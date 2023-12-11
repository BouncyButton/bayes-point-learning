# outputs: a dataframe that contains each model computed and various metrics of interest.
import json
import math
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from scipy import stats

from bpllib import get_dataset, utils
from bpllib._brs_bp import BayesianRuleSetClassifier
from bpllib._dataset import get_dataset_continue_and_discrete
from bpllib._find_rs_bp import FindRsClassifier
from bpllib._id3_bp import Id3Classifier
from bpllib._ripper_bp import RipperClassifier
from bpllib._aq_bp import AqClassifier
from bpllib.utils import remove_inconsistent_data

if __name__ == '__main__':

    output_file = 'fixed-ohe-find-rs.pkl'  # find-rs-perm-cv5.pkl'

    # check if file exists
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            df = pickle.load(f)

    else:
        df = pd.DataFrame()
        df['dataset'] = []
        df['dataset_size'] = []
        df['encoding'] = []
        df['seed'] = []
        df['strategy'] = []
        df['method'] = []
        df['model'] = []
        df['accuracy'] = []
        df['f1'] = []
        df['avg_rule_len'] = []
        df['avg_ruleset_len'] = []


    def compute_stats(model, X_test, y_test, strategy):
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_

        # check if model is not a subclass of AbstractBayesPointClassifier
        if not hasattr(model, 'rule_sets_'):
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), None, None

        y_pred = model.predict(X_test, strategy=strategy)
        f1 = f1_score(y_test, y_pred)

        if strategy == 'single':
            avg_rule_len = np.mean([len(rule) for rule in model.rule_sets_[0]])
            avg_ruleset_len = len(model.rule_sets_[0])

        elif strategy == 'bo' or strategy == 'old-bo':
            avg_rule_len = np.mean([len(rule) for ruleset in model.rule_sets_ for rule in ruleset])
            avg_ruleset_len = np.mean([len(ruleset) for ruleset in model.rule_sets_])

        elif strategy == 'bp' or strategy == 'old-bp':
            avg_rule_len = np.mean([len(rule) for rule in model.counter_.keys()])
            avg_ruleset_len = len(model.counter_)

        elif strategy == 'best-k':
            rules = model.counter_.most_common(model.suggested_k_)
            avg_rule_len = np.mean([len(rule) for rule in [r[0] for r in rules]])
            avg_ruleset_len = len(rules)

        else:
            raise NotImplementedError('Strategy not implemented.')

        return accuracy_score(y_test, y_pred), f1, avg_rule_len, avg_ruleset_len


    uninterpretable_methods = ['SVM', 'RF', 'TabNet']

    easy_config = {
        'Find-RS': {
            'T': 100,
            'max_rules': 100,
            'generalization_probability': 0.9,
            'rule_pruning': True,
            # 'tol': 0,
        },
        'FindRSGridSearch': {
            'T': 100,
            'rule_pruning': True,
            'max_rules': 100,
            'bin_purity': 1.0,
        },
        'RIPPER': {
            'T': 100
        },
        'ID3': {
            'T': 100
        },
        'AQ': {
            'T': 20,
            'maxstar': 3
        },
        'BRS': {
            'T': 100,
            'num_iterations': 100,
            'maxlen': 3
        },
        'TabNet': {
            'verbose': 1,
        }
    }

    medium_config = {
        'Find-RS': {
            'T': 100,
            'max_rules': 1000,
            'generalization_probability': 0.9,
            'rule_pruning': True,
            'verbose': 0
        },
        'RIPPER': {
            'T': 100
        },
        'ID3': {
            'T': 100
        },
        'AQ': {
            'T': 5,
            'maxstar': 1
        },
        'BRS': {
            'T': 20,
            'num_iterations': 50,
            'maxlen': 3
        },
        'TabNet': {
            'verbose': 1,
        }
    }

    hard_config = {
        'Find-RS': {
            'T': 20,
            'max_rules': 10000,
            'generalization_probability': 0.9,
            'rule_pruning': True,
            'verbose': 6
        },
        'RIPPER': {
            'T': 20
        },
        'ID3': {
            'T': 20
        },
        'AQ': {  # don't even try
            'T': None,
            'maxstar': 1
        },
        'BRS': {
            'T': 20,
            'num_iterations': 50,
            'maxlen': 3
        },
        'TabNet': {
            'verbose': 1,
        }
    }

    debug_config = {
        'Find-RS': {
            'T': 3,
            'max_rules': 10000,
            'generalization_probability': 0.9,
            'rule_pruning': True,
            'verbose': 6
        },
        'RIPPER': {
            'T': 3
        },
        'ID3': {
            'T': 3
        },
        'AQ': {
            'T': 3,
            'maxstar': 1
        },
        'BRS': {
            'T': 3,
            'num_iterations': 100,
            'maxlen': 3
        },
        'TabNet': {
            'verbose': 1
        }
    }

    DEBUG = False
    POOL_SIZE = 5  # if not DEBUG else 1

    easy_datasets = utils.easy_datasets
    medium_datasets = utils.medium_datasets
    hard_datasets = utils.hard_datasets

    # Define the hyperparameter grid to search
    param_grid_rf = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        #  'min_samples_split': [2, 5, 10],
        'random_state': [42],
    }

    # Define the hyperparameter grid to search
    param_grid_svm = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'random_state': [42],
        'max_iter': [1000000],
    }

    param_grid_find_rs = {
        # 'tol': [0, 1, 2],
        # 'n_bins': [3, 10, 30, 100],
        # 'bin_purity': [0.9, 0.95, 1],
        # 'max_rules': [20, 40, 80, None],
        'generalization_probability': [0.0, 0.25, 0.5, 0.75, 0.9, 1],
        # 'permute_constraints': [True, False]
        # 'random_state': [42],
    }


    def SVM(random_state=42, **kwargs):
        # Set up the k-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_search_svm = GridSearchCV(
            estimator=SVC(random_state=random_state, verbose=False, tol=1e-3),
            param_grid=param_grid_svm, cv=cv,
            n_jobs=POOL_SIZE, error_score='raise', verbose=4, refit=True)
        return grid_search_svm


    def RF(random_state=42, **kwargs):
        # Set up the k-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=random_state),
                                      param_grid=param_grid_rf, cv=cv, n_jobs=POOL_SIZE, error_score='raise', verbose=1,
                                      refit=True)
        return grid_search_rf


    def FindRSGridSearch(**kwargs):
        import numpy as np
        # Set up the k-fold cross-validation
        cv = KFold(n_splits=5, shuffle=False)  # True, random_state=kwargs['random_state'])
        grid_search_find_rs = GridSearchCV(
            estimator=FindRsClassifier(**kwargs),
            param_grid=param_grid_find_rs, cv=cv, n_jobs=1, error_score='raise',
            verbose=10,
            refit=True, scoring='f1'
        )
        return grid_search_find_rs


    def TabNet(**kwargs):
        from pytorch_tabnet.tab_model import TabNetClassifier as TNC

        return TNC(**kwargs)


    methods = [  # SVM,
        # RF,
        # FindRSGridSearch,
        FindRsClassifier,
        # TabNet
        # RipperClassifier,
        # BayesianRuleSetClassifier,
        # Id3Classifier,
        # AqClassifier
    ]

    # 1. run baselines
    for SEED in range(3):
        for DATASET in ['MARKET']:  # + hard_datasets:  # , 'CONNECT-4']:
            if DEBUG:
                config = debug_config
            elif DATASET in easy_datasets:
                config = easy_config
            elif DATASET in medium_datasets:
                config = medium_config
            elif DATASET in hard_datasets:
                config = hard_config
            else:
                raise NotImplementedError('Dataset not found in config.')

            X_cont, X_discr, y = get_dataset_continue_and_discrete(DATASET)

            for DATASET_SIZE in [0.5]:
                # if DATASET in hard_datasets and SEED > 0:
                #    print(f'Skipping hard dataset seed={SEED}>0...')
                #    continue
                for METHOD in methods:
                    if METHOD == AqClassifier and DATASET in hard_datasets + medium_datasets:
                        print('Skipping AQ on hard or med dataset...')
                        continue

                    descr = METHOD.description if hasattr(METHOD, 'description') else METHOD.__name__

                    cfg = config.get(descr, dict())
                    T = cfg.get('T', 1)

                    for encoding in ['ohe']:  # ['av', 'ohe']:
                        if encoding == 'av' and METHOD in [BayesianRuleSetClassifier]:
                            continue
                        if encoding == 'ohe' and METHOD in [TabNet]:
                            continue
                        if encoding == 'ohe' and DATASET == 'CONNECT-4' and METHOD in [FindRsClassifier]:
                            continue
                        if encoding == 'av' and DATASET == 'ADULT' and METHOD in [FindRsClassifier]:
                            continue
                        if encoding == 'av' and DATASET == 'MARKET' and METHOD in [FindRsClassifier]:
                            continue
                        if encoding == 'ohe' and DATASET == 'MUSH' and METHOD in [FindRsClassifier]:
                            continue

                        # check if this experiment has already been run
                        if len(df[(df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (
                                df['seed'] == SEED) & (df['method'] == descr) & (df['encoding'] == encoding)]) > 0:
                            print(
                                f'Skipping {descr} on {DATASET} with size {DATASET_SIZE} and seed {SEED} ({encoding})...')
                            continue

                        print(
                            f'Running {descr} (T={T}) on {DATASET} with '
                            f'size {DATASET_SIZE} and seed {SEED} (enc={encoding})...')

                        # run the experiment
                        if METHOD != TabNet:
                            model = METHOD(**cfg, encoding=encoding, pool_size=POOL_SIZE, random_state=SEED,
                                           bp_verbose=1,
                                           find_best_k=False if DATASET in hard_datasets else True)
                        else:
                            cat_idxs = list(range(X_discr.shape[1]))

                            # encode X_discr using a categorical encoder
                            X_discr = np.array([LabelEncoder().fit_transform(X_discr.values[:, i]) for i in cat_idxs]).T

                            cat_dims = [len(np.unique(X_discr[:, i])) for i in cat_idxs]
                            model = METHOD(**cfg, seed=SEED, cat_idxs=cat_idxs, cat_dims=cat_dims)
                        # if method is svm or rf, use float data
                        if descr in ['SVM', 'RF']:
                            X = X_cont
                            if encoding == 'ohe':
                                print('(skipping ohe for svm and rf)')
                                break  # lazy patch (svm and rf already compute ohe on their own if needed
                                # no need for two runs)
                        else:
                            X = X_discr

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                            random_state=SEED, shuffle=True)

                        # if train > 10000, take 10000 and move other to test
                        # numpy
                        # if len(X_train) > 10000:
                        #     X_test = np.concatenate([X_test, X_train[10000:]])
                        #     y_test = np.concatenate([y_test, y_train[10000:]])
                        #     X_train = X_train[:10000]
                        #     y_train = y_train[:10000]

                        if METHOD != TabNet:
                            model.fit(X_train, y_train)
                        else:
                            if config == easy_config:
                                batch_size = 64
                                max_epochs = 500
                            elif config == medium_config:
                                batch_size = 256
                                max_epochs = 250
                            elif config == hard_config:
                                batch_size = 1024
                                max_epochs = 100
                            else:
                                raise ValueError('Config not found.')
                            model.fit(X_train, y_train, batch_size=batch_size, max_epochs=max_epochs)

                        avg_performance = None

                        if isinstance(model, GridSearchCV):
                            print(model.best_params_)

                            # Extract the hyperparameter values and corresponding scores
                            hyperparams = model.cv_results_['params']
                            scores = model.cv_results_['mean_test_score']

                            # Calculate the average performance for each hyperparameter value
                            avg_performance = {}
                            for param in param_grid_find_rs:
                                avg_performance[param] = {}
                                unique_values = np.unique([params[param] for params in hyperparams])
                                for value in unique_values:
                                    mask = np.array([params[param] == value for params in hyperparams])
                                    avg_performance[param][value] = np.mean(scores[mask])

                            # Print the average performance for each hyperparameter value
                            for param in avg_performance:
                                print(f"Average performance for {param}:")
                                for value in avg_performance[param]:
                                    print(f"{value}: {avg_performance[param][value]}")
                                print()

                        strategies = ['single', 'bo', 'bp', 'best-k'] if DATASET not in hard_datasets else ['single',
                                                                                                            'bo',
                                                                                                            'bp']
                        for strategy in strategies:
                            # compute metrics
                            accuracy, f1, avg_rule_len, avg_ruleset_len = compute_stats(model, X_test, y_test, strategy)

                            inner_model = model if not isinstance(model, GridSearchCV) else model.best_estimator_
                            find_rs_model = inner_model if isinstance(inner_model, FindRsClassifier) else None

                            new_row = {
                                'strategy': strategy,
                                'method': descr,
                                'encoding': encoding,
                                'dataset': DATASET,
                                'dataset_size': DATASET_SIZE,
                                'seed': SEED,
                                'T': T,
                                'model': model,
                                'accuracy': accuracy,
                                'f1': f1,
                                'avg_rule_len': avg_rule_len,
                                'avg_ruleset_len': avg_ruleset_len,
                                'avg_performance': json.dumps(avg_performance) if avg_performance is not None else None,
                                'bin_size_frequency': json.dumps(
                                    find_rs_model.get_bin_size_frequency()) if find_rs_model is not None else None
                            }

                            if f1 <= 0:
                                print('[!] F1 is zero {}')
                            print(new_row)

                            # save the results
                            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

                            # lazy patch (there are no strategies for svm and rf)
                            if descr in uninterpretable_methods:
                                break

                        # Save dataframe to temporary file
                        temp_file = 'tmp.pkl'
                        df.to_pickle(temp_file)

                        # Rename temporary file to final file name
                        final_file_name = output_file
                        os.replace(temp_file, final_file_name)
