# outputs: a dataframe that contains each model computed and various metrics of interest.
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from bpllib import get_dataset
from bpllib._find_rs_bp import FindRsClassifier
from bpllib._id3_bp import Id3Classifier
from bpllib._ripper_bp import RipperClassifier
from bpllib._aq_bp import AqClassifier
from bpllib.utils import remove_inconsistent_data

if __name__ == '__main__':

    output_file = 'results.pkl'

    # check if file exists
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame()
        df['dataset'] = []
        df['dataset_size'] = []
        df['seed'] = []
        df['strategy'] = []
        df['method'] = []
        df['model'] = []
        df['accuracy'] = []
        df['f1'] = []
        df['avg_rule_len'] = []
        df['avg_ruleset_len'] = []


    def compute_stats(model, X_test, y_test, strategy):
        y_pred = model.predict(X_test, strategy=strategy)
        f1 = f1_score(y_test, y_pred)

        if strategy == 'single':
            avg_rule_len = np.mean([len(rule) for rule in model.rule_sets_[0]])
            avg_ruleset_len = len(model.rule_sets_[0])

        elif strategy == 'bo':
            avg_rule_len = np.mean([len(rule) for ruleset in model.rule_sets_ for rule in ruleset])
            avg_ruleset_len = np.mean([len(ruleset) for ruleset in model.rule_sets_])

        elif strategy == 'bp':
            avg_rule_len = np.mean([len(rule) for rule in model.counter_.keys()])
            avg_ruleset_len = len(model.counter_)

        elif strategy == 'best-k':
            rules = model.counter_.most_common(model.suggested_k_)
            avg_rule_len = np.mean([len(rule) for rule in [r[0] for r in rules]])
            avg_ruleset_len = len(rules)

        else:
            raise NotImplementedError('Strategy not implemented.')

        return accuracy_score(y_test, y_pred), f1, avg_rule_len, avg_ruleset_len


    easy_config = {
        'Find-RS': {
            'T': 100
        },
        'RIPPER': {
            'T': 100
        },
        'ID3': {
            'T': 100
        },
        'AQ': {
            'T': 10,
            'maxstar': 2
        }
    }

    medium_config = {
        'Find-RS': {
            'T': 100
        },
        'RIPPER': {
            'T': 100
        },
        'ID3': {
            'T': 20
        },
        'AQ': {
            'T': 5,
            'maxstar': 1
        }
    }

    hard_config = {
        'Find-RS': {
            'T': 20
        },
        'RIPPER': {
            'T': 20
        },
        'ID3': {
            'T': 10
        },
        'AQ': {
            'T': 3,
            'maxstar': 1
        }
    }

    debug_config = {
        'Find-RS': {
            'T': 3
        },
        'RIPPER': {
            'T': 3
        },
        'ID3': {
            'T': 3
        },
        'AQ': {
            'T': 3,
            'maxstar': 2
        }
    }

    DEBUG = False

    easy_datasets = ['CAR', 'TTT',
                     # 'BREAST', has something strange, low f1
                     'HIV', 'LYMPHOGRAPHY', 'PRIMARY', 'MONKS1',
                     # 'MONKS2', too
                     'MONKS3', 'VOTE']
    medium_datasets = ['COMPAS', 'MUSH', 'KR-VS-KP']
    hard_datasets = ['CONNECT-4', 'ADULT']

    methods = [FindRsClassifier, RipperClassifier, Id3Classifier]  # , AqClassifier]

    # 1. run baselines

    for DATASET in easy_datasets + medium_datasets + hard_datasets:
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

        # ['CAR', 'TTT', 'MUSH', 'MONKS1', 'MONKS2', 'MONKS3', 'KR-VS-KP', 'VOTE', 'BREAST', 'HIV','LYMPHOGRAPHY', 'PRIMARY']:
        X, y = get_dataset(DATASET)

        X = X.astype(str)

        if DATASET in ['BREAST', 'PRIMARY', 'MONKS3', 'COMPAS', 'ADULT', 'MONKS2']:
            old_len = len(X)
            print(f"removing inconsistent data from {DATASET}, N={len(X)}")
            X, y = remove_inconsistent_data(X, y)
            print("removed", old_len - len(X), "inconsistent data points (now N={} left)".format(len(X)))
        for DATASET_SIZE in [0.5]:
            for SEED in range(1):
                for METHOD in methods:
                    # check if this experiment has already been run
                    if len(df[(df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (
                            df['seed'] == SEED) & (df['method'] == METHOD.description)]) > 0:
                        print(f'Skipping {METHOD.description} on {DATASET} with size {DATASET_SIZE} and seed {SEED}...')
                        continue

                    print(
                        f'Running {METHOD.description} (T={config[METHOD.description]["T"]}) on {DATASET} with size {DATASET_SIZE} and seed {SEED}...')

                    # run the experiment
                    model = METHOD(**config[METHOD.description], pool_size=5, random_state=SEED, verbose=10)

                    T = config[METHOD.description]['T'] if 'T' in config[METHOD.description] else 1

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                                        random_state=SEED, shuffle=True)
                    model.fit(X_train, y_train)

                    for strategy in ['single', 'bo', 'bp', 'best-k']:
                        # compute metrics
                        accuracy, f1, avg_rule_len, avg_ruleset_len = compute_stats(model, X_test, y_test, strategy)

                        assert f1 > 0

                        new_row = {
                            'strategy': strategy,
                            'method': METHOD.description,
                            'dataset': DATASET,
                            'dataset_size': DATASET_SIZE,
                            'seed': SEED,
                            'T': T,
                            'model': model,
                            'accuracy': accuracy,
                            'f1': f1,
                            'avg_rule_len': avg_rule_len,
                            'avg_ruleset_len': avg_ruleset_len
                        }

                        print(new_row)

                        # save the results
                        df = df.append(new_row, ignore_index=True)

                    # Save dataframe to temporary file
                    temp_file = 'tmp.pkl'
                    df.to_pickle(temp_file)

                    # Rename temporary file to final file name
                    final_file_name = output_file
                    os.replace(temp_file, final_file_name)
