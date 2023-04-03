import itertools
import pickle
# get the latest result file
import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from tqdm import tqdm

from bpllib import get_dataset
from bpllib._bpl import DiscreteConstraint, Rule


def remove_inconsistent_data(X, y):
    # remove from X and y all rows that are labeled inconsistently
    # i.e. if there are two rows with the same features but different labels -> remove both

    # get indexes
    X = pd.DataFrame(X)
    y = pd.Series(y)
    indexes = list(X.index)

    inconsistent_indexes = set()
    for i, j in tqdm(list(itertools.combinations(indexes, 2))):
        row_i = X.loc[i]
        row_j = X.loc[j]
        if (row_i == row_j).all() and y[i] != y[j]:
            inconsistent_indexes.add(i)
            inconsistent_indexes.add(j)

    X = X.drop(list(inconsistent_indexes))
    y = y.drop(list(inconsistent_indexes))
    return X, y


TARGET_METRIC = 'f1'

# for METHOD in ['find-rs', 'ripper', 'id3', 'aq']:

# DATASET = 'CAR'


test_datasets = [
    'CAR',
    'TTT',
    'MUSH',
    'MONKS1',
    'MONKS2',
    'MONKS3',
    'KR-VS-KP',
    'VOTE',
    'BREAST',
    'HIV',
    'LYMPHOGRAPHY',
    'PRIMARY'
]
test_datasets.sort()

print("""
\\begin{table}[ht]
\\centering
\\small

\\begin{tabular}{ l | c c c c }
\\hline
dataset & find-rs & ripper & id3 & aq  \\\\ 
 \\hline
""")
METHOD = 'find-rs'
DATASET_SIZE = 0.5
for DATASET in test_datasets:  # , ground_truth in zip(test_datasets, ground_truths):

    for dataset_size in [DATASET_SIZE]:
        print('\n\\texttt{%s} & \t' % (DATASET if DATASET != 'LYMPHOGRAPHY' else 'LYMPH'), end='')
        for method_name, method in zip(['find-rs', 'ripper', 'id3', 'aq'],
                                       ['FindRsClassifier', 'RIPPERClassifier', 'ID3Classifier', 'AqClassifier']):
            filename = 'baseline-2-all.csv'
            #filename = f'baseline-2-{method_name}.csv'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)
            #if method_name == 'aq':
            #    filename = 'aq-baselines.csv'

            df = pd.read_csv(filename)

            # filter the dataframe
            df = df[(df['dataset'] == DATASET) & (df['method'] == method) & (df['dataset_size'] == dataset_size)]
            avg_rules_len = df['avg_ruleset_len'].mean()
            avg_rule_len = df['avg_rule_len'].mean()

            filename = f'final-dat/multi-iteration-3-{method_name}.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

            # read from the pickle file
            # with open(filename, 'rb') as f:
            #     df = pickle.load(f)

            # # filter the dataframe
            # df_bo = df[(df['dataset'] == DATASET) & (df['method'] == method) &
            #            (df['dataset_size'] == dataset_size) & (df['strategy'] == 'bo')]
            # model = df_bo.iloc[0]['model']
            # avg_rules_len_bo = np.array([len(ruleset) for ruleset in model.rulesets_]).mean()
            # avg_rule_len_bo = [len(rule) for rule in ]
            #
            # df_bp = df[(df['dataset'] == DATASET) & (df['method'] == method) &
            #            (df['dataset_size'] == dataset_size) & (df['strategy'] == 'bp')]
            # avg_rules_len_bp = df_bp['avg_ruleset_len'].mean()
            # avg_rule_len_bp = df_bp['avg_rule_len'].mean()
            #
            # df_best_k = df[(df['dataset'] == DATASET) & (df['method'] == method) &
            #                (df['dataset_size'] == dataset_size) & (df['strategy'] == 'best_k')]
            # avg_rules_len_best_k = df_best_k['avg_ruleset_len'].mean()
            # avg_rule_len_best_k = df_best_k['avg_rule_len'].mean()
            print(f'{avg_rules_len:.1f} \\tiny{{{avg_rule_len:.1f}}} ', end=' & ' if method_name != 'aq' else '\\\\\n')
            # f'& {avg_rules_len_bo:.2f} & {avg_rule_len_bo:.2f} & ' +
            # f'{avg_rules_len_bp:.2f} & {avg_rule_len_bp:.2f} & {avg_rules_len_best_k:.2f} & {avg_rule_len_best_k:.2f}

print("""
\\hline
\\end{tabular}
\\caption{Average ruleset (complexity) and rule length (specificity) for each dataset. Rule length is shown in brackets. 
We used %.0f%% of the examples in the datasets to train each model.}
\\label{tab:avg_ruleset_len}
\\end{table}
""" % (DATASET_SIZE*100))
