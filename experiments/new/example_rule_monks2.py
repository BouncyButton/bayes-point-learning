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
from experiments.new import exp_utils


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
    'MONKS2'
]

METHOD = 'Find-RS'

ground_truths = [
    # EXACTLY TWO of {a1 = 1, a2 = 1, a3 = 1, a4 = 1, a5 = 1, a6 = 1}
    # let's use OH to encode this concept
    # (a1=1)->1, (a2=1)->1, (a3=1)->0, (a4=1)->0, (a5=1)->0, (a6=1)->0
    # (a1=1)->1, (a2=1)->0, (a3=1)->1, (a4=1)->0, (a5=1)->0, (a6=1)->0
    # (a1=1)->1, (a2=1)->0, (a3=1)->0, (a4=1)->1, (a5=1)->0, (a6=1)->0
    # (a1=1)->1, (a2=1)->0, (a3=1)->0, (a4=1)->0, (a5=1)->1, (a6=1)->0
    # (a1=1)->1, (a2=1)->0, (a3=1)->0, (a4=1)->0, (a5=1)->0, (a6=1)->1
    # (a1=1)->0, (a2=1)->1, (a3=1)->1, (a4=1)->0, (a5=1)->0, (a6=1)->0
    # (a1=1)->0, (a2=1)->1, (a3=1)->0, (a4=1)->1, (a5=1)->0, (a6=1)->0
    # (a1=1)->0, (a2=1)->1, (a3=1)->0, (a4=1)->0, (a5=1)->1, (a6=1)->0
    # (a1=1)->0, (a2=1)->1, (a3=1)->0, (a4=1)->0, (a5=1)->0, (a6=1)->1
    # (a1=1)->0, (a2=1)->0, (a3=1)->1, (a4=1)->1, (a5=1)->0, (a6=1)->0
    # (a1=1)->0, (a2=1)->0, (a3=1)->1, (a4=1)->0, (a5=1)->1, (a6=1)->0
    # (a1=1)->0, (a2=1)->0, (a3=1)->1, (a4=1)->0, (a5=1)->0, (a6=1)->1
    # (a1=1)->0, (a2=1)->0, (a3=1)->0, (a4=1)->1, (a5=1)->1, (a6=1)->0
    # (a1=1)->0, (a2=1)->0, (a3=1)->0, (a4=1)->1, (a5=1)->0, (a6=1)->1
    # (a1=1)->0, (a2=1)->0, (a3=1)->0, (a4=1)->0, (a5=1)->1, (a6=1)->1

    # (a1=1) : idx=0
    # (a2=1) : idx=3
    # (a3=1) : idx=6
    # (a4=1) : idx=8
    # (a5=1) : idx=11
    # (a6=1) : idx=15

    [
        Rule({0: DiscreteConstraint('1.0', 0), 3: DiscreteConstraint('1.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('1.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('1.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('1.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('1.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('1.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('1.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('1.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('1.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('1.0', 3), 6: DiscreteConstraint('1.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('1.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('1.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('1.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('1.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('1.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('1.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('1.0', 6),
              8: DiscreteConstraint('1.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('1.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('1.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('1.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('1.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('1.0', 8), 11: DiscreteConstraint('1.0', 11), 15: DiscreteConstraint('0.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('1.0', 8), 11: DiscreteConstraint('0.0', 11), 15: DiscreteConstraint('1.0', 15)}),
        Rule({0: DiscreteConstraint('0.0', 0), 3: DiscreteConstraint('0.0', 3), 6: DiscreteConstraint('0.0', 6),
              8: DiscreteConstraint('0.0', 8), 11: DiscreteConstraint('1.0', 11), 15: DiscreteConstraint('1.0', 15)}),
    ],
    # since dataset is
    #    2. a1:    1, 2, 3  -> x0, x1, x2
    # 3. a2:    1, 2, 3     -> x3, x4, x5
    # 4. a3:    1, 2        -> x6, x7
    # 5. a4:    1, 2, 3     -> x8, x9, x10
    # 6. a5:    1, 2, 3, 4  -> x11, x12, x13, x14
    # 7. a6:    1, 2        -> x15, x16

    # Rule(8: dc(1, 8), 9: dc(0,9), 10: dc(0,10))

]

# print("""
# \\begin{table}[ht]
# \\centering
# \\small

print("% autogenerated by example_rule_monks2.py")

print("""
\\begin{tabular}{ c c | c c | c  }
\\toprule
 \\hline
""")

filename = f'../forsimilarityy.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

# read from the pickle file
# with open(filename, 'rb') as f:
#     df = pickle.load(f)

df = exp_utils.get_df(datasets=test_datasets)

for DATASET, ground_truth in zip(test_datasets, ground_truths):
    for DATASET_SIZE in [0.5]:  # , 0.33, 0.5, '-']:
        if DATASET_SIZE == '-':
            print('\\hline')
            continue

        # print a two level index
        # rule coverage | rule precision
        # train | test  | train | test

        print("& \\multicolumn{2}{l}{rule coverage} & \\multicolumn{2}{l}{rule precision} \\\\")
        print("& train & test & train & test \\\\")

        encoding = 'ohe'
        rows = df[(df['method'] == METHOD) & (df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (
                df['strategy'] == 'bp') & (df['encoding'] == encoding)]
        if len(rows) == 0:
            print('-- \t', end=' & \t' if METHOD != 'AQ' else ' \\\\\n')
            continue
        row = rows.iloc[0]

        model = row['model']
        seed = row['seed']

        # split train test
        X, y = get_dataset(DATASET)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

        X_train = model.validation(X_train)
        X_test = model.validation(X_test)

        # get top 5 rules by alpha
        rules = model.counter_.most_common(10)

        for rule, freq in rules:
            train_coverage = np.array([rule.covers(x) for x in X_train[y_train == 1]]).mean()
            test_coverage = np.array([rule.covers(x) for x in X_test[y_test == 1]]).mean()
            train_precision = np.array([rule.covers(x) for x in X_train[y_train == 1]]).sum() / np.array(
                [rule.covers(x) for x in X_train]).sum()
            test_precision = np.array([rule.covers(x) for x in X_test[y_test == 1]]).sum() / np.array(
                [rule.covers(x) for x in X_test]).sum()

            print("{:.3f} & {:.3f} & {:.3f} & {:.3f} & ".format(train_coverage, test_coverage, train_precision,
                                                                test_precision), end='')
            print(f"\\texttt{{{rule.str_with_ohe(model.enc_, monks=True)}}} \\\\")

print("""
\\bottomrule
\\end{tabular}
""")

# \\caption{Average similarity of the rulesets found by the BP methods to the ground truth rulesets.}
# \\label{tab:bp-similarity}
# \\end{table}
# """)
