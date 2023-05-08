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
    'TTT',
    'MONKS1',
    'MONKS2',
    'MONKS3'
]

ground_truths = [
    [
        Rule({0: DiscreteConstraint('x', 0), 1: DiscreteConstraint('x', 1), 2: DiscreteConstraint('x', 2)}),
        Rule({3: DiscreteConstraint('x', 3), 4: DiscreteConstraint('x', 4), 5: DiscreteConstraint('x', 5)}),
        Rule({6: DiscreteConstraint('x', 6), 7: DiscreteConstraint('x', 7), 8: DiscreteConstraint('x', 8)}),
        Rule({0: DiscreteConstraint('x', 0), 3: DiscreteConstraint('x', 3), 6: DiscreteConstraint('x', 6)}),
        Rule({1: DiscreteConstraint('x', 1), 4: DiscreteConstraint('x', 4), 7: DiscreteConstraint('x', 7)}),
        Rule({2: DiscreteConstraint('x', 2), 5: DiscreteConstraint('x', 5), 8: DiscreteConstraint('x', 8)}),
        Rule({0: DiscreteConstraint('x', 0), 4: DiscreteConstraint('x', 4), 8: DiscreteConstraint('x', 8)}),
        Rule({2: DiscreteConstraint('x', 2), 4: DiscreteConstraint('x', 4), 6: DiscreteConstraint('x', 6)})
    ],
    [
        Rule({0: DiscreteConstraint('1', 0), 1: DiscreteConstraint('1', 1)}),
        Rule({0: DiscreteConstraint('2', 0), 1: DiscreteConstraint('2', 1)}),
        Rule({0: DiscreteConstraint('3', 0), 1: DiscreteConstraint('3', 1)}),
        Rule({4: DiscreteConstraint('1', 4)}),
    ],
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
    #    2. a1:    1, 2, 3
    # 3. a2:    1, 2, 3
    # 4. a3:    1, 2
    # 5. a4:    1, 2, 3
    # 6. a5:    1, 2, 3, 4
    # 7. a6:    1, 2

    # (a5 = 3 and a4 = 1) or (a5 /= 4 and a2 /= 3)

    # then, ohe is:
    # x[13]=1 and x[15]=1 or x[14]=0 and x[5]=0
    [
        Rule({13: DiscreteConstraint('1.0', 13), 15: DiscreteConstraint('1.0', 15)}),
        Rule({14: DiscreteConstraint('0.0', 14), 5: DiscreteConstraint('0.0', 5)}),
    ]
]

print("""
\\begin{table}[ht]
\\centering
\\small

\\begin{tabular}{ l l | c c c c  }
\\hline
dataset & size & find-rs & ripper & id3 & aq \\\\ 
 \\hline
""")

filename = f'../forsimilarityy.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

# read from the pickle file
with open(filename, 'rb') as f:
    df = pickle.load(f)

for DATASET, ground_truth in zip(test_datasets, ground_truths):
    for DATASET_SIZE in [0.25, 0.33, 0.5]:
        print(f'\n{DATASET} & {DATASET_SIZE} & \t', end='')
        for METHOD in ['Find-RS', 'RIPPER', 'ID3', 'AQ']:  # , 'BRS']:
            # [0.5, 0.33, 0.25]:
            encoding = 'ohe' if METHOD == 'BRS' or DATASET in ['MONKS2', 'MONKS3'] else 'av'
            rows = df[(df['method'] == METHOD) & (df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (
                    df['strategy'] == 'bp') & (df['encoding'] == encoding)]
            if len(rows) == 0:
                print('-- \t', end=' & \t' if METHOD != 'AQ' else ' \\\\\n')
                continue
            row = rows.iloc[0]

            model = row['model']
            seed = row['seed']


            def similarity(rs1, rs2):
                return len(set(rs1).intersection(set(rs2))) / len(set(rs1).union(set(rs2)))


            if model.rule_sets_ is None:
                # monks3/brs fails to find a ruleset
                print('- \t', end=' & \t' if METHOD != 'AQ' else ' \\\\\n')
                continue
            similarities = [similarity(ground_truth, ruleset) for ruleset in model.rule_sets_]
            print(f'{np.mean(similarities):.4f}', end=' & \t' if METHOD != 'AQ' else ' \\\\\n')
print("""
\\hline
\\end{tabular}
\\caption{Average similarity of the rulesets found by the BP methods to the ground truth rulesets.}
\\label{tab:bp-similarity}
\\end{table}
""")
