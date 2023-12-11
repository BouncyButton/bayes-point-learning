import itertools
import pickle
# get the latest result file
import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from tqdm import tqdm

from bpllib import get_dataset
from bpllib._bpl import DiscreteConstraint
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
METHOD = 'Find-RS'
filename = 'new-datasets-merged.pkl'  # f'multi-iteration-3-{METHOD}.pkl'  # max(glob.iglob('results_*.pkl'), key=os.path.getctime)

# read from the pickle file
# with open(filename, 'rb') as f:
#     df = pickle.load(f)

DATASET = 'MONKS3'

df = exp_utils.get_df(datasets=[DATASET])

encoding = 'ohe'
DATASET_SIZE = 0.5
row = df[(df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE) & (df['strategy'] == 'bp') & (
            df['encoding'] == encoding) & (df['method'] == METHOD)].iloc[0]
model = row['model']
seed = row['seed']
X, y = get_dataset(DATASET, remove_inconsistent=True)
names = X.columns

# if DATASET in ['MONKS3', 'BREAST', 'PRIMARY']:
#    X, y = remove_inconsistent_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - DATASET_SIZE,
                                                    random_state=int(seed))

X_train = model.validation(X_train)
X_test = model.validation(X_test)

print("""
\\begin{table*}[ht]
\\centering
\\small

\\begin{tabular}{ r | r r r r | p{4.3cm} }
 \\toprule
""")

print("alpha & \\multicolumn{2}{l}{rule coverage} & \\multicolumn{2}{l}{rule precision} & rule\\\\")
print("& train & test & train & test \\\\")

for rule, alpha in model.counter_.most_common(10):
    print("%.2f" % (alpha / model.T), end=' & ')


    def coverage(X, y):
        train_coverage = np.array([rule.covers(x) for x in X[y == 1]]).mean()
        assert train_coverage > 0
        print("%.3f" % (train_coverage,), end=" & ")


    coverage(X_train, y_train)
    coverage(X_test, y_test)


    def precision(X, y):
        print('%.3f' % (np.array([rule.covers(x) for x in X[y == 1]]).sum() / np.array(
            [rule.covers(x) for x in X]).sum(),),
              end=" & ")


    precision(X_train, y_train)
    precision(X_test, y_test)

    if METHOD == 'aq':
        reprs = []
        for c in rule.constraints.values():
            column = names[c.index]
            if isinstance(c, DiscreteConstraint):
                reprs.append(f'{column}={c.value}')
            else:
                reprs.append("%s $\in$ \\{%s\\}" % (column, c.values))
        final_str = " ^ ".join(reprs)
    else:
        if encoding == 'av':
            final_str = rule.str_with_column_names(names)
        else:
            final_str = rule.str_with_column_names(names, encoding='ohe', monks=True, enc=model.enc_)

    print(("\\texttt{%s}" % final_str).replace(" ^ ", " $\\land$ ")
          , end=" \\\\\n")
print('\\hline')
print('\\hline')

print("total train coverage & %.3f & &" % np.array(
    [any([rule.covers(x) for rule, alpha in model.counter_.most_common(10)]) for x in X_train[y_train == 1]]).mean())
print("total test coverage & %.3f & &" % np.array(
    [any([rule.covers(x) for rule, alpha in model.counter_.most_common(10)]) for x in X_test[y_test == 1]]).mean())
print('\\end{tabular}')
print('\\caption{Top 10 rules for dataset %s using %s}' % (DATASET, METHOD))
print('\\end{table*}')
