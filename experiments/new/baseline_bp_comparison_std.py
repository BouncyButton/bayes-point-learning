import math
from itertools import product

import pandas as pd

from bpllib import utils

import sys

from experiments.new import exp_utils

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'f1'  # 'accuracy'
DATASETS = utils.easy_datasets + ['-'] + utils.medium_datasets + ['-'] + utils.hard_datasets
METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']

filename = sys.argv[2] if len(sys.argv) > 2 else '../merged.pkl'
df = exp_utils.get_df(['model'])  # pd.read_pickle(filename)
N = len(df)
print("% autogenerated by baseline_bp_comparison_std.py (N={}, filename={})".format(N, filename))

df = df[(df['strategy'] == 'single') | (df['strategy'] == 'bp')]

# group by dataset, method, encoding
df = df.groupby(['dataset', 'method', 'strategy', 'encoding']).agg({METRIC: ['mean', 'std']}).reset_index()

df = df.pivot_table(index=['dataset', 'method'], columns=['strategy', 'encoding'], values=METRIC).reset_index()

# extract for each dataset/method the best encoding for the strategy bp
df['best_encoding'] = df.apply(lambda row: 'av' if row[('mean', 'bp', 'av')] >= row[('mean', 'bp', 'ohe')] else 'ohe',
                               axis=1)

best_encoding_map = {(dataset, method): best_encoding for dataset, method, best_encoding in
                     df[['dataset', 'method', 'best_encoding']].values}

df['bp'] = df.apply(lambda row: row[('std', 'bp', row[('best_encoding', '', '')])], axis=1)
df[' single'] = df.apply(lambda row: row[('std', 'single', row[('best_encoding', '', '')])], axis=1)

df = df.pivot_table(index=['dataset'], columns=['method'], values=['bp', ' single'])  # .reset_index()

df = df.droplevel((1, 2), axis=1).swaplevel(0, axis=1).reorder_levels(['method', None], axis=1).sort_index(axis=1)
# print(df)


# print a two level latex table
# \\begin{table}[ht]
# \\centering
# \\small
# \\addtolength{\\tabcolsep}{-4pt}

print("""

\\begin{tabular}{ l | l l l l l l l l l l}
\\hline

""")

print("method ", end='')
for method in METHODS:
    print(f'& \\multicolumn{{2}}{{l}}{{{method}}} ', end='')

print('\\\\ \\hline')

print("dataset ", end='')
for method in METHODS:
    print(f'& single & bp ', end='')

print('\\\\ \\hline')

for dataset in DATASETS:
    if dataset == '-':
        print(f'\\hline')
        continue
    if dataset == 'LYMPHOGRAPHY':
        print(f'\\texttt{{LYMPH.}} ', end='')
    else:
        print(f'\\texttt{{{dataset}}} ', end='')

    for method in METHODS:
        try:
            single = df[(method, ' single')].loc[dataset]
            bp = df[(method, 'bp')].loc[dataset]
        except KeyError:
            print(f'& - & - ', end='')
            continue

        best_enc = best_encoding_map.get((dataset, method), 'av')
        marker = "" if best_enc == "av" else "$\\dagger$"

        if math.isnan(single) and math.isnan(bp):
            print(f'& - & - ', end='')

        elif single < bp:
            print(f'& \\textbf{{{single:.3f}}} & {bp:.3f} {marker}', end='')
        elif single > bp:
            print(f'& {single:.3f} & \\textbf{{{bp:.3f}}} {marker}', end='')
        else:
            print(f'& \\textbf{{{single:.3f}}} & \\textbf{{{bp:.3f}}} {marker}', end='')
    print('\\\\')

# avg rank
print(f'\\hline')
print(f'avg rank ', end='')
ranking = df[product(METHODS, ['bp'])].rank(axis=1, ascending=True, method='average').mean()

for method, strategy in product(METHODS, ['bp']):
    print(f'& & {ranking.loc[(method, strategy)]:.2f} ', end='')

print("""
\\end{tabular}
""")
