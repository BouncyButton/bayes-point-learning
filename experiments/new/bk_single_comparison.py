import math
from itertools import product

import pandas as pd

from bpllib import utils

import sys

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'accuracy'  # 'accuracy'
DATASETS = utils.easy_datasets + ['-'] + utils.medium_datasets  # + ['-'] + utils.hard_datasets
METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']

filename = sys.argv[2] if len(sys.argv) > 2 else '../merged.pkl'
df = pd.read_pickle(filename)
N = len(df)
print("% autogenerated by bk_single_comparison.py (N={}, filename={})".format(N, filename))

df = df[(df['strategy'] == 'best-k') | (df['strategy'] == 'single')]

# group by dataset, method, encoding
df = df.groupby(['dataset', 'method', 'strategy', 'encoding']).agg({METRIC: ['mean']}).reset_index()

df = df.pivot_table(index=['dataset', 'method'], columns=['strategy', 'encoding'], values=METRIC).reset_index()

# extract for each dataset/method the best encoding for the strategy bp
df['best_encoding'] = df.apply(
    lambda row: 'av' if row[('mean', 'single', 'av')] >= row[('mean', 'single', 'ohe')] else 'ohe',
    axis=1)
best_encoding_map = {(dataset, method): best_encoding for dataset, method, best_encoding in
                     df[['dataset', 'method', 'best_encoding']].values}

df['single'] = df.apply(lambda row: row[('mean', 'single', row[('best_encoding', '', '')])], axis=1)
df['best-k'] = df.apply(lambda row: row[('mean', 'best-k', row[('best_encoding', '', '')])], axis=1)

df = df.pivot_table(index=['dataset'], columns=['method'], values=['single', 'best-k'])  # .reset_index()

df = df.droplevel((1, 2), axis=1).swaplevel(0, axis=1).reorder_levels(['method', None], axis=1).sort_index(axis=1)

# print a two level latex table

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
    print(f'& base & pruned ', end='')

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
            best_k = df[(method, 'best-k')].loc[dataset]
            single = df[(method, 'single')].loc[dataset]
        except KeyError:
            print(f'& - & - ', end='')
            continue

        best_enc = best_encoding_map.get((dataset, method), 'av')
        marker = "" if best_enc == "av" else "$\\dagger$"

        if math.isnan(best_k) and math.isnan(single):
            print(f'& - & - ', end='')
        elif single > best_k:
            print(f'& \\textbf{{{single:.3f}}} & {best_k:.3f} \\small{{{marker}}}', end='')
        elif single < best_k:
            print(f'& {single:.3f} & \\textbf{{{best_k:.3f}}} \\small{{{marker}}}', end='')
        else:
            print(f'& \\textbf{{{single:.3f}}} & \\textbf{{{best_k:.3f}}} \\small{{{marker}}}', end='')
    print('\\\\')

# avg rank
print(f'\\hline')
print(f'\\textbf{{AvgRank}} ', end='')
ranking = df.rank(axis=1, ascending=False, method='average').mean()

for method, strategy in product(METHODS, ['single', 'best-k']):
    print(f'& {ranking.loc[(method, strategy)]:.2f} ', end='')

print("""
\\end{tabular}
""")