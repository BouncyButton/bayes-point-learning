import math
from itertools import product

import pandas as pd

from bpllib import utils

import sys

table = sys.argv[1] if len(sys.argv) > 1 else 'complexity'  #
METRIC = 'accuracy'  # sys.argv[1] if len(sys.argv) > 1 else 'accuracy'  # 'accuracy'
DATASETS = utils.easy_datasets + ['-'] + utils.medium_datasets + ['-'] + utils.hard_datasets
METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']

filename = '../merged.pkl'
df = pd.read_pickle(filename)
N = len(df)

df = df[(df['strategy'] == 'bp') | (df['strategy'] == 'best-k') | (df['strategy'] == 'single')]

# group by dataset, method, encoding
df = df.groupby(['dataset', 'method', 'strategy', 'encoding']).agg(
    {'avg_ruleset_len': ['mean'], 'avg_rule_len': ['mean'], METRIC: ['mean']}).reset_index()

df = df.pivot_table(index=['dataset', 'method'], columns=['strategy', 'encoding'],
                    values=[METRIC, 'avg_ruleset_len', 'avg_rule_len']).reset_index()

# extract for each dataset/method the best encoding for the strategy bp
df['best_encoding'] = df.apply(
    lambda row: 'av' if row[(METRIC, 'mean', 'best-k', 'av')] >= row[(METRIC, 'mean', 'best-k', 'ohe')] else 'ohe',
    axis=1)

df[('BP', 'compl.', '', '')] = df.apply(
    lambda row: row[('avg_ruleset_len', 'mean', 'bp', row[('best_encoding', '', '', '')])], axis=1)
df[('BP', 'spec.', '', '')] = df.apply(
    lambda row: row[('avg_rule_len', 'mean', 'bp', row[('best_encoding', '', '', '')])], axis=1)
df[('best-k', 'compl.', '', '')] = df.apply(
    lambda row: row[('avg_ruleset_len', 'mean', 'best-k', row[('best_encoding', '', '', '')])], axis=1)
df[('best-k', 'spec.', '', '')] = df.apply(
    lambda row: row[('avg_rule_len', 'mean', 'best-k', row[('best_encoding', '', '', '')])], axis=1)
df[(' base', 'compl.', '', '')] = df.apply(
    lambda row: row[('avg_ruleset_len', 'mean', 'single', row[('best_encoding', '', '', '')])], axis=1)
df[(' base', 'spec.', '', '')] = df.apply(
    lambda row: row[('avg_rule_len', 'mean', 'single', row[('best_encoding', '', '', '')])], axis=1)

df = df.pivot_table(index=['dataset'], columns=['method'], values=['BP', 'best-k', ' base'])  # .reset_index()

df = df.droplevel((2, 3), axis=1).swaplevel(0, axis=1).swaplevel(1, axis=1)
df.columns = df.columns.rename("strategy", level=1)
df.columns = df.columns.rename("metric", level=2)
df = df.reorder_levels(['method', "strategy", 'metric'], axis=1).sort_index(axis=1)
print(df)

print("% autogenerated by complexity.py")

# select only metric='spec.'
df1 = df.xs('spec.', level='metric', axis=1)

df2 = df.xs('compl.', level='metric', axis=1)

if table == 'complexity':
    # print a two level latex table
    print("% autogenerated by complexity.py (compl) (N={}, filename={})".format(N, filename))

    print(df1.to_latex(float_format="%.1f", multicolumn_format='l', multirow=True).replace('NaN', '-'))
elif table == 'specificity':
    print("% autogenerated by complexity.py (spec) (N={}, filename={})".format(N, filename))

    print(df2.to_latex(float_format="%.1f", multicolumn_format='l', multirow=True).replace('NaN', '-'))
else:
    raise ValueError("table must be 'complexity' or 'specificity'")
