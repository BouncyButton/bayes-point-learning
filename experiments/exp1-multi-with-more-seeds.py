import csv
import pickle
# get the latest result file
import glob
import os

import numpy as np

METHOD = 'find-rs'
filename_in = max(glob.iglob(f'more-multi-iteration-{METHOD}.pkl'), key=os.path.getctime)
# read from the pickle file
with open(filename_in, 'rb') as f:
    df = pickle.load(f)

# filter by method
df = df[df['method'] == 'FindRsClassifier']

for DATASET_SIZE in [0.3333, 0.5]:
    for TARGET_METRIC in ['f1', 'accuracy']:
        for METHOD in ['find-rs']:
            # use iterrows

            # filename = f'multi-iteration-{METHOD}{DATASET_SIZE}-{TARGET_METRIC}.csv'
            filename_out = f'more-multi-iteration-{METHOD}-{DATASET_SIZE}-{TARGET_METRIC}.csv'

            results = csv.writer(open(filename_out, 'w'))
            results.writerow(
                ['dataset', 'method', 'strategy', TARGET_METRIC, 'std', 'avg_ruleset_len', 'avg_rule_len'])

            # save a csv with average and std for each dataset
            df_2 = df.groupby(['dataset', 'method']).agg(
                {TARGET_METRIC: ['mean', 'std'], 'dataset': ['min'], 'method': ['min'], 'strategy': ['min'],
                 # 'avg_rule_len': ['mean'], 'avg_ruleset_len': ['mean']
                 }).copy()

            for index, row in df_2.iterrows():
                if row['dataset_size'] == DATASET_SIZE:
                    # avg_ruleset_len = np.array([len(ruleset) for ruleset in row['model'].rulesets_]).mean()
                    # avg_rule_len = np.array(
                    #    [len(rule) for ruleset in row['model'].rulesets_ for rule in ruleset]).mean()

                    assert len(row['method']) <= 1
                    assert len(row['strategy']) <= 1

                    results.writerow(
                        [row['dataset']['min'], METHOD, row['strategy']['min'],
                         row[TARGET_METRIC]['mean'], row[TARGET_METRIC]['std'],
                         0,  # avg_ruleset_len,
                         0,  # avg_rule_len])
                         ])
