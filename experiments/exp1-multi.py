import csv
import pickle
# get the latest result file
import glob
import os

import numpy as np

METHOD = 'id3'
BASE_NAME = 'multi-iteration-3'
filename_in = 'kp-vs-kr-aq-id3.pkl' #{BASE_NAME}-{METHOD}.pkl'), key=os.path.getctime)
# read from the pickle file
with open(filename_in, 'rb') as f:
    df = pickle.load(f)

for DATASET_SIZE in [0.25, 0.33, 0.5]:
    for TARGET_METRIC in ['f1', 'accuracy']:
        for METHOD in [METHOD]:
            # use iterrows

            # filename = f'multi-iteration-{METHOD}{DATASET_SIZE}-{TARGET_METRIC}.csv'
            filename_out = f'{BASE_NAME}-{METHOD}-{DATASET_SIZE}-{TARGET_METRIC}.csv'


            results = csv.writer(open(filename_out, 'w'))
            results.writerow(
                ['dataset', 'method', 'strategy', TARGET_METRIC, 'std', 'avg_ruleset_len', 'avg_rule_len'])
            for index, row in df.iterrows():
                if row['dataset_size'] == DATASET_SIZE:
                    avg_ruleset_len = np.array([len(ruleset) for ruleset in row['model'].rulesets_]).mean()
                    avg_rule_len = np.array(
                        [len(rule) for ruleset in row['model'].rulesets_ for rule in ruleset]).mean()

                    row = [row['dataset'], row['method'], row['strategy'], row[TARGET_METRIC], 0, avg_ruleset_len,
                         avg_rule_len]
                    results.writerow(row
                        )
                    print(row)
