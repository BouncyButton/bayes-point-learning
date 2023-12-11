import json
import math

import numpy as np
import pandas as pd

from bpllib.utils import easy_datasets, medium_datasets, hard_datasets

METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']
DATASETS = hard_datasets  # easy_datasets + medium_datasets + hard_datasets
df = pd.read_csv('../images/best-k/bestk_results-hard.csv')

for method in METHODS:
    for dataset in DATASETS:
        encoding = 'ohe' if dataset in ['BREAST', 'CAR', 'HIV', 'MONKS2', 'MONKS3', 'VOTE', 'ADULT', 'MARKET',
                                        'CONNECT-4'] else 'av'
        for size in ['up', 'avg', 'down']:
            # create a txt file
            with open(f'../data/best-k/{dataset.replace("-", "")}_{method.replace("-", "")}_{size}.txt', 'w') as f:
                # output 'x y'
                f.write('x y\n')

                # create a list of values in a log scale, n=100
                log_range = np.logspace(np.log10(1), np.log10(250), num=100, base=10.0)
                # make int and filter out duplicates
                log_range = sorted(list(set([int(x) for x in log_range])))

                # output x y pairs
                for n_rules in log_range:
                    rows = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == encoding)]
                    if len(rows) == 0:
                        f.write(f'{n_rules:.1f} 0.0\r\n')
                    else:
                        row = rows.iloc[0]
                        val = json.loads(row['y'])[n_rules - 1]

                        if val == 'nan' or math.isnan(val):
                            val = 0.0

                        elif size == 'up':
                            val += json.loads(row['y_std'])[n_rules - 1]
                        elif size == 'down':
                            val -= json.loads(row['y_std'])[n_rules - 1]
                        else:
                            pass

                        f.write(f'{n_rules:.1f} {val}\r\n')
