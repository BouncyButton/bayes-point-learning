import csv
import glob
import os

import pandas as pd

METHOD = 'find-rs'

for DATASET_SIZE in [0.25, 0.33, 0.5]:
    for TARGET_METRIC in ['f1']:
        for METHOD in [METHOD]:  # , 'ripper', 'id3', 'aq']:
            filename_in = f'baseline-runtime-{METHOD}.csv'
            # filename_in = max(glob.iglob('results_*.csv'), key=os.path.getctime)

            filename_out = f'{METHOD}-statistics-runtime-{DATASET_SIZE}-{TARGET_METRIC}.csv'

            # read the results from the file putting it in a pandas dataframe
            df = pd.read_csv(filename_in)

            df = df[df['dataset_size'] == DATASET_SIZE]

            # save a csv with average and std for each dataset
            df = df.groupby(['dataset', 'method', 'optimization']).agg(
                {TARGET_METRIC: ['mean', 'std'], 'dataset': ['min'], 'method': ['min'], 'strategy': ['min'],
                 'time_elapsed': ['mean', 'std'], 'optimization': ['min']}).copy()

            # write statistics in new csv file
            results = csv.writer(open(filename_out, 'w'))
            results.writerow(['dataset', 'method', 'strategy', 'optimization', 'time_elapsed', 'std'])

            # use iterrows
            for index, row in df.iterrows():
                results.writerow(
                    [row['dataset']['min'], row['method']['min'], 'baseline', row['optimization']['min'],
                     row['time_elapsed']['mean'], row['time_elapsed']['std']])
