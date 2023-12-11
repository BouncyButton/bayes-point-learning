import gzip
import math
import os
import pickle

from tqdm import tqdm

# i merged new-results.pkl -> new-datasets-merged.pkl -> hard.pkl.
# i got 5789 instead of 5987 :( but i think that's because i removed the CERV dataset
# i checked with new-results-merged-no-model.pkl, that is used to produce the tables
# i got some diffs in the results, market/find-rs has different results for seed 0
# ->  spect/aq misses seed=9, => that's because i wasnt committing
# (but are better as f1!)
# also, i found that brs produced some nans in monks3 and soybean. not in every run though.

# i merged fixed-ohe-find-rs.pkl.

filename = 'fixed-ohe-find-rs.pkl'

# check if file exists
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        df = pickle.load(f)
else:
    raise FileNotFoundError(f'File {filename} not found ({os.getcwd()})')

import sqlite3
import pickle

# Connect to the SQLite database
conn = sqlite3.connect('results.sqlite')
cursor = conn.cursor()

insert = True
check_duplicates = False
delete_existing = True

cursor.execute("CREATE TABLE IF NOT EXISTS "
               "results ("
               "dataset TEXT,"
               "method TEXT,"
               "seed INTEGER,"
               "encoding TEXT,"
               "strategy TEXT,"
               "T INTEGER,"
               "model BLOB,"
               "accuracy REAL,"
               "f1 REAL,"
               "dataset_size REAL,"
               "avg_rule_len REAL,"
               "avg_ruleset_len REAL,"
               "avg_performance REAL,"
               "bin_size_frequency REAL,"
               "extra BLOB,"
               "PRIMARY KEY (dataset, method, seed, encoding, strategy, T)"
               ")")

try:
    for i, row in tqdm(df.iterrows()):
        col_to_drop = list(
            {'dataset', 'method', 'seed', 'encoding', 'strategy', 'model', 'T', 'f1', 'accuracy', 'dataset_size',
             'avg_rule_len', 'avg_ruleset_len', 'avg_performance', 'bin_size_frequency'}.intersection(set(df.columns)))
        extra = row.drop(
            col_to_drop
        ).to_dict()

        # check if the row already exists
        cursor.execute(
            'SELECT * FROM results WHERE dataset=? AND method=? AND seed=? AND encoding=? AND strategy=? AND T=?',
            (row['dataset'], row['method'], row['seed'], row['encoding'], row['strategy'], row['T']))
        result = cursor.fetchone()
        if result is not None:
            # get first row and check accuracy
            if check_duplicates:
                if result[7] == row['accuracy']:
                    pass
                if result[7] is None and math.isnan(row['accuracy']):
                    print(f'nan values: {[r for r in result[:6]]}')

                else:
                    print(f"Accuracy mismatch: {result[7]} != {row['accuracy']}, f1: {result[8]} != {row['f1']} "
                          f"\n{[row['dataset'], row['method'], row['seed'], row['encoding'], row['strategy'], row['T']]}, \n{[r for r in result[:6]]}")

            if not delete_existing:
                continue

        if row['dataset'] == 'CERV':
            continue

        print(f"inserting dataset=%s method=%s seed=%d encoding=%s strategy=%s T=%d" % (
            row['dataset'], row['method'], row['seed'], row['encoding'], row['strategy'], row['T']))

        if row.get('model') is not None:
            model = gzip.compress(pickle.dumps(row['model']))
        else:
            model = None

        if delete_existing:
            cursor.execute(
                'DELETE FROM results WHERE dataset=? AND method=? AND seed=? AND encoding=? AND strategy=? AND T=?',
                (row['dataset'], row['method'], row['seed'], row['encoding'], row['strategy'], row['T']))

        if insert:
            cursor.execute(
                'INSERT INTO results (dataset, method, seed, encoding, strategy, T, model, accuracy, f1, dataset_size, avg_rule_len, avg_ruleset_len, avg_performance, bin_size_frequency, extra)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (row['dataset'], row['method'], row['seed'], row['encoding'], row['strategy'], row['T'],
                 model if row['method'] in ['TabNet', 'SVM', 'RF'] or row['strategy'] == 'bp' else None,
                 row['accuracy'],
                 row['f1'],
                 row['dataset_size'],
                 row.get('avg_rule_len'),
                 row.get('avg_ruleset_len'),
                 row.get('avg_performance'),
                 row.get('bin_size_frequency'),
                 pickle.dumps(extra),
                 )
            )
        if i % 10 == 0:
            conn.commit()
    conn.commit()
except Exception as e:
    print(e)
    raise e
finally:
    conn.close()
