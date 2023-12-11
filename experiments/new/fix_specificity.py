# connect to sqlite db
import gzip
import pickle
import sqlite3

from sklearn.model_selection import train_test_split

from bpllib import get_dataset
from bpllib.utils import easy_datasets, medium_datasets, hard_datasets

conn = sqlite3.connect('../results.sqlite')
cursor = conn.cursor()

MAX_RULES = 50

ENCODING = 'ohe'
METRIC = 'precision'  # precision
skip_existing = True

METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']  # 'AQ', 'BRS',
for dataset in easy_datasets + medium_datasets + hard_datasets:
    max_model_len = 0
    encoding = ENCODING

    for method in METHODS:
        avg_metric = {m: [] for m in METHODS}
        std_metric = {m: [] for m in METHODS}

        metrics = []
        for seed in range(10):
            if method == 'BRS':
                encoding = 'ohe'

            X, y = get_dataset(dataset)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                random_state=seed, shuffle=True)

            print('retrieving model for', dataset, method, encoding, seed)

            result = cursor.execute(
                f"SELECT * FROM results WHERE "
                f"dataset='{dataset}' AND method='{method}' AND encoding='{encoding}' AND seed={seed} AND strategy='bp'").fetchone()

            # reverse this gzip.compress(pickle.dumps(row['model']))
            if result is None:
                print('[!] row is none! :(')
                continue

            if result[6] is None:
                print('[!] model is none! :(')
                continue

            model = gzip.decompress(result[6])
            model = pickle.loads(model)

            if model.counter_ is None:
                print('[!] counter is none! :(')
                continue

            model.simplify_onehot_encoded_rulesets()
