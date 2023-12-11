import sqlite3


def get_df(to_remove=None, datasets=None):
    if to_remove is None:
        to_remove = []
    if datasets is None:
        datasets = []
    conn = sqlite3.connect('../results.sqlite')
    cursor = conn.cursor()

    columns = ['dataset', 'method', 'seed', 'encoding', 'strategy', 'T', 'model', 'accuracy', 'f1', 'dataset_size',
               'avg_rule_len', 'avg_ruleset_len', 'avg_performance', 'bin_size_frequency', 'extra']

    for col in to_remove:
        columns.remove(col)

    if len(datasets) == 0:
        cursor.execute("SELECT %s FROM results" % (','.join(columns)))

    else:
        cursor.execute(
            "SELECT %s FROM results WHERE dataset IN ('%s')" % (','.join(columns), "','".join(datasets)))
    rows = cursor.fetchall()

    import pandas as pd

    df = pd.DataFrame(rows,
                      columns=columns)

    # decompress the model
    import gzip
    import pickle

    def decompress_model(row):
        if row.get('model') is None:
            return None
        model = gzip.decompress(row['model'])
        model = pickle.loads(model)
        return model

    df['model'] = df.apply(decompress_model, axis=1)

    return df


if __name__ == '__main__':
    get_df()
