# all necessary data is in find-rs-oh.pkl and multi-iteration-3-find-rs.pkl.
# also, we need to compare it to baselines-svm-rf.csv.
import pickle

import pandas as pd

test_datasets = [
    'CAR',
    'TTT',
    'MUSH',
    'MONKS1',
    'MONKS2',
    'MONKS3',
    'KR-VS-KP',
    'VOTE',
    'BREAST',
    'HIV',
    'LYMPHOGRAPHY',
    'PRIMARY'
]
test_datasets.sort()
METRIC = 'accuracy'
DATASET_SIZE = 0.5
print(f"""
\\begin{{table}}[h]
\\centering
\\small
\\begin{{tabular}}{{ l | l l l l }}
\\hline
dataset & find-rs (BO) & find-rs (BP) & svm & rf \\\\ 
 \\hline
""")

for DATASET in test_datasets:
    vals_to_print = []

    print(f"{DATASET}  & ", end="")

    for method in ['find-rs', 'svm', 'rf']:
        if method == 'find-rs':

            for strategy in ['bo', 'bp']:
                filename = f'final-dat/multi-iteration-3-find-rs.pkl'

                # read pkl file
                with open(filename, 'rb') as f:
                    df = pickle.load(f)

                rows = df[(df['dataset'] == DATASET) & (df['dataset_size'] == DATASET_SIZE)]
                rows = rows[(rows['strategy'] == strategy)]
                assert len(rows) == 1
                row = rows.iloc[0]

                metric_av = row[METRIC]

                filename = f'final-dat/find-rs-oh.pkl'

                # read pkl file
                with open(filename, 'rb') as f:
                    df = pickle.load(f)

                rows = df[(df['dataset'] == DATASET)]
                rows = rows[rows['strategy'] == strategy]
                if DATASET == 'MUSH':
                    metric_oh = 0  # av works perfectly anyway
                else:
                    assert len(rows) == 1
                    row = rows.iloc[0]
                    metric_oh = row[METRIC]
                metric = max(metric_oh, metric_av)
                vals_to_print.append(metric)

        else:
            filename = 'final-dat/baselines-svm-rf.csv'
            # read the results from the file putting it in a pandas dataframe
            df = pd.read_csv(filename)
            rows = df[(df['dataset'] == DATASET) & (df['method'] == method)]
            assert len(rows) == 10

            metric, std = rows[METRIC].mean(), rows[METRIC].std()
            vals_to_print.append((metric, std))

    assert len(vals_to_print) == 4
    # when we elaborated all the values for a single dataset, print them

    for i, x in enumerate(vals_to_print):
        if isinstance(x, tuple):
            metric, std = x
            if metric == max(vals_to_print[0], vals_to_print[1], vals_to_print[2][0], vals_to_print[3][0]):
                print(f"\\textbf{{{metric:.3f}}} $\pm$ {std:.3f}", end=" & " if i != 3 else " \\\\\n")
            else:
                print(f"{metric:.3f} $\pm$ {std:.3f}", end=" & " if i != 3 else " \\\\\n")
        else:
            metric = x
            if metric == max(vals_to_print[0], vals_to_print[1], vals_to_print[2][0], vals_to_print[3][0]):
                print(f"\\textbf{{{metric:.3f}}}", end=" & " if i != 3 else " \\\\\n")
            else:
                print(f"{metric:.3f}", end=" & " if i != 3 else " \\\\\n")

            # print(f"{metric:.3f} $\pm$ {std:.3f}", end=" & " if method != 'rf' else " \\\\\n")

print("""
\\hline
\\end{tabular}
\\caption{Comparison of the performance (%s) of the black-box baselines and our method
 (BO and BP, using as base learner \method{}).
  We considered the best performing encoding (AV or OH) for \method{}. 
  The best performing method is highlighted in \\textbf{bold}.}
\\label{tab:exp-vs-unint}
\\end{table}
""" % METRIC)
