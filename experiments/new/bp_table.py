import math

import pandas as pd

from bpllib import utils

import sys

from experiments.new import exp_utils

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'accuracy'  # 'accuracy'
DATASETS = utils.easy_datasets + ['-'] + utils.medium_datasets + ['-'] + utils.hard_datasets
METHODS = ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS', 'SVM', 'RF', 'TabNet']  # 'AQ', 'BRS',

filename = sys.argv[2] if len(sys.argv) > 2 else '../new-results-merged-no-model.pkl'
df = exp_utils.get_df(['model'])  # pd.read_pickle(filename)
N = len(df)
print("% autogenerated by bp_table.py (N={}, filename={})".format(N, filename))

# df = df[(df['strategy'] == 'single')]
df = df[(df['strategy'] == 'bp') | ((df['strategy'] == 'single') & df['method'].isin(['SVM', 'RF', 'TabNet']))]
print("""
\\begin{tabular}{ l | l l l l l | l l}
\\hline
""")

for method in METHODS:
    print(f'& {method} ', end='')
print('\\\\ \\hline')

for dataset in DATASETS:
    if dataset == '-':
        print(f'\\hline')
        continue
    if dataset == 'LYMPHOGRAPHY':
        print(f'\\texttt{{LYMPH.}} ', end='')
    else:
        print(f'\\texttt{{{dataset}}} ', end='')

    best_method = None
    best_metric7 = float('-inf')
    best_metric5 = float('-inf')

    for method in METHODS:
        # if method in ['SVM', 'RF']:
        #    continue
        avgs = []
        stds = []
        for enc in ["av", "ohe"]:
            avg_metric = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == enc)][
                METRIC].mean()
            std_metric = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == enc)][
                METRIC].std()
            avgs.append(avg_metric)
            stds.append(std_metric)

        # set nan to 0 (damn brs)
        avgs = [0 if math.isnan(x) else x for x in avgs]
        stds = [0 if math.isnan(x) else x for x in stds]

        best_enc_idx = avgs.index(max(avgs))
        best_enc = ["av", "ohe"][best_enc_idx]
        avg_metric = avgs[best_enc_idx]
        std_metric = stds[best_enc_idx]

        if avg_metric > best_metric5 and method not in ['RF', 'SVM', 'TabNet']:
            best_method = method
            best_metric5 = avg_metric

        if avg_metric > best_metric7:
            best_method = method
            best_metric7 = avg_metric


    for method in METHODS:
        avgs = []
        stds = []
        for enc in ["av", "ohe"]:
            avg_metric = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == enc)][
                METRIC].mean()
            std_metric = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == enc)][
                METRIC].std()
            avgs.append(avg_metric)
            stds.append(std_metric)

        # set nan to 0 (damn brs)
        avgs = [0 if math.isnan(x) else x for x in avgs]
        stds = [0 if math.isnan(x) else x for x in stds]

        best_enc_idx = avgs.index(max(avgs))
        best_enc = ["av", "ohe"][best_enc_idx]
        avg_metric = avgs[best_enc_idx]
        std_metric = stds[best_enc_idx]
        marker = "" if best_enc == "av" else "$\\dagger$"

        if math.isnan(avg_metric) or avg_metric == 0:
            # put a dash
            print(f'& - ', end='')
        elif math.isnan(std_metric):
            # avoid +- nan for 1 seed
            if avg_metric == best_metric7:
                print(f'& \\textbf{{{avg_metric:.3f}}} \\small{{{marker}}} ', end='')
            else:
                print(f'& {avg_metric:.3f} \\small{{{marker}}} ', end='')
        else:

            # special formatting
            if best_metric7 == avg_metric or best_metric5 == avg_metric:
                if method not in ['RF', 'SVM', 'TabNet']:  # interpretable methods
                    if avg_metric == best_metric5:
                        if avg_metric == best_metric7:
                            print(
                                f'& \\underline{{\\textbf{{{avg_metric:.3f}}}}}\\tiny{{$\\pm${std_metric:.2f}}} \\small{{{marker}}} ',
                                end='')
                        else:
                            print(
                                f'& \\textbf{{{avg_metric:.3f}}}\\tiny{{$\\pm${std_metric:.2f}}} \\small{{{marker}}} ',
                                end='')
                else:  # uninterpretable methods
                    if avg_metric == best_metric7:
                        print(f'& \\underline{{{avg_metric:.3f}}}\\tiny{{$\\pm${std_metric:.2f}}} \\small{{{marker}}} ',
                              end='')
            # print as usual
            else:
                print(f'& {avg_metric:.3f}\\tiny{{$\\pm${std_metric:.2f}}} \\small{{{marker}}} ', end='')

    print('\\\\')

# Add average ranks

print(f'\\hline')
print(f'\\textbf{{AvgRank}} ', end='')

method_avgs = []
for method in ['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS']:
    avgs = []
    stds = []
    for dataset in DATASETS:
        if dataset == '-':
            continue
        av_avg_metric = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == 'av')][
            METRIC].mean()
        oh_avg_metric = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['encoding'] == 'ohe')][
            METRIC].mean()
        av_avg_metric = 0 if math.isnan(av_avg_metric) else av_avg_metric
        oh_avg_metric = 0 if math.isnan(oh_avg_metric) else oh_avg_metric
        avg_metric = max(av_avg_metric, oh_avg_metric)
        avgs.append(avg_metric)
    method_avgs.append(avgs)

sc = pd.DataFrame(method_avgs)

avg_ranks = (sc.T).rank(axis=1, ascending=False, method='average').mean(axis=0)

for method, avg_rank in zip(['Find-RS', 'RIPPER', 'ID3', 'AQ', 'BRS'], avg_ranks):
    print(f'& {avg_rank:.2f} ', end='')
print(' & & \\\\')

print("""
\\end{tabular}
""")
