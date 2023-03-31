import pandas as pd

for METHOD in ['find-rs', 'ripper', 'id3', 'aq']:
    for DATASET_SIZE in [0.5, 0.33, 0.25]:
        for TARGET_METRIC in ['f1', 'accuracy']:
            df = pd.read_csv(f'{METHOD}-statistics-2-{DATASET_SIZE}-{TARGET_METRIC}.csv')
            df_b = pd.read_csv(f'multi-iteration-3-{METHOD}-{DATASET_SIZE}-{TARGET_METRIC}.csv')

            print("""
            \\begin{table}[ht]
            \\centering
            \\small

            \\begin{tabular}{ l | c  c c c c c c c}
            \\hline
            & %s $(N=10)$ & BO & BP & best-k \\\\ 
             \\hline
""" % (METHOD,))
            for dataset in sorted(list(df['dataset'].unique())):
                print("\\texttt{%s}" % (dataset if dataset != 'LYMPHOGRAPHY' else 'LYMPH'), end=' & ')
                baseline = df[(df['dataset'] == dataset) & (df['strategy'] == 'baseline')][TARGET_METRIC].mean()
                baseline_std = df[(df['dataset'] == dataset) & (df['strategy'] == 'baseline')]['std'].mean()
                bo = df_b[(df_b['dataset'] == dataset) & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean()
                bp = df_b[(df_b['dataset'] == dataset) & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean()
                best_k = df_b[(df_b['dataset'] == dataset) & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean()

                def write_val(val, max_val, last):
                    if val == max_val:
                        print("\\textbf{%.4f}" % val, end=' & ' if not last else '\\\\\n')
                    else:
                        print("%.4f" % val, end=' & ' if not last else '\\\\\n')

                def write_val_std(val, std, max_val, last):
                    if val == max_val:
                        print("\\textbf{%.4f (%.4f)}" % (val, std), end=' & ' if not last else '\\\\\n')
                    else:
                        print("%.4f (%.4f)" % (val, std), end=' & ' if not last else '\\\\\n')

                max_val = max(baseline, bo, bp, best_k)
                i = 0

                for val in [baseline, bo, bp, best_k]:
                    if i == 0:
                        write_val_std(val, baseline_std, max_val, False)
                    else:
                        write_val(val, max_val, i == 3)
                    i += 1
            print("""
                        \\hline
                        \\end{tabular}
            
                        \\caption{Base method: %s. \\label{tab:%s-%s-%s-exp1} Test \\texttt{%s-score} averaged across ten runs. Dataset size=%.2f
                        }
                        \\end{table}
            
                        """ % (
                            METHOD,
                            METHOD,
                            TARGET_METRIC,
                            DATASET_SIZE,
                            TARGET_METRIC,
                            DATASET_SIZE
                        ))

# filter the dataframe df_b by dataset breast and strategy bo
# df_b[(df_b['dataset'] == 'BREAST') & (df_b['strategy'] == 'bo')]