import pandas as pd


for METHOD in ['find-rs', 'ripper', 'id3']:
    for DATASET_SIZE in [0.5, 0.3333, 0.25, 0.1]:
        for TARGET_METRIC in ['f1', 'accuracy']:

            df = pd.read_csv(f'{METHOD}-statistics-{DATASET_SIZE}-{TARGET_METRIC}.csv')
            df_b = pd.read_csv(f'multi-iteration-{METHOD}-{DATASET_SIZE}-{TARGET_METRIC}.csv')

            print("""
            \\begin{table}[ht]
            \\centering
            \\small

            \\begin{tabular}{ l | c  c c c c c c c}
            \\hline
            & %s $(N=10)$ & BO & BP & best-k \\\\ 
             \\hline
            
            \\texttt{breast}     & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{car}        & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{hiv}        & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{kp-vs-kr}   & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{lymph}      & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{monks1}     & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{monks2}     & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{monks3}     & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{mush}       & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{primary}    & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{ttt}        & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            \\texttt{vote}       & %.4f (%.4f) & %.4f & %.4f & %.4f \\\\
            
            \\hline
            \\end{tabular}

            \\caption{Base method: %s. \\label{tab:%s-%s-%s-exp1} Test \\texttt{%s-score} averaged across ten runs. Dataset size=%.2f
            }  
            \\end{table}
            
            """ % (
                METHOD,

                df[df['dataset'] == 'BREAST'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'BREAST']['std'].mean(),
                df_b[(df_b['dataset'] == 'BREAST') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'BREAST') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'BREAST') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'CAR'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'CAR']['std'].mean(),
                df_b[(df_b['dataset'] == 'CAR') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'CAR') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'CAR') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'HIV'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'HIV']['std'].mean(),
                df_b[(df_b['dataset'] == 'HIV') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'HIV') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),

                df_b[(df_b['dataset'] == 'HIV') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'KP-VS-KR'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'KP-VS-KR']['std'].mean(),
                df_b[(df_b['dataset'] == 'KP-VS-KR') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'KP-VS-KR') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'KP-VS-KR') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'LYMPH'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'LYMPH']['std'].mean(),

                df_b[(df_b['dataset'] == 'LYMPH') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'LYMPH') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'LYMPH') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MONKS1'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MONKS1']['std'].mean(),
                df_b[(df_b['dataset'] == 'MONKS1') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'MONKS1') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'MONKS1') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MONKS2'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MONKS2']['std'].mean(),
                df_b[(df_b['dataset'] == 'MONKS2') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'MONKS2') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),

                df_b[(df_b['dataset'] == 'MONKS2') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MONKS3'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MONKS3']['std'].mean(),
                df_b[(df_b['dataset'] == 'MONKS3') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'MONKS3') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),

                df_b[(df_b['dataset'] == 'MONKS3') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MUSH'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'MUSH']['std'].mean(),
                df_b[(df_b['dataset'] == 'MUSH') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'MUSH') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'MUSH') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'PRIMARY'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'PRIMARY']['std'].mean(),
                df_b[(df_b['dataset'] == 'PRIMARY') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'PRIMARY') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'PRIMARY') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'TTT'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'TTT']['std'].mean(),
                df_b[(df_b['dataset'] == 'TTT') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'TTT') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'TTT') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),
                df[df['dataset'] == 'VOTE'][TARGET_METRIC].mean(),
                df[df['dataset'] == 'VOTE']['std'].mean(),
                df_b[(df_b['dataset'] == 'VOTE') & (df_b['strategy'] == 'bo')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'VOTE') & (df_b['strategy'] == 'bp')][TARGET_METRIC].mean(),
                df_b[(df_b['dataset'] == 'VOTE') & (df_b['strategy'] == 'best-k')][TARGET_METRIC].mean(),






                METHOD,
                METHOD,
                TARGET_METRIC,
                DATASET_SIZE,
                TARGET_METRIC,
                DATASET_SIZE
            ))


# filter the dataframe df_b by dataset breast and strategy bo
# df_b[(df_b['dataset'] == 'BREAST') & (df_b['strategy'] == 'bo')]