import pandas as pd

METHOD = 'FindRsClassifier'
DATASET_SIZE = 0.3333
TARGET_METRIC = 'f1'
filename_in = f'{METHOD}-statistics-{DATASET_SIZE}-{TARGET_METRIC}.csv'
df = pd.read_csv(filename_in)

print("""
\\begin{table*}[ht]
\\centering
\\small
\\resizebox{1.6\columnwidth}{!}{
\\begin{tabular}{ l | c c c c c  }
\\hline
Dataset & \\method & complexity & BO & complexity & BP & complexity \\\\ 
 \\hline
""")
for dataset in df['dataset'].unique():
    print(f"{dataset} & {METHOD} & ", end='')
    print(
        f"{df[(df['dataset'] == dataset) & (df['method'] == METHOD) & (df['strategy'] == 'bo')]['avg_ruleset_len'].mean():.2f} & ",
        end='')
    print(
        f"{df[(df['dataset'] == dataset) & (df['method'] == METHOD) & (df['strategy'] == 'bo')]['avg_rule_len'].mean():.2f} & ",
        end='')
    print(
        f"{df[(df['dataset'] == dataset) & (df['method'] == METHOD) & (df['strategy'] == 'bp')]['avg_ruleset_len'].mean():.2f} & ",
        end='')
    print(
        f"{df[(df['dataset'] == dataset) & (df['method'] == METHOD) & (df['strategy'] == 'bp')]['avg_rule_len'].mean():.2f} \\\\ ",
        end='')
    print()
print("""
\\hline
\\end{tabular}
}
      """)
