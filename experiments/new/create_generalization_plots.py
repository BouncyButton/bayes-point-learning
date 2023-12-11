import json
import pickle
from itertools import repeat

from matplotlib import pyplot as plt

# read the data
input_file = '../find-rs-performance-bins2.pkl'

with open(input_file, 'rb') as f:
    df = pickle.load(f)

# plot the data using two plots
fig, ax = plt.subplots(figsize=(5, 10), nrows=2)

# first plot: generalization probability vs performance using a plot with std
df = df[df['method'] == 'FindRSGridSearch']

xs = []
ys = []
yerrs = []
names = []

for dataset, encoding in df[['dataset', 'encoding']].drop_duplicates().values:
    df2 = df[(df['dataset'] == dataset) & (df['encoding'] == encoding) & (df['strategy'] == 'bp')]
    if dataset not in ['AUDIO', 'VOTE', 'COMPAS', 'MONKS2']:
        continue
    # compute the mean for each generalization_probability value in the json elements
    all_performance = {k: v for k, v in zip([0, 0.25, 0.5, 0.75, 0.9, 1], [[] for _ in range(6)])}
    for i, row in df2.iterrows():
        gp = json.loads(row['avg_performance'])['generalization_probability']
        for k, v in gp.items():
            all_performance[float(k)].append(v)

    # compute the mean and std for each generalization_probability value
    mean_performance = {k: (sum(v) / len(v)) for k, v in all_performance.items()}
    std_performance = {k: (sum([(x - mean_performance[k]) ** 2 for x in v]) / len(v)) ** 0.5 for k, v in
                       all_performance.items()}
    xs.append(list(mean_performance.keys()))
    ys.append(list(mean_performance.values()))
    yerr = list(std_performance.values())
    yerrs.append(yerr)
    names.append(f'{dataset}_{encoding}')
    ax[0].errorbar(list(mean_performance.keys()), list(mean_performance.values()), yerr=yerr,
                   label=f'{dataset} ({encoding})', capsize=5, capthick=2)

ax[0].set_xlabel('Generalization probability')
ax[0].set_ylabel('Performance')
ax[0].set_title('Generalization probability vs performance')
ax[0].legend()

# output the xs, ys, and yerr to a file for visualizing in tex using tikz
# better to output three lines, one for ys-yerr, one for ys, and one for ys+yerr
# the format is \addplot[color=green] coordinates {(0,0.224)(1,3.950)(2,22.342) ... };
with open('generalization_probability_vs_performance.txt', 'w') as f:
    f.write("% created by create_generalization_plots.py\n")

    # pick two element sublists from xs, ys, yerrs, and names; they are av and oh. put it in _sub
    for i in range(0, len(xs), 2):
        xs_sub = xs[i:i + 2]
        ys_sub = ys[i:i + 2]
        yerrs_sub = yerrs[i:i + 2]
        names_sub = names[i:i + 2]

        ymin = min(min(ys_sub[0]) - max(yerrs_sub[0]), min(ys_sub[1]) - max(yerrs_sub[1]))
        ymax = max(max(ys_sub[0]) + max(yerrs_sub[0]), max(ys_sub[1]) + max(yerrs_sub[1]))

        f.write(f"""
    \\begin{{tikzpicture}}
        \\begin{{axis}}[
          xmin=0, xmax=1,
          ymin={ymin}, ymax={ymax},
          ymajorgrids=true,
          grid style=dashed,
          width = 0.45\\columnwidth,
          height = 0.35\\columnwidth,
          legend pos=north west,
          xlabel={{Generalization probability}},
            ylabel={{F1}},
          title={{{names_sub[0].replace('_', ' ').replace('av', '')}}}
        ]
        \\addlegendimage{{blue, line width=1pt}}
    \\addlegendentry{{av}}
 \\addlegendimage{{purple, line width=1pt}}
    \\addlegendentry{{oh}}
    
        """)
        for x, y, yerr, name, color in zip(xs_sub, ys_sub, yerrs_sub, names_sub, ['blue', 'purple']):
            f.write(f'\\addplot[name path={name}_down,color={color}] coordinates {{')
            for x_, y_, yerr_ in zip(x, y, yerr):
                f.write(f'({x_},{y_ - yerr_})')
            f.write('};\n')

            f.write(f'\\addplot[name path={name},color={color}!70] coordinates {{')
            for x_, y_, yerr_ in zip(x, y, yerr):
                f.write(f'({x_},{y_})')
            f.write('};\n')

            f.write(f'\\addplot[name path={name}_top,color={color}!70] coordinates {{')
            for x_, y_, yerr_ in zip(x, y, yerr):
                f.write(f'({x_},{y_ + yerr_})')
            f.write('};\n')
            f.write(f'\\addplot[{color}!50,fill opacity=0.5] fill between[of={name}_top and {name}_down];')

        f.write('\n \\legend{av, oh}')
        f.write("""\n    \end{axis}
    \end{tikzpicture}
        """)
        if i == len(xs) - 2:
            break
        if i % 4 == 0:
            f.write("\n & \n")
        else:
            f.write("\n \\\\[2em] \n")

# second plot: bin size vs freq (histogram with std)
for dataset, encoding in df[['dataset', 'encoding']].drop_duplicates().values:
    if dataset not in ['AUDIO', 'VOTE', 'COMPAS']:
        continue

    df2 = df[(df['dataset'] == dataset) & (df['encoding'] == encoding) & (df['strategy'] == 'bp')]
    N = 50
    all_size_frequency = {k: v for k, v in zip(list(range(1, N)), [[] for _ in range(N)])}
    for i, row in df2.iterrows():
        size_frequency = json.loads(row['bin_size_frequency'])
        for k, v in size_frequency.items():
            if int(k) < N:
                all_size_frequency[int(k)].append(v * int(k))

    # compute the mean and std for each bin size
    mean_size_frequency = {k: (sum(v) / len(v)) if len(v) > 0 else 0 for k, v in all_size_frequency.items()}
    std_size_frequency = {
        k: (sum([(x - mean_size_frequency[k]) ** 2 for x in v]) / len(v)) ** 0.5 if len(v) > 0 else 0 for k, v in
        all_size_frequency.items()}

    ax[1].errorbar(list(mean_size_frequency.keys()), list(mean_size_frequency.values()),
                   # yerr=list(std_size_frequency.values()),
                   label=f'{dataset} ({encoding})', capsize=5, capthick=2)
# set ax1 yscale to log
# ax[1].set_yscale('log')

ax[1].set_xlabel('Bin size')
ax[1].set_ylabel('f*|B| (how many instances go to a bin size)')
ax[1].set_title('Bin size vs frequency')
ax[1].legend()

plt.show()
