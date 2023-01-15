import sys

# filename = sys.argv[1]

# get the latest result file
import glob
import os


TARGET_METRIC = 'f1'
filename = max(glob.iglob('results_*.csv'), key=os.path.getctime)

# read the results from the file putting it in a pandas dataframe
import pandas as pd

df = pd.read_csv(filename)

# plot the results
import matplotlib.pyplot as plt

for dataset in df['dataset'].unique():
    for method in df['method'].unique():
        df2 = df[df['dataset'] == dataset].copy()
        df2 = df2[df2['method'] == method].copy()
        # calculate the mean and std for each dataset
        df2 = df2.groupby(['strategy', 'dataset_size']).agg(
            {TARGET_METRIC: ['mean', 'std'], 'strategy': ['min'], 'dataset_size': ['min']}).copy()

        # get all strategies
        strategies = df2['strategy']['min'].unique()

        for strategy in strategies:
            dataset_sizes = []
            f1s = []
            stds = []
            for row in df2[df2['strategy']['min'] == strategy].iterrows():
                dataset_size = row[0][1]
                dataset_sizes.append(dataset_size)
                f1 = row[1][TARGET_METRIC]['mean']
                f1s.append(f1)
                std = row[1][TARGET_METRIC]['std']
                stds.append(std)
            plt.errorbar(dataset_sizes, f1s, yerr=stds, label=strategy, alpha=0.5)
        plt.legend()
        plt.title(str(dataset) + str(method))
        plt.xlabel('Dataset size')
        plt.ylabel(TARGET_METRIC)
        plt.show()
        # plt.plot(df2['dataset_size'], df2['f1'], label=dataset)
