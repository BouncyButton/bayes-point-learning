# compute the similarity between the rulesets produced by the T runs of each algorithm
# then, compute the average similarity between each different run of the same algorithm-dataset-etc.
import pickle
from itertools import combinations

import numpy as np
from sklearn.model_selection import GridSearchCV

filename = '../find-rs-cv4.pkl'

# read from the pickle file
with open(filename, 'rb') as f:
    df = pickle.load(f)


def intersection_over_union(rs1, rs2):
    return len(set(rs1).intersection(set(rs2))) / len(set(rs1).union(set(rs2)))


def compute_intersimilarity(model):
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    avg_iou = np.mean([intersection_over_union(rs1, rs2) for rs1, rs2 in combinations(model.rule_sets_, 2)])
    return avg_iou


df = df.groupby(['dataset', 'method', 'encoding', 'seed']).agg({'model': 'first'}).reset_index()

df['similarity'] = df['model'].apply(compute_intersimilarity)
df = df.groupby(['dataset', 'method', 'encoding']).agg({'similarity': ['mean', 'std']}).reset_index()

df['similarity (mean $\\pm$ std)'] = df.apply(
    lambda row: f'{row[("similarity", "mean")]:.4f} $\pm$ {row[("similarity", "std")]:.4f}', axis=1)
df = df.drop(columns=[('similarity', 'mean'), ('similarity', 'std')])

df = df.pivot(index=['dataset', 'encoding'], columns='method')

print(df.to_latex(float_format='%.4f', na_rep='-', escape=False))
