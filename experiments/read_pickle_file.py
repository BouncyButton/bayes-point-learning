import pickle
# get the latest result file
import glob
import os

TARGET_METRIC = 'f1'
filename = 'results_22-01-2023_15-09-50.pkl'  #'kp-vs-kr-aq-id3.pkl'  # 'multi-iteration-3-id3.pkl' #max(glob.iglob('results_*.pkl'), key=os.path.getctime)

# read from the pickle file
with open(filename, 'rb') as f:
    df = pickle.load(f)

print()
