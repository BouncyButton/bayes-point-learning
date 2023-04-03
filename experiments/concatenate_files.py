import pickle
# get the latest result file
import glob
import os

TARGET_METRIC = 'f1'
filename = 'kp-vs-kr-aq-id3.pkl'  # 'multi-iteration-3-id3.pkl' #max(glob.iglob('results_*.pkl'), key=os.path.getctime)

# read from the pickle file
with open(filename, 'rb') as f:
    df = pickle.load(f)

print()

filename2 = 'multi-iteration-3-aq.pkl'
# read from the pickle file
with open(filename2, 'rb') as f:
    df_out = pickle.load(f)

row = df[(df['method'] == 'AqClassifier')]
assert len(row) == 3
row0 = row.iloc[0]
row1 = row.iloc[1]
row2 = row.iloc[2]

df_out = df_out.append(row0, ignore_index=True)
df_out = df_out.append(row1, ignore_index=True)
df_out = df_out.append(row2, ignore_index=True)

# write to new pickle file
with open('out-' + filename2, 'wb') as f:
    pickle.dump(df_out, f)
