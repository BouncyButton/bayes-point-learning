# i used a .strip('-old') but apparently it removes a portion of the string
# so i'll fix the data by renaming strategy 'b' to strategy 'bo'.
import pandas as pd

# read the data
filename = '../newnew.pkl'
df = pd.read_pickle(filename)

# fix the data
df['strategy'] = df['strategy'].apply(lambda x: 'bo' if x == 'b' else x)

print(df)

# save the data

df.to_pickle('../newnew-fixed.pkl')
