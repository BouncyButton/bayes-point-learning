from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bpllib import FindRsClassifier, get_dataset
from matplotlib import pyplot as plt
from sklearn import datasets

X, y = get_dataset('COMPAS')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = FindRsClassifier()
import time

t = time.time()
clf.fit(X_train, y_train, verbose=6)

print(time.time() - t, "seconds")
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
