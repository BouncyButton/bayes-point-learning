from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bpllib import FindRsClassifier, get_dataset, RIPPERClassifier
from matplotlib import pyplot as plt
from sklearn import datasets

X, y = get_dataset('COMPAS')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RIPPERClassifier(T=1)
clf.fit(X_train, y_train)

y_pred_ba = clf.predict(X_test, strategy='bo')
print("ba", accuracy_score(y_test, y_pred_ba))
print(len(clf.ruleset_))

clf = RIPPERClassifier(T=20)
clf.fit(X_train, y_train)

y_pred_bo = clf.predict(X_test, strategy='bo')
y_pred_bp = clf.predict(X_test, strategy='bp')
y_pred_bk = clf.predict(X_test, strategy='best-k', n_rules=20)

print("bo", accuracy_score(y_test, y_pred_bo))
print("bp", accuracy_score(y_test, y_pred_bp))
print(len(clf.counter_))
print("bk", accuracy_score(y_test, y_pred_bk))
