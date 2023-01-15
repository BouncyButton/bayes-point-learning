from bpllib import FindRsClassifier
from matplotlib import pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=1000, noise=0.3)
X_test, y_test = datasets.make_moons(n_samples=200, noise=0.3)

clf = FindRsClassifier()
clf.fit(X, y)

y_pred = clf.predict(X_test)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)

plt.show()
