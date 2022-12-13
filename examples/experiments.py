import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from bpllib import get_dataset, BplClassifierOptimization, BplClassifier, BplClassifierSplit

test_datasets = [  # 'CAR',
     'TTT',
    #'CONNECT-4',
    # 'MUSH',
    # 'MONKS1',
    # 'MONKS2',
    # 'MONKS3',
    # 'KR-VS-KP',
    # 'VOTE'
]


def data():
    return [(name, get_dataset(name)) for name in test_datasets]


def template_estimator(data, strategy='bp'):
    for name, (X, y) in data:
        f1s = []
        import time
        start = time.time()
        for i in range(5):
            if name == 'MONKS2':
                enc = OneHotEncoder(handle_unknown='ignore')
                X = enc.fit_transform(X).toarray().astype(int)
                tol = 1
            else:
                tol = 0
            est = BplClassifierSplit(T=1, strategy=None, tol=tol)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
            #enc = OneHotEncoder(handle_unknown='ignore')
            #X_train = enc.fit_transform(X_train).toarray().astype(int)
            #X_test = enc.transform(X_test).toarray().astype(int)
            est.fit(X_train, y_train, pool_size=1)
            y_pred = est.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            print(name, f1)
            print("tn,fp,fn,tp:")
            print(confusion_matrix(y_test, y_pred))
            f1s.append(f1)
        print("time elapsed:", time.time() - start)
        print(np.mean(f1s), "+-", np.std(f1s))


def optimization():
    est = BplClassifierOptimization(T=1, strategy=None, tol=0)
    X_train = np.array([['a', 'Asmall', 0], ['a', 'Bmedium', 1], ['b', 'Asmall', 1], ['c', 'Clarge', 1],  # pos
                        ['a', 'Clarge', 1], ['a', 'Clarge', 0], ['b', 'Bmedium', 1], ['b', 'Bmedium', 1],  # neg
                        ['c', 'Clarge', 0]])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    est.fit(X_train, y_train, pool_size=1)
    y_train_pred = est.predict(X_train)
    print(confusion_matrix(y_train, y_train_pred))


if __name__ == '__main__':
    print('==== BO ====')
    template_estimator(data(), 'bo')
    print('==== BP ====')
    template_estimator(data(), 'bp')
