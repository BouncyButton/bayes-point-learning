import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from bpllib import get_dataset, BplClassifierV5

test_datasets = [ #'CAR',
    # 'TTT',
    'CONNECT-4',
    # 'MUSH',
    #'MONKS1',
    #'MONKS2',
    #'MONKS3',
    # 'KR-VS-KP',
    #'VOTE'
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
            est = BplClassifierV5(T=3, strategy=strategy, tol=tol)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
            est.fit(X_train, y_train, pool_size=1)
            y_pred = est.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            print(name, f1)
            f1s.append(f1)
        print("time elapsed:", time.time() - start)
        print(np.mean(f1s), "+-", np.std(f1s))


if __name__ == '__main__':
    print('==== BO ====')
    template_estimator(data(), 'bo')
    print('==== BP ====')
    template_estimator(data(), 'bp')
