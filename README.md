# Bayes point Rule set learning

[![codecov](https://codecov.io/gh/BouncyButton/bayes-point-learning/branch/main/graph/badge.svg?token=202TOBNEVQ)](https://codecov.io/gh/BouncyButton/bayes-point-learning)

This repository contains a scikit-learn compatible implementation of the Bayes point learning algorithm described in the
paper [Bayes Point Rule Set Learning](https://arxiv.org/abs/2204.05251). The algorithm is implemented in
the `BPLClassifier` class in `bpl.py`.

## Installation

To install the package, run

    pip install git+https://github.com/BouncyButton/bayes-point-learning.git

For Cython development, install https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/ on Windows.


## Usage

The `BplClassifier` class is a scikit-learn compatible classifier. It can be used in the same way as any other classifier.
For example, to train a BPL classifier on the iris dataset, run

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from bpllib import BplClassifier

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = BplClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

## Citation

If you use this code in your research, please cite the paper [Bayes Point Rule Set Learning](https://arxiv.org/abs/2204.05251).

    @article{Aiolli2022BayesPR,
      title={Bayes Point Rule Set Learning},
      author={Fabio Aiolli and Luca Bergamin and Tommaso Carraro and Mirko Polato},
      journal={ArXiv},
      year={2022},
      volume={abs/2204.05251}
    }

## License

This code is licensed under the MIT license. See the LICENSE file for details.

## Contact

For questions or comments, please contact Luca Bergamin. You can also open an issue on GitHub. 
