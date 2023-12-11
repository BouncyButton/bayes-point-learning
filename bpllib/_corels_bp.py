import pandas as pd

from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._rule import Rule
from corels import CorelsClassifier as BaseCorelsClassifier


class CorelsClassifier(BayesPointClassifier):
    '''
    This class implements the CORELS algorithm.
    '''

    def __init__(self, n_iter=10000, max_card=2, c=0.8, **kwargs):
        self.use_bootstrap = True
        self.max_card = max_card
        self.c = c
        self.n_iter = n_iter
        super().__init__(**kwargs)

    # TODO should implement this method to encode data in one-hot encoding
    # def checks_for_base_method(self, X, y):
    #     pass

    def base_method(self, X, y, target_class, **kwargs):
        '''
        This method implements the base algorithm.
        '''

        clf = BaseCorelsClassifier(n_iter=self.n_iter, max_card=self.max_card, c=self.c).fit(X, y)
        # TODO adapt the rulelist to our (ruleset) format
        return clf.rl()
