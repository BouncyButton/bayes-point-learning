import numpy as np
from tqdm import tqdm
from wittgenstein import RIPPER

from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._discrete_or_constraint import DiscreteOrConstraint
from bpllib.rules._rule import Rule


class DummyClassifier(BayesPointClassifier):
    '''
    This class implements a dummy algorithm.
    '''

    def __init__(self, **kwargs):
        self.use_bootstrap = True
        super().__init__(**kwargs)

    def base_method(self, X, y, target_class, **kwargs):
        '''
        This method implements the base algorithm.
        '''
        # pick a random attribute / value
        attr = np.random.randint(0, X.shape[1])
        val = X[y == target_class][:, attr][0]

        rule_set = [Rule({attr: DiscreteConstraint(index=attr, value=val)})]
        return rule_set
