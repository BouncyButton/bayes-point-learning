from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._rule import Rule
from imodels import SlipperClassifier as SC


class SlipperClassifier(BayesPointClassifier):
    '''
    This class implements the slipper algorithm.
    '''
    description = 'BRS'

    def __init__(self, verbose=0, threshold_acc=0.99, **kwargs):
        self.use_bootstrap = True

        super().__init__(verbose=verbose, threshold_acc=threshold_acc, **kwargs)

    def base_method(self, X, y, target_class):
        # to complete
        clf = SC(**self.kwargs)
        clf.fit(X, y)
        return clf.rules_
