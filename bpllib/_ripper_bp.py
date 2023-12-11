import numpy as np
import pandas as pd
from tqdm import tqdm
from wittgenstein import RIPPER

from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._discrete_or_constraint import DiscreteOrConstraint
from bpllib.rules._rule import Rule


class RipperClassifier(BayesPointClassifier):
    '''
    This class implements the RIPPER algorithm.
    '''
    description = 'RIPPER'

    def __init__(self, T=3, verbose=0, threshold_acc=0.99, target_class=None, pool_size='auto', find_best_k=True,
                 random_state=None, encoding='av', to_string=True,
                 max_rules=None, bp_verbose=0):
        super().__init__(T=T,
                         verbose=verbose,
                         bp_verbose=bp_verbose,
                         threshold_acc=threshold_acc,
                         target_class=target_class,
                         pool_size=pool_size,
                         find_best_k=find_best_k,
                         random_state=random_state,
                         encoding=encoding,
                         to_string=to_string,
                         max_rules=max_rules
                         )
        # self.use_bootstrap = True  # probably not needed?

    def base_method(self, X, y, target_class):
        '''
        This method implements the base algorithm.
        '''
        clf = RIPPER(random_state=self.random_state)
        # build a dataframe with the target class as the last column
        df = pd.DataFrame(X)
        df['class'] = y

        clf.fit(df, class_feat='class', pos_class=target_class)

        ruleset = clf.ruleset_
        our_ruleset = []
        # convert the ruleset found in our format
        for rule in ruleset:
            constraints = dict()
            for cond in rule.conds:
                constraints[cond.feature] = DiscreteConstraint(index=cond.feature, value=cond.val)
            our_rule = Rule(constraints)
            our_ruleset.append(our_rule)
        return our_ruleset
