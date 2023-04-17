import numpy as np

from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._ordinal_constraint import OrdinalConstraint
from bpllib.rules._rule import Rule


class RuleByExample(Rule):
    '''
    It is useful to instance a rule starting from a given input data point.
    '''

    def __init__(self, example: np.array):
        x = example.flatten()
        constraints = {}

        for i, feature in enumerate(x):
            # we create discrete constraints for strings (for sure)
            # TODO using ints for discrete constraints is very quick and dirty, but we may regret it down the line
            if isinstance(feature, (str, int, np.int32, np.int64)):
                constraints[i] = DiscreteConstraint(value=feature, index=i)
            # we create ordinal constraints for floats (for sure)
            elif isinstance(feature, (float, np.float32, np.float64)):
                constraints[i] = OrdinalConstraint(value_min=feature, value_max=feature, index=i)
            else:
                raise NotImplementedError(f'found {type(feature)} for {feature}')

        super().__init__(constraints)
