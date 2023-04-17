from bpllib.rules._constraint import Constraint


class OrdinalConstraint(Constraint):
    '''
    An ordinal constraint contains a value_min and a value_max, that define a bound for continuous values.
    '''

    def __init__(self, value_min=None, value_max=None, index=None):
        self.value_min = value_min
        self.value_max = value_max
        super().__init__(index=index)

    def __repr__(self):
        return f'{self.value_min} <= X[{self.index}] <= {self.value_max}'

    def satisfied(self, x):
        return self.value_min <= x[self.index] <= self.value_max

    def generalize(self, x):
        return OrdinalConstraint(value_min=min(self.value_min, x[self.index]),
                                 value_max=max(self.value_max, x[self.index]),
                                 index=self.index)

    def __and__(self, other):
        if not isinstance(other, OrdinalConstraint):
            raise TypeError("Only ordinal constraints can be combined with an ordinal constraint")
        return OrdinalConstraint(value_min=max(self.value_min, other.value_min),
                                 value_max=min(self.value_max, other.value_max),
                                 index=self.index)
