from bpllib.rules._constraint import Constraint
from bpllib.rules._discrete_or_constraint import DiscreteOrConstraint


class DiscreteConstraint(Constraint):
    '''
    A discrete constraint contains a value, that needs to be checked for equality in a constraint check.
    '''

    def __init__(self, value=None, index=None):
        self.value = value
        super().__init__(index=index)

    def __eq__(self, other):
        if not isinstance(other, DiscreteConstraint):
            return False
        return super().__eq__(other) and self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, DiscreteConstraint):
            return False
        return super().__lt__(other) and self.value < other.value

    def __hash__(self):
        return hash((self.index, self.value))

    def __len__(self):
        return 1

    def __and__(self, other):
        if isinstance(other, DiscreteOrConstraint):
            if self.value in other.values:
                return self
            else:
                return None
        if self.value == other.value:
            return self
        return None

    def satisfied(self, x):
        return x[self.index] == self.value

    def generalize(self, x):
        if self.value == x[self.index]:
            return self
        return None

    def __repr__(self):
        return f'X[{self.index}] == {self.value.__repr__()}'
