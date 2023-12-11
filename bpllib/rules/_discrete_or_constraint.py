from bpllib.rules._constraint import Constraint


class DiscreteOrConstraint(Constraint):
    def __init__(self, values, index):
        super().__init__(index=index)
        self.values = set(values)

    def __eq__(self, other):
        if not isinstance(other, DiscreteOrConstraint):
            return False
        return super().__eq__(other) and self.values == other.values

    def __len__(self):
        return len(self.values)

    def __and__(self, other):
        from bpllib.rules._discrete_constraint import DiscreteConstraint

        if isinstance(other, DiscreteConstraint):
            if other.value in self.values:
                return other
            else:
                return None
        intersection = self.values.intersection(other.values)
        if len(intersection) > 1:
            return DiscreteOrConstraint(values=intersection, index=self.index)
        if len(intersection) == 1:
            return DiscreteConstraint(value=intersection.pop(), index=self.index)
        return None

    def satisfied(self, x):
        return x[self.index] in self.values

    def __repr__(self):
        return f'X[{self.index}] in {self.values}'

    def __hash__(self):
        return hash((self.index, tuple(sorted(self.values))))
