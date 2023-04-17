class Constraint:
    '''
    A constraint always has an index, that is used to pick up the i-th feature which is constrained.
    '''

    def __init__(self, index=None):
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def satisfied(self, x):
        '''
        Checks if a constraint is satisfied.
        Parameters
        ----------
        x np.array data point to check if is satisfied

        Returns True if the constraint is satisfied for x
        -------

        '''
        raise NotImplementedError()

    def generalize(self, x):
        '''
        Returns the generalization of a constraint, that can be None if the constraint would be always satisfied.
        Parameters
        ----------
        x np.array data point to use for the generalization

        Returns a new constraint, or None
        -------

        '''
        raise NotImplementedError()
