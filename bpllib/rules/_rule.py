import numpy as np


class Rule:
    '''
    A rule describes a condition for classifying a data point to the target class.
    It is made up by many constraints, each one considering a single feature.
    '''

    def __init__(self, constraints: dict):
        self.constraints = constraints  # was set()
        self.columns = None

    def __mul__(self, other):
        '''
        Returns the conjunction of two rules
        Parameters
        ----------
        other: the other rule

        -------

        '''
        new_constraints = {}
        for index in set(self.constraints.keys()).union(set(other.constraints.keys())):
            c1 = self.constraints.get(index)
            c2 = other.constraints.get(index)
            if c1 is None:
                new_constraints[index] = c2
            elif c2 is None:
                new_constraints[index] = c1
            else:
                c = c1 & c2
                if c is None:
                    return None
                else:
                    new_constraints[index] = c
        return Rule(new_constraints)

    def generalize(self, x: np.array):
        '''
        Generalizes a rule w.r.t. a given input.
        In practice, we relax constraints by removing them or dilating their bounds.
        Parameters
        ----------
        x np.array which contains a single example

        Returns the generalization of the current rule compared to the current example.
        -------

        '''
        new_constraints = {}
        for idx, constraint in self.constraints.items():
            new_constraint = constraint.generalize(x)
            if new_constraint is not None:
                new_constraints[idx] = new_constraint
        return Rule(new_constraints)

    def covers(self, x: np.array):
        '''
        Checks if the current rule covers an input example
        Parameters
        ----------
        x np.array containing a single example

        Returns True if x is covered
        -------

        '''
        for constraint in self.constraints.values():
            if not constraint.satisfied(x):
                return False
        return True

    def covers_any(self, data: np.array, tol=0):
        '''
        Checks if any data point in the data array is covered by this rule.
        Parameters
        ----------
        data: np.array which contains the data to be processed
        tol: the hyperparameter which enables unpure bins that contain negative examples.

        -------

        '''
        not_covered = []
        for i, data_point in enumerate(data):
            if self.covers(data_point):
                not_covered.append(i)
                if len(not_covered) > tol:
                    return not_covered
        return []

    def covers_all(self, data: np.array):
        '''
        Checks if all data points in the data array are covered by this rule.
        Parameters
        ----------
        data: np.array which contains the data to be processed

        -------

        '''
        for data_point in data:
            if not self.covers(data_point):
                return False
        return True

    def __call__(self, *args, **kwargs):
        return self.covers(*args, **kwargs)

    def examples_covered(self, X):
        return X[[self.covers(x) for x in X]]

    def examples_not_covered(self, X):
        return X[[not self.covers(x) for x in X]]

    def __repr__(self):
        return " ^ ".join(str(c) for c in sorted(self.constraints.values(), key=lambda c: c.index))

    def __len__(self):
        return len(self.constraints)

    def values_count(self):
        return sum(len(c) for c in self.constraints.values())

    def __eq__(self, other):
        return tuple(sorted(self.constraints.items())) == tuple(sorted(other.constraints.items()))

    def __hash__(self):
        return hash(tuple(sorted(self.constraints.items())))

    def str_with_column_names(self, columns):
        self.columns = columns
        reprs = []
        for c in self.constraints.values():
            column = columns[c.index]
            reprs.append(f"{column}={c.value}")
        return " ^ ".join(reprs)
