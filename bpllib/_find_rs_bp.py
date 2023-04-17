import numpy as np

from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._rule_by_example import RuleByExample


class FindRsClassifier(BayesPointClassifier):
    '''
    This class implements the find-rs algorithm.
    '''
    description = 'Find-RS'

    def __init__(self, put_negative_on_top=False, tol=0, **kwargs):
        self.put_negative_on_top = put_negative_on_top
        self.tol = tol

        super().__init__(**kwargs)

    def base_method(self, X, y, target_class):
        '''
        This method implements the find-rs algorithm.
        '''
        tol = self.tol
        put_negative_on_top = self.put_negative_on_top
        train_p = list(X[y == target_class])
        train_n = list(X[y != target_class])

        D, B, k = [], [], 0

        while len(train_p) > 0:
            if self.verbose > 5:
                print("{0} ({1:.2%})".format(len(train_p), 1 - len(train_p) / (y == target_class).sum()))
            first = train_p.pop(0)
            B.append([first])
            D.append(RuleByExample(first))

            incompatibles = []
            while len(train_p) > 0:
                r = D[-1]
                p = train_p.pop(0)

                new_r = r.generalize(p)
                not_covered = new_r.covers_any(train_n, tol=tol)
                if not not_covered:
                    D[-1] = new_r  # RuleByExample(p) ??
                    B[-1].append(p)
                else:
                    incompatibles.append(p)  # occhio all'ordine!

                    if put_negative_on_top:
                        train_n = np.insert(train_n, 0, train_n[not_covered[0]], axis=0)
                        train_n = np.delete(train_n, not_covered[0] + 1, axis=0)

            train_p = incompatibles
        D, B = self._prune(D, B)
        # right now i won't use the B array, but it could be useful in the future
        return D

    def _prune(self, D, B):
        old_len = len(D)

        i = 0
        while i < len(D):
            current_b = list(B[i].copy())
            for j in range(i + 1, len(D)):
                next_rule = D[j]
                k = 0

                while k < len(current_b):
                    p = current_b[k]
                    if next_rule.covers(p):
                        current_b.pop(k)
                    else:
                        k += 1
            if not current_b:
                B.pop(i)
                D.pop(i)
            else:
                i += 1

        if old_len > len(D):
            pass
            if self.verbose > 5:
                print("pruned from " + str(old_len) + " to " + str(len(D)) + " rules")
        return D, B
