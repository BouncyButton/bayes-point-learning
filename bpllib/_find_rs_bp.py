import numpy as np

from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._rule_by_example import RuleByExample


class FindRsClassifier(BayesPointClassifier):
    '''
    This class implements the find-rs algorithm.
    '''
    description = 'Find-RS'

    def __init__(self,
                 put_negative_on_top=False,
                 tol=0,
                 n_bins=None,
                 T=3,
                 verbose=0,
                 threshold_acc=0.99,
                 target_class=None,
                 pool_size='auto',
                 find_best_k=True,
                 random_state=None,
                 encoding='av',
                 to_string=True,
                 cachedir=None,
                 prune_strategy=None,
                 max_rules=None):
        self.put_negative_on_top = put_negative_on_top
        self.n_bins = n_bins
        self.tol = tol

        # read BaseEstimator docs
        #     All estimators should specify all the parameters that can be set
        #     at the class level in their ``__init__`` as explicit keyword
        #     arguments (no ``*args`` or ``**kwargs``).
        # hence, we need to repeat ourselves here
        # otherwise, we break the sklearn API contract (and CV does not work)
        # alternative here: https://stackoverflow.com/questions/51430484/

        super().__init__(T=T,
                         verbose=verbose,
                         threshold_acc=threshold_acc,
                         target_class=target_class,
                         pool_size=pool_size,
                         find_best_k=find_best_k,
                         random_state=random_state,
                         encoding=encoding,
                         to_string=to_string,
                         cachedir=cachedir,
                         prune_strategy=prune_strategy,
                         max_rules=max_rules)

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

            # if self.n_bins is not None and len(B) >= 2 * self.n_bins:
            #    break
            # i believe that this will lead to worse performance down the line
            # since pruning increases effectiveness
            # but the obvious issue is that the runtime will increase

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

        # trim to n_bins if still too long
        # if self.n_bins is not None and len(D) > self.n_bins:
        #     D, B = D[:self.n_bins], B[:self.n_bins]
        return D, B
