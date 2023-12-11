import random
from collections import Counter
from copy import copy, deepcopy

import numpy as np

from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._rule import Rule
from bpllib.rules._rule_by_example import RuleByExample


class FindRsClassifier(BayesPointClassifier):
    '''
    This class implements the find-rs algorithm.
    '''
    description = 'Find-RS'

    def __init__(self,
                 put_negative_on_top=False,
                 tol=None,
                 bin_purity=1.0,
                 n_bins=None,
                 rule_pruning=False,
                 T=3,
                 verbose=0,
                 bp_verbose=0,
                 threshold_acc=0.99,
                 target_class=None,
                 pool_size='auto',
                 find_best_k=True,
                 random_state=None,
                 encoding='av',
                 to_string=True,
                 max_rules=None,
                 generalization_probability=0.9,
                 permute_constraints=True):
        self.permute_constraints = permute_constraints
        self.generalization_probability = generalization_probability
        self.put_negative_on_top = put_negative_on_top
        self.n_bins = n_bins
        self.tol = tol
        self.rule_pruning = rule_pruning
        self.bin_purity = bin_purity
        self.Bs_ = []

        # read BaseEstimator docs
        #     All estimators should specify all the parameters that can be set
        #     at the class level in their ``__init__`` as explicit keyword
        #     arguments (no ``*args`` or ``**kwargs``).
        # hence, we need to repeat ourselves here
        # otherwise, we break the sklearn API contract (and CV does not work)
        # alternative here: https://stackoverflow.com/questions/51430484/

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
                         max_rules=max_rules)

    def base_method(self, X, y, target_class):
        '''
        This method implements the find-rs algorithm.
        '''
        tol = self.tol
        permute_constraints = self.permute_constraints
        bin_purity = self.bin_purity
        put_negative_on_top = self.put_negative_on_top
        train_p = list(X[y == target_class])
        train_n = list(X[y != target_class])

        D, B, k = [], [], 0
        random.seed(self.random_state)

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
                not_covered = new_r.covers_any(train_n, tol=tol, bin_purity=bin_purity, n_positive_covered=len(B[-1]))
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
        D, B = self._post_pruning(D, B)

        if hasattr(self, 'enc_'):
            D = self._simplify_onehot_encoded(D)

        if self.rule_pruning:
            D = [self._rule_pruning(rule, train_n, len(b), permute_constraints=permute_constraints) for rule, b in
                 zip(D, B)]

        # right now i won't use the B array, but it could be useful in the future
        return D, B

    def _simplify_onehot_encoded(self, D):
        new_ruleset = []

        for rule in D:
            simplified_rule = self.simplify_onehot_encoded_rule_(rule)
            new_ruleset.append(simplified_rule)
        return new_ruleset

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

    def _post_pruning(self, D, B):
        max_rules = self.max_rules
        # pick the top N rules. select most popolous rules first (check bins)
        if max_rules is not None and len(D) > max_rules:
            # order D and B by len(B[i])
            D, B = zip(*sorted(zip(D, B), key=lambda x: len(x[1]), reverse=True))
            D, B = list(D), list(B)
            D, B = D[:max_rules], B[:max_rules]
        return D, B

    def _rule_pruning(self, rule, N, n_positive_covered, permute_constraints=True):
        # greedy version
        # for each constraint, remove it and check if it still covers any negative example
        # if it does not, remove it and continue

        # this is not optimal, we could try to search exhaustively
        # using the properties of the rule lattice
        # (e.g.,
        # - R covers any N => R-c covers any N)

        final_rule = deepcopy(rule)

        indexes = list(rule.constraints.keys())
        if permute_constraints:
            random.shuffle(indexes)

        for i in indexes:
            new_rule = final_rule
            c = new_rule.constraints.pop(i)
            covers = new_rule.covers_any(N, bin_purity=self.bin_purity, n_positive_covered=n_positive_covered)
            if covers:
                # this constraint was needed, put it back
                final_rule.constraints[i] = c
            else:
                p = random.random()
                # the constraint was not needed, leave it removed with probability generalization_probability

                # tenembaum says we should pick according to size of the bin.
                gen_prob = self.generalization_probability  # ** n_positive_covered
                if p <= gen_prob:
                    continue  # this is the prob of removal
                else:
                    final_rule.constraints[i] = c

        return final_rule
