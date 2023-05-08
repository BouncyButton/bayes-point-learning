from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._rule import Rule
from imodels import BayesianRuleSetClassifier as BRS


class BayesianRuleSetClassifier(BayesPointClassifier):
    '''
    This class implements the BRS algorithm.
    '''
    description = 'BRS'

    def __init__(self, verbose=0, threshold_acc=0.99, **kwargs):
        self.n_rules = kwargs.pop('n_rules', 2000)
        self.supp = kwargs.pop('supp', 5)
        self.maxlen = kwargs.pop('maxlen', 10)
        self.num_iterations = kwargs.pop('num_iterations', 5000)
        self.num_chains = kwargs.pop('num_chains', 3)
        self.q = kwargs.pop('q', 0.1)
        self.alpha_pos = kwargs.pop('alpha_pos', 100)
        self.beta_pos = kwargs.pop('beta_pos', 1)
        self.alpha_neg = kwargs.pop('alpha_neg', 100)
        self.beta_neg = kwargs.pop('beta_neg', 1)
        self.alpha_l = kwargs.pop('alpha_l', None)
        self.beta_l = kwargs.pop('beta_l', None)
        self.discretization_method = kwargs.pop('discretization_method', 'randomforest')
        self.random_state = kwargs.pop('random_state', 0)

        super().__init__(verbose=verbose, threshold_acc=threshold_acc, **kwargs)

        # BRS requires numerical data (i'll skip using to_string)
        self.to_string = False
        self.use_bootstrap = True

    def base_method(self, X, y, target_class):
        clf = BRS(n_rules=self.n_rules, supp=self.supp, maxlen=self.maxlen, num_iterations=self.num_iterations,
                  num_chains=self.num_chains, q=self.q, alpha_pos=self.alpha_pos, beta_pos=self.beta_pos,
                  alpha_neg=self.alpha_neg, beta_neg=self.beta_neg, alpha_l=self.alpha_l, beta_l=self.beta_l,
                  discretization_method=self.discretization_method, random_state=self.random_state)

        # todo: fpgrowth does not work
        # and it needs to be rewritten;
        # start with line 246 of brs.py
        #             rules = fpgrowth([itemMatrix[i] for i in pindex], min_support=self.supp, max_len=self.maxlen)

        # also a parameter here would be nice
        # n_estimators = min(pow(df.shape[1], length), 300)

        clf.fit(X, y, verbose=self.verbose)
        rules = clf.rules_

        my_rules = []
        for rule in rules:
            data = [(int(attribute.replace("X", "").split('_')[0]), 'neg' in attribute) for attribute in rule]
            my_rule = Rule(
                constraints={idx: DiscreteConstraint(value=0 if neg else 1, index=idx) for idx, neg in data})
            my_rules.append(my_rule)

        return my_rules
