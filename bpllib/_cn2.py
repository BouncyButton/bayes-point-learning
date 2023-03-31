import collections

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from bpllib._bp import callable_rules_bo, callable_rules_bp, predict_one_with_best_k


class CN2:
    _dataPath = '../Data/csv/'
    _E = []
    _selectors = []

    def __init__(self, star_max_size=5, min_significance=0.5):
        self.data = None
        self.star_max_size = star_max_size
        self.min_significance = min_significance

    def fit(self, df):
        """
        This function is used to learn the rule-based classification model with the CN2 algorithm.
        :param file_name: the name of the training file in CSV format.
        The file must be located in the '../Data/csv/' folder.
        """
        self.data = df
        self._E = df.copy()
        self.compute_selectors()

        # This list will contain the complex-class pairs that will represent the rules found by the CN2 algorithm.
        rule_list = []
        classes = df.loc[:, [list(df)[-1]]]
        classes_count = classes.iloc[:, 0].value_counts()

        while len(self._E) > 0:
            best_cpx = self.find_best_complex()
            if best_cpx is not None:
                covered_examples = self.get_covered_examples(self._E, best_cpx)
                most_common_class, count = self.get_most_common_class(covered_examples)
                self._E = self.remove_examples(self._E, covered_examples)

                total = 0
                if most_common_class in classes_count.keys():
                    total = classes_count[most_common_class]
                coverage = count / total
                # Precision: how many covered examples belong to the most common class
                precision = count / len(covered_examples)

                rule_list.append((best_cpx, most_common_class, coverage, precision))
            else:
                break

        most_common_class, count = self.get_most_common_class(df.index)
        total = classes_count[most_common_class]
        coverage = count / total
        precision = count / len(df)
        rule_list.append((None, most_common_class, coverage, precision))
        self.rules_ = rule_list  # TODO convert to my format
        return self

    def predict(self, X):
        """
        This function is used to test the CN2 classification model on the test file required as parameter, using the
        rule list also received as parameter. The rule list can be either produced with the fit function, or loaded with
        pickle.
        :param test_file_name: the name of the testing file in CSV format.
        The file must be located in the '../Data/csv/' folder.
        :param rule_list: a list containing the rules to be used to test the dataset.
        The rules are assumed to be in the correct format (the same produced by the fit function).
        """
        pass


    def compute_selectors(self):
        """
        This function computes the selectors from the input data, which are
        the pairs attribute-value, excluding the class attribute.
        Assumption: the class attribute is the last attribute of the dataset.
        """
        attributes = list(self.data)

        # removing the class attribute
        del attributes[-1]

        for attribute in attributes:
            possible_values = set(self.data[attribute])
            for value in possible_values:
                self._selectors.append((attribute, value))

    def find_best_complex(self):
        '''
        This function finds the best complex by continuously specializing the list of the best complex found so far and
        updating the best complex if the new complex found has a lower entropy than the previous one.
        The function keeps searching until the best complex has an accepted significance level.
        :return: the best complex found.
        '''
        best_complex = None
        best_complex_entropy = float('inf')
        best_complex_significance = 0
        star = []

        while True:
            entropy_measures = {}
            new_star = self.specialize_star(star, self._selectors)
            for idx in range(len(new_star)):
                tested_complex = new_star[idx]
                significance = self.significance(tested_complex)
                if significance > self.min_significance:
                    entropy = self.entropy(tested_complex)
                    entropy_measures[idx] = entropy
                    if entropy < best_complex_entropy:
                        best_complex = tested_complex.copy()
                        best_complex_entropy = entropy
                        best_complex_significance = significance
            top_complexes = sorted(entropy_measures.items(), key=lambda x: x[1], reverse=False)[:self.star_max_size]
            star = [new_star[x[0]] for x in top_complexes]
            if len(star) == 0 or best_complex_significance < self.min_significance:
                break

        return best_complex

    def remove_examples(self, all_examples, indexes):
        '''
        Removes from the dataframe of the remaining examples, the covered examples with the indexes received as parameter.
        :param all_examples: the dataframe from which we want to remove the examples.
        :param indexes: list of index labels that identify the instances to remove.
        :return: the remaining examples after removing the required examples.
        '''
        remaining_examples = all_examples.drop(indexes)
        return remaining_examples

    def get_covered_examples(self, all_examples, best_cpx):
        '''
        Returns the indexes of the examples from the list of all examples that are covered by the complex.
        :param all_examples: the dataframe from which we want to find the covered examples.
        :param best_cpx: list of attribute-value tuples.
        :return: the indexes of the covered examples.
        '''
        # Creating a dictionary with the attributes of the best complex as key, and the values of that attribute as a
        # list of values. Then, add all the possible values for the attributes that are not part of the rules of the
        # best complex.
        values = dict()
        [values[t[0]].append(t[1]) if t[0] in list(values.keys())
         else values.update({t[0]: [t[1]]}) for t in best_cpx]
        for attribute in list(self.data):
            if attribute not in values:
                values[attribute] = set(self.data[attribute])

        # Getting the indexes of the covered examples
        covered_examples = all_examples[all_examples.isin(values).all(axis=1)]
        return covered_examples.index

    def get_most_common_class(self, covered_examples):
        '''
        Returns the most common class among the examples received as parameter. It assumes that the class is the last
        attribute of the examples.
        :param covered_examples: Pandas DataFrame containing the examples from which we want to find the most common
        class.
        :return: label of the most common class.
        '''
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        most_common_class = classes.iloc[:, 0].value_counts().index[0]
        count = classes.iloc[:, 0].value_counts()[0]
        return most_common_class, count

    def specialize_star(self, star, selectors):
        '''
        This function creates a new_star list by combining the complexes in star with the selectors, and removing the
        non-valid complexes created.
        :param star: the list of complexes to be specialized
        :param selectors: the list of selector with which to specialize star
        :return: the new_star list with the specialized complexes
        '''
        new_star = []
        if len(star) > 0:
            for complex in star:
                for selector in selectors:
                    new_complex = complex.copy()
                    new_complex.append(selector)

                    # Add the new complex only if they are valid
                    count = collections.Counter([x[0] for x in new_complex])
                    duplicate = False
                    for c in count.values():
                        if c > 1:
                            duplicate = True
                            break
                    if not duplicate:
                        new_star.append(new_complex)
        else:
            for selector in selectors:
                new_star.append([selector])
        return new_star

    def significance(self, tested_complex):
        '''
        This function computes the significance of a complex
        :param tested_complex: the complex for which we want to compute the significance.
        :return: the entropy of the significance.
        '''
        covered_examples = self.get_covered_examples(self._E, tested_complex)
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        covered_num_instances = len(classes)
        covered_counts = classes.iloc[:, 0].value_counts()
        covered_probs = covered_counts.divide(covered_num_instances)

        train_classes = self.data.iloc[:, -1]
        train_num_instances = len(train_classes)
        train_counts = train_classes.value_counts()
        train_probs = train_counts.divide(train_num_instances)

        significance = covered_probs.multiply(np.log(covered_probs.divide(train_probs))).sum() * 2

        return significance

    def entropy(self, tested_complex):
        '''
        This function computes the entropy of a complex
        :param tested_complex: the complex for which we want to compute the entropy.
        :return: the entropy of the complex.
        '''
        covered_examples = self.get_covered_examples(self._E, tested_complex)
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        num_instances = len(classes)
        class_counts = classes.iloc[:, 0].value_counts()
        class_probabilities = class_counts.divide(num_instances)
        log2 = np.log2(class_probabilities)
        plog2p = class_probabilities.multiply(log2)
        entropy = plog2p.sum() * -1

        return entropy

    def print_rules(self, rules):
        '''
        This function prints the rules received as parameter in an understandable way.
        It also prints the coverage and precision of each rule.
        :param rules: the rules that have to be printed.
        '''
        rule_string = ''
        for rule in rules:
            complex = rule[0]
            complex_class = rule[1]
            coverage = rule[2]
            precision = rule[3]

            if complex is not None:
                for idx in range(len(complex)):
                    if idx == 0:
                        rule_string += 'If '
                    rule_string += str(complex[idx][0]) + '=' + str(complex[idx][1])
                    if idx < len(complex) - 1:
                        rule_string += ' and '
                rule_string += ', then class=' + complex_class + ' [covered examples = ' + str(
                    coverage) + ', precision = ' \
                               + str(precision) + ']'
            else:
                rule_string += 'Default: class=' + complex_class + ' [covered examples = ' + str(
                    coverage) + ', precision = ' \
                               + str(precision) + ']'
            print(rule_string)
            rule_string = ''


class CN2Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, maxstar=3, T=1, verbose=0, min_significance=0.5):
        self.maxstar = maxstar
        self.T = T
        self.verbose = verbose
        self.min_significance = min_significance

    def fit(self, X, y, target_class=1):
        self.target_class_ = target_class
        X, y = check_X_y(X, y, dtype=[str, int])
        self.classes_ = unique_labels(y)
        if len(self.classes_) == 2:
            self.other_class_ = (set(self.classes_) - {target_class}).pop()
        else:
            raise NotImplementedError('multiclass not available yet')
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        df = pd.DataFrame(X)
        df['class'] = y.astype(int)

        if self.T == 1:
            self.inner_clf_ = CN2(star_max_size=self.maxstar, min_significance=self.min_significance)
            self.inner_clf_.fit(df)
            self.rules = self.inner_clf_.rules_
        else:
            self.rulesets_ = self.cn2_with_multiple_runs(X, y, target_class, T=self.T, maxstar=self.maxstar)

        return self

    def predict(self, X, strategy='bo'):
        if self.T == 1:
            return [self.predict_one(x) for x in X]
        else:
            return [self.predict_one(x, strategy='bo') for x in X]

    def predict_one(self, x, strategy=None, **kwargs):
        if self.T == 1:
            for rule in self.rules_:
                if rule.covers(x):
                    return self.target_class_
            return 1 - self.target_class_
        elif strategy == 'bo':
            rules_bo = callable_rules_bo(self.rulesets_)
            value = sum([ht(x) for ht in rules_bo])
            return self.target_class_ if np.sign(value) > 0 else self.other_class_
        elif strategy == 'bp':
            h_bp = callable_rules_bp(self.rulesets_)
            value = h_bp(x)
            return self.target_class_ if value > self.T / 2 else self.other_class_
        elif strategy == 'best-k':
            most_freq_rules = kwargs.get('most_freq_rules')
            n_rules = kwargs.get('n_rules')
            return predict_one_with_best_k(x, n_rules, n_rules, self.counter_, most_freq_rules, self.target_class_,
                                           self.other_class_, self.T)

        else:
            raise NotImplementedError('strategy not implemented')
