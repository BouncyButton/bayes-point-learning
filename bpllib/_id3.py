# from __future__ import print_function

import math

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from bpllib._bp import alpha_representation, callable_rules_bp, callable_rules_bo
from bpllib._bpl import Rule, DiscreteConstraint


def extract_ruleset(tree):
    rules = []
    if isinstance(tree, Leaf):
        rules.append(tree.label)
        return rules
    for v in tree.subtrees:
        rules.append(tree.label + ' = ' + v)
        rules.extend(extract_ruleset(tree.subtrees[v]))
    return rules


class ID3Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, T=1):
        self.T = T


    def predict(self, X, strategy='bo', n_rules=None):
        if self.T == 1:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return np.array([1 if any([r.covers(x) for r in self.ruleset_]) else 0 for x in X])

        if strategy == 'bo' or strategy == 'bp' or strategy == 'best-k':
            if isinstance(X, pd.DataFrame):
                X = np.array(X.values)
            if strategy == 'best-k':
                most_freq_rules = self.counter_.most_common(n_rules)
                return np.array([self.predict_one(x, most_freq_rules=most_freq_rules,
                                                  strategy=strategy, n_rules=n_rules) for x in X])
            else:
                return np.array([self.predict_one(x, strategy=strategy, n_rules=n_rules) for x in X])
        else:
            raise NotImplementedError('strategy not implemented')

    def predict_one(self, x, strategy='bo', n_rules=None, most_freq_rules=None):
        if strategy == 'bo':
            rules_bo = callable_rules_bo(self.rulesets_)
            value = sum([ht(x) for ht in rules_bo])
            return self.target_class_ if np.sign(value) > 0 else self.other_class_
        elif strategy == 'bp':
            h_bp = callable_rules_bp(self.rulesets_)
            value = h_bp(x)
            return self.target_class_ if value > self.T / 2 else self.other_class_
        elif strategy == 'best-k':
            if n_rules is None and not hasattr(self, 'suggested_k_'):
                raise ValueError('n_rules must be specified')

            # if n_rules is None:
            #     n_rules = self.suggested_k_
            vote = 0

            alpha_rules = sum(self.counter_.values())
            if alpha_rules == 0:
                return self.other_class_
            alpha_k_rules = sum([alpha for r, alpha in self.counter_.most_common(n_rules)])
            gamma_k = alpha_k_rules / alpha_rules

            if most_freq_rules is None:
                most_freq_rules = self.counter_.most_common(n_rules)

            for rule, alpha in most_freq_rules:
                vote += rule.covers(x) * alpha
            return self.target_class_ if vote > gamma_k * self.T / 2 else self.other_class_
        else:
            raise NotImplementedError('strategy not available')

    def fit(self, X, y, target_class=1, find_best_k=False, starting_seed=0):
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

        df_with_no_class = df.drop('class', axis=1)


        classes = get_attribute_value_space('class', df)
        tot_entropy = E(df, classes, 'class')
        # get dataframe column names
        A = list(df.columns)
        # remove class column
        A.remove('class')

        if self.T == 1:
            my_dict = {col: list(df[col]) for col in df.columns}
            tree = mine_c45(my_dict, 'class')  # DecisionTreeID3()  #
            # tree = ID3(tot_entropy, df, A, classes, 0, 'class')
            # tree.fit(df_with_no_class, df.iloc[:, -1])  # df['class'])
            self.inner_clf_ = tree
            ruleset = tree_to_my_rules(tree) #extract_rules_from_id3(tree)  #tree_to_rules(tree)
            self.ruleset_ = ruleset

        else:
            self.rulesets_ = []

            self.classifiers_ = []
            for t in range(self.T):
                # permute the df
                # we use bootstrap to have a different dataset for each classifier
                # hoping that the rules will be different enough
                perm_df = df.sample(frac=1, random_state=t, replace=True)
                my_dict = {col: list(perm_df[col]) for col in df.columns}

                # create a new classifier
                # tree = ID3(tot_entropy, perm_df, A, classes, 0, 'class')  #random_state=starting_seed * self.T + t)
                tree = mine_c45(my_dict, 'class')  # DecisionTreeID3()
                # tree.fit(df_with_no_class, df['class'])
                self.classifiers_.append(tree)
                ruleset = tree_to_my_rules(tree)
                self.rulesets_.append(ruleset)
            self.counter_ = alpha_representation(self.rulesets_)

        return self


from collections import OrderedDict


def deldup(li):
    """ Deletes duplicates from list _li_
        and return new list with unique values.
    """
    return list(OrderedDict.fromkeys(li))


def is_mono(t):
    """ Returns True if all values of _t_ are equal
        and False otherwise.
    """
    for i in t:
        if i != t[0]:
            return False
    return True


def get_indexes(table, col, v):
    """ Returns indexes of values _v_ in column _col_
        of _table_.
    """
    li = []
    start = 0
    for row in table[col]:
        if row == v:
            index = table[col].index(row, start)
            li.append(index)
            start = index + 1
    return li


def get_values(t, col, indexes):
    """ Returns values of _indexes_ in column _col_
        of the table _t_.
    """
    return [t[col][i] for i in range(len(t[col])) if i in indexes]


def del_values(t, ind):
    """ Creates the new table with values of _ind_.
    """
    return {k: [v[i] for i in range(len(v)) if i in ind] for k, v in t.items()}


def print_list_tree(tree, tab=''):
    """ Prints list of nested lists in
        hierarchical form.
    """
    print('%s[' % tab)
    for node in tree:
        if isinstance(node, str):
            print('%s  %s' % (tab, node))
        else:
            print_list_tree(node, tab + '  ')
    print('%s]' % tab)


def formalize_rules(list_rules):
    """ Gives an list of rules where
        facts are separeted by coma.
        Returns string with rules in
        convinient form (such as
        'If' and 'Then' words, etc.).
    """
    text = ''
    for r in list_rules:
        t = [i for i in r.split(',') if i]
        text += 'If %s,\n' % t[0]
        for i in t[1:-1]:
            text += '   %s,\n' % i
        text += 'Then: %s.\n' % t[-1]
    return text


def get_subtables(t, col):
    """ Returns subtables of the table _t_
        divided by values of the column _col_.
    """
    return [del_values(t, get_indexes(t, col, v)) for v in deldup(t[col])]


import math



def freq(table, col, v):
    """ Returns counts of variant _v_
        in column _col_ of table _table_.
    """
    return table[col].count(v)


def info(table, res_col):
    """ Calculates the entropy of the table _table_
        where res_col column = _res_col_.
    """
    s = 0  # sum
    for v in deldup(table[res_col]):
        p = freq(table, res_col, v) / float(len(table[res_col]))
        s += p * math.log(p, 2)
    return -s


def infox(table, col, res_col):
    """ Calculates the entropy of the table _table_
        after dividing it on the subtables by column _col_.
    """
    s = 0  # sum
    for subt in get_subtables(table, col):
        s += (float(len(subt[col])) / len(table[col])) * info(subt, res_col)
    return s


def gain(table, x, res_col):
    """ The criterion for selecting attributes for splitting.
    """
    return info(table, res_col) - infox(table, x, res_col)


def mine_c45(table, result):
    """ An entry point for C45 algorithm.
        _table_ - a dict representing data table in the following format:
        {
            "<column name>': [<column values>],
            "<column name>': [<column values>],
            ...
        }
        _result_: a string representing a name of column indicating a result.
    """
    col = max([(k, gain(table, k, result)) for k in table.keys() if k != result],
              key=lambda x: x[1])[0]
    tree = []
    for subt in get_subtables(table, col):
        v = subt[col][0]
        if is_mono(subt[result]):
            tree.append(['%s=%s' % (col, v),
                         '%s=%s' % (result, subt[result][0])])
        else:
            del subt[col]
            tree.append(['%s=%s' % (col, v)] + mine_c45(subt, result))
    return tree

def tree_to_rules(tree):
    return formalize_rules(__tree_to_rules(tree))

def tree_to_my_rules(tree):
    rulelist = __tree_to_rules(tree)
    myruleset = []
    for rule in rulelist:
        myconstraints = {}
        constraints = rule.split(",")
        for constraint in constraints:
            v = constraint.split('=')
            if v == ['']:
                continue
            attrindex, val = v[0], v[1]
            # convert attrindex to int if possible
            if attrindex != 'class':
                attrindex = int(attrindex)

            # convert val to int if possible
            try:
                val = int(val)
            except ValueError:
                pass

            if attrindex == 'class' and val == 0:
                continue
            if attrindex == 'class' and val == 1:
                myrule = Rule(myconstraints)
                myruleset.append(myrule)
            else:
                myconstraints[attrindex] = DiscreteConstraint(index=attrindex, value=val)
    return myruleset

def __tree_to_rules(tree, rule=''):
    rules = []
    for node in tree:
        if isinstance(node, str):
            rule += node + ','
        else:
            rules += __tree_to_rules(node, rule)
    if rules:
        return rules
    return [rule]

def validate_table(table):
    assert isinstance(table, dict)
    for k, v in table.items():
        assert k
        assert isinstance(k, str)
        assert len(v) == len(table.values()[0])
        for i in v: assert i










####################### no

class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids  # index of data in this node
        self.entropy = entropy  # entropy, will fill later
        self.depth = depth  # distance to root node
        self.split_attribute = None  # which attribute is chosen, it non-leaf
        self.children = children  # list of its child nodes
        self.order = None  # order of values of split_attribute in children
        self.label = None  # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0 * np.log(prob_0))


class DecisionTreeID3(object):
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    def _entropy(self, ids):
        # calculate entropy of a node with index ids
        if len(ids) == 0: return 0
        ids = [i for i in ids]  # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting
        target_ids = [i for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0])  # most frequent label

    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue  # entropy = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id for sub_id in sub_ids])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split: continue
            # information gain
            HxS = 0
            for split in splits:
                HxS += len(split) * self._entropy(split) / len(ids)
            gain = node.entropy - HxS
            if gain < self.min_gain: continue  # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split,
                                entropy=self._entropy(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        """
        :param new_data: a new dataframe, each row is a datapoint
        :return: predicted labels for each row
        """
        if isinstance(new_data, pd.DataFrame):
            npoints = new_data.count()[0]
            labels = [None] * npoints
            for n in range(npoints):
                x = new_data.iloc[n, :]  # one point
                # start from root and recursively travel if not meet a leaf
                node = self.root
                while node.children:
                    node = node.children[node.order.index(x[node.split_attribute])]
                labels[n] = node.label

            return labels
        else:
            npoints = len(new_data)
            labels = [None] * npoints
            for n in range(npoints):
                x = new_data[n]
                # start from root and recursively travel if not meet a leaf
                node = self.root
                while node.children:
                    node = node.children[node.order.index(x[node.split_attribute])]
                labels[n] = node.label

            return labels


# Define a hierarchy of classes in order to handle the construction of the Decision Tree
class Node:
    def __init__(self, label, level):
        self.label = label
        self.level = level


class InnerNode(Node):
    def __init__(self, label, level, information_gain):
        super(InnerNode, self).__init__(label, level)
        self.subtrees = {}
        self.information_gain = information_gain

    def __str__(self):
        s = str(self.label) + '[ Inf. Gain: ' + str(self.information_gain) + ' ]\n'
        offset = ''
        for _ in range(self.level):
            offset += '     '
        for n in self.subtrees.keys():
            s += offset + '|\n' + offset[:-1] + str(n) + '\n' + offset + '|____' + str(self.subtrees[n])
        return s


class Leaf(Node):
    def __init__(self, label, level):
        super(Leaf, self).__init__(label, level)

    def __str__(self):
        s = str(self.label) + '\n'
        return s


# Entropy
def E(S, classes, column_classes_label):
    entropy = 0
    for c in classes:
        card_Sc = len(S[S[column_classes_label] == c])
        pc = card_Sc / len(S)
        if pc == 1:
            return 0
        if pc == 0:
            continue
        entropy += pc * math.log2(pc)
    return -entropy


# Returns the set of values that a specific argument can have
def get_attribute_value_space(selected_attribute, S):
    values = S[[selected_attribute]].to_numpy().flatten()
    V = list(dict.fromkeys(values))
    return V


# Information Gain
def G(total_entropy, selected_attribute, S, classes, column_classes_label):
    sum = 0
    # Get set of values of 'selected_attribute'
    V = get_attribute_value_space(selected_attribute, S)
    for v in V:
        S_av = S[S[selected_attribute] == v]
        sum += (len(S_av) / len(S)) * E(S_av, classes, column_classes_label)
    return total_entropy - sum


def ID3(total_entropy, S, A, classes, level, column_classes_label):
    if len(A) == 0:
        majority_class = max(classes, key=lambda c: len(S[S[column_classes_label] == c]))
        return Leaf(label=majority_class, level=level)

    if total_entropy == 0:
        return Leaf(label=S[column_classes_label].values[0], level=level)

    print('Total Entropy at level ' + str(level) + ': ', str(total_entropy) + '\n')
    max_gain = np.inf
    t = tuple()
    # Find best attribute
    for a in A:
        a_gain = G(total_entropy, a, S, classes, column_classes_label)
        print(str(a) + '[ Information Gain: ', str(a_gain) + ' ]')
        if (a_gain > max_gain) or (max_gain == np.inf):
            max_gain = a_gain
            t = (a, a_gain)

    best_attribute = t[0]
    T = InnerNode(label=best_attribute, level=level, information_gain=t[1])

    ### Printing information
    print('\nBest Attribute:', T)
    print('\n-----------------------------------------\n')
    ###

    A.remove(best_attribute)
    V = get_attribute_value_space(best_attribute, S)
    for v in V:
        S_v = S[S[best_attribute] == v]
        T.subtrees[v] = ID3(E(S_v, classes, column_classes_label), S_v, A.copy(), classes, level + 1,
                            column_classes_label)
    return T


def test_ID3(df, A, column_classes_label):
    classes = get_attribute_value_space(column_classes_label, df)
    tot_entropy = E(df, classes, column_classes_label)
    Tree = ID3(tot_entropy, df, A, classes, 0, column_classes_label)
    print('Decision Tree representation\n')
    print(Tree)


def extract_rules_from_id3(tree, rules, rule):
    if isinstance(tree, Leaf):
        rules.append(rule)
        return
    for k in tree.subtrees.keys():
        new_rule = rule.copy()
        new_rule.append((tree.label, k))
        extract_rules_from_id3(tree.subtrees[k], rules, new_rule)