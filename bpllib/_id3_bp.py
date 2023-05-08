from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._rule import Rule


class Id3Classifier(BayesPointClassifier):
    '''
    This class implements the id3 algorithm.
    '''
    description = 'ID3'

    def __init__(self, verbose=0, threshold_acc=0.99, **kwargs):
        super().__init__(verbose=verbose, threshold_acc=threshold_acc, **kwargs)
        self.use_bootstrap = True

    def base_method(self, X, y, target_class, **kwargs):
        my_dict = {i: list(X[:, i]) for i in range(X.shape[1])}
        my_dict['class'] = list(['1' if v else '0' for v in y])
        tree = mine_c45(my_dict, 'class')
        rule_set = tree_to_my_rules(tree)
        return rule_set


def tree_to_my_rules(tree):
    rulelist = __tree_to_rules(tree)
    myruleset = []
    for rule in rulelist:
        myconstraints = {}
        constraints = rule.split(";")
        for constraint in constraints:
            v = constraint.split('-->')
            if v == ['']:
                continue
            attrindex, val = v[0], v[1]
            # convert attrindex to int if possible
            if attrindex != 'class':
                attrindex = int(attrindex)

            # convert val to str
            val = str(val)

            if attrindex == 'class' and val == '0':
                continue
            if attrindex == 'class' and val == '1':
                myrule = Rule(myconstraints)
                myruleset.append(myrule)
            else:
                myconstraints[attrindex] = DiscreteConstraint(index=attrindex, value=val)
    return myruleset


def __tree_to_rules(tree, rule=''):
    rules = []
    for node in tree:
        if isinstance(node, str):
            rule += node + ';'
        else:
            rules += __tree_to_rules(node, rule)
    if rules:
        return rules
    return [rule]


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
            tree.append(['%s-->%s' % (col, v),  # was '%s=%s' %
                         '%s-->%s' % (result, subt[result][0])])  # also here
        else:
            del subt[col]
            tree.append(['%s-->%s' % (col, v)] + mine_c45(subt, result))  # also here
    return tree


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
    return {k: [v[i] for i in ind] for k, v in t.items()}
    # return {k: [v[i] for i in range(len(v)) if i in ind] for k, v in t.items()}


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
