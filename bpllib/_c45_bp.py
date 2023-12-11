from bpllib._abstract_bayes_point import BayesPointClassifier
from bpllib.rules._discrete_constraint import DiscreteConstraint
from bpllib.rules._rule import Rule
import copy
from copy import deepcopy
from typing import List
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import xml.dom.minidom


# implementation taken from the imodels repo (and adapted)


def get_paths_from_c45_tree(root, current_path="", paths=None):
    """
    Recursively traverse a C4.5 decision tree and return a list of all possible paths.

    :param root: The root node of the tree as an `xml.dom.minidom.Element` object.
    :param current_path: The current path in the tree (defaults to an empty string).
    :param paths: The list of paths found so far (defaults to an empty list).
    :return: A list of all possible paths in the tree.
    """
    if paths is None:
        paths = []

    if not root.childNodes:
        # Leaf node
        paths.append(current_path)
        return paths

    for child in root.childNodes:
        if child.nodeName == "node":
            # Internal node
            attribute = child.getAttribute("var")
            for value_node in child.childNodes:
                if value_node.nodeName == "edge":
                    value = value_node.getAttribute("name")
                    new_path = current_path + f"{attribute}={value},"
                    paths = get_paths_from_c45_tree(value_node, new_path, paths)
        elif child.nodeName == "leaf":
            # Leaf node
            class_name = child.getAttribute("class")
            new_path = current_path + f"class={class_name},"
            paths.append(new_path)

    return paths


class C45Classifier(BayesPointClassifier):
    '''
    This class implements the BRS algorithm.
    '''
    description = 'BRS'

    def __init__(self, max_rules=2000, **kwargs):
        self.max_rules = max_rules

        super().__init__(**kwargs)

        self.to_string = False
        self.use_bootstrap = True

    def base_method(self, X, y, target_class):
        clf = C45TreeClassifier(max_rules=self.max_rules)

        clf.fit(X, y)

        # travel the tree (clf.root, is a DOM Element) and get the rules
        rules = extract_paths_from_dom_tree(clf.dom_.documentElement)

        my_rules = []
        for rule in rules:
            idx_val = [(int(attribute.replace("X", "")), val) for attribute, val in rule]
            my_rule = Rule(
                constraints={idx: DiscreteConstraint(value=val, index=idx) for idx, val in idx_val})
            my_rules.append(my_rule)

        return my_rules


def _add_label(node, label):
    if hasattr(node, "labels"):
        node.labels.append(label)
        return
    node.labels = [label]
    return


def _get_next_node(children, att):
    for child in children:
        is_equal = child.getAttribute("flag") == "m" and child.getAttribute("feature") == att
        is_less_than = child.getAttribute("flag") == "l" and float(att) < float(child.getAttribute("feature"))
        is_greater_than = child.getAttribute("flag") == "r" and float(att) >= float(child.getAttribute("feature"))
        if is_equal or is_less_than or is_greater_than:
            return child


def shrink_node(node, reg_param, parent_val, parent_num, cum_sum, scheme, constant):
    """Shrink the tree
    """

    is_leaf = not node.hasChildNodes()
    # if self.prediction_task == 'regression':
    val = node.nodeValue
    is_root = parent_val is None and parent_num is None
    n_samples = len(node.labels) if (scheme != "leaf_based" or is_root) else parent_num

    if is_root:
        val_new = val

    else:
        reg_term = reg_param if scheme == "constant" else reg_param / parent_num

        val_new = (val - parent_val) / (1 + reg_term)

    cum_sum += val_new

    if is_leaf:
        if scheme == "leaf_based":
            v = constant + (val - constant) / (1 + reg_param / node.n_obs)
            node.nodeValue = v
        else:
            node.nodeValue = cum_sum

    else:
        for c in node.childNodes:
            shrink_node(c, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)

    return node


def check_fit_arguments(model, X, y, feature_names):
    """Process arguments for fit and predict methods.
    """

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            model.feature_names_ = X.columns
        elif isinstance(X, list):
            model.feature_names_ = ['X' + str(i) for i in range(len(X[0]))]
        else:
            model.feature_names_ = ['X' + str(i) for i in range(X.shape[1])]
    else:
        model.feature_names_ = feature_names

    X, y = check_X_y(X, y, dtype=None)  # had to fix this
    _, model.n_features_in_ = X.shape
    assert len(model.feature_names_) == model.n_features_in_, 'feature_names should be same size as X.shape[1]'
    y = y.astype(float)
    return X, y, model.feature_names_


import xml.dom.minidom


def extract_paths_from_dom_tree(root, positive_class='1', p=0.5):
    paths = []
    stack = [(root, [])]

    while stack:
        node, path = stack.pop()
        if node.nodeType == xml.dom.Node.ELEMENT_NODE:
            if node.hasAttribute("feature"):
                name = node.nodeName
                feature = node.getAttribute("feature")
                val = node.getAttribute("p")
                path.append((name, feature))

            children = node.getElementsByTagName("*")
            if children:
                for child in children:
                    stack.append((child, path.copy()))
            elif node.firstChild.data == "1.0" and float(node.getAttribute("p")) >= p:
                paths.append(path)

    return paths


def get_paths(node, path=None):
    if path is None:
        path = []
    paths = []
    if len(node.childNodes) == 0:
        return [path]
    else:
        for child in node.childNodes:
            if isinstance(child, xml.dom.minidom.Element):
                feature = child.getAttribute('feature')
                val = child.getAttribute('p')
                path_copy = path.copy()
                path_copy.append((feature, val))
                paths += get_paths(child, path_copy)
    return paths


def get_paths_from_dom_tree2(root):
    def get_paths_helper(node, path):
        if not node.hasChildNodes():
            return [path]
        else:
            paths = []
            for child in node.childNodes:
                if child.nodeType == xml.dom.Node.ELEMENT_NODE:
                    child_path = path + [(child.getAttribute("feature"), child.getAttribute("p"))]
                    paths.extend(get_paths_helper(child, child_path))
            return paths

    paths = []
    for child in root.childNodes:
        if child.nodeType == xml.dom.Node.ELEMENT_NODE:
            path = [(child.getAttribute("feature"), child.getAttribute("p"))]
            paths.extend(get_paths_helper(child, path))
    return paths


class C45TreeClassifier(BaseEstimator, ClassifierMixin):
    """A C4.5 tree classifier.

    Parameters
    ----------
    max_rules : int, optional (default=None)
        Maximum number of split nodes allowed in the tree
    """

    def __init__(self, max_rules: int = None):
        super().__init__()
        self.max_rules = max_rules

    def fit(self, X, y, feature_names: str = None):
        self.complexity_ = 0
        # X, y = check_X_y(X, y)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        self.resultType = type(y[0])
        if feature_names is None:
            self.feature_names = [f'X_{x}' for x in range(X.shape[1])]
        else:
            # only include alphanumeric chars / replace spaces with underscores
            self.feature_names = [''.join([i for i in x if i.isalnum()]).replace(' ', '_')
                                  for x in feature_names]
            self.feature_names = [
                'X_' + x if x[0].isdigit()
                else x
                for x in self.feature_names
            ]

        assert len(self.feature_names) == X.shape[1]

        data = [[] for i in range(len(self.feature_names))]
        categories = []

        for i in range(len(X)):
            categories.append(str(y[i]))
            for j in range(len(self.feature_names)):
                data[j].append(X[i][j])
        root = ET.Element('GreedyTree')
        self.grow_tree(data, categories, root, self.feature_names)  # adds to root
        self.tree_ = ET.tostring(root, encoding="unicode")
        # print('self.tree_', self.tree_)
        self.dom_ = minidom.parseString(self.tree_)
        return self

    def impute_nodes(self, X, y):
        """
        Returns
        ---
        the leaf by which this sample would be classified
        """
        source_node = self.root
        for i in range(len(y)):
            sample, label = X[i, ...], y[i]
            _add_label(source_node, label)
            nodes = [source_node]
            while len(nodes) > 0:
                node = nodes.pop()
                if not node.hasChildNodes():
                    continue
                else:
                    att_name = node.firstChild.nodeName
                    if att_name != "#text":
                        att = sample[self.feature_names.index(att_name)]
                        next_node = _get_next_node(node.childNodes, att)
                    else:
                        next_node = node.firstChild
                    _add_label(next_node, label)
                    nodes.append(next_node)

        self._calc_probs(source_node)
        # self.dom_.childNodes[0] = source_node
        # self.tree_.source = source_node

    def _calc_probs(self, node):
        node.nodeValue = np.mean(node.labels)
        if not node.hasChildNodes():
            return
        for c in node.childNodes:
            self._calc_probs(c)

    def raw_preds(self, X):
        check_is_fitted(self, ['tree_', 'resultType', 'feature_names'])
        X = check_array(X)
        if isinstance(X, pd.DataFrame):
            X = deepcopy(X)
            X.columns = self.feature_names
        root = self.root
        prediction = []
        for i in range(X.shape[0]):
            answerlist = decision(root, X[i], self.feature_names, 1)
            answerlist = sorted(answerlist.items(), key=lambda x: x[1], reverse=True)
            answer = answerlist[0][0]
            # prediction.append(self.resultType(answer))
            prediction.append(float(answer))

        return np.array(prediction)

    def predict(self, X):
        raw_preds = self.raw_preds(X)
        return (raw_preds > np.ones_like(raw_preds) * 0.5).astype(int)

    def predict_proba(self, X):
        raw_preds = self.raw_preds(X)
        return np.vstack((1 - raw_preds, raw_preds)).transpose()

    def __str__(self):
        check_is_fitted(self, ['tree_'])
        return self.dom_.toprettyxml(newl="\r\n")

    def grow_tree(self, X_t: List[list], y_str: List[str], parent, attrs_names):
        """
        Parameters
        ----------
        X_t: List[list]
            input data transposed (num_features x num_observations)
        y_str: List[str]
            outcome represented as strings

        parent
        attrs_names

        """
        # check that y contains more than 1 distinct value
        if len(set(y_str)) > 1:
            split = []

            # loop over features and build up potential splits
            for i in range(len(X_t)):
                if set(X_t[i]) == set("?"):
                    split.append(0)
                else:
                    if is_numeric_feature(X_t[i]):
                        split.append(gain(y_str, X_t[i]))
                    else:
                        split.append(gain_ratio(y_str, X_t[i]))

            # no good split, return child node
            if max(split) == 0:
                set_as_leaf_node(parent, y_str)

            # there is a good split
            else:
                index_selected = split.index(max(split))
                name_selected = str(attrs_names[index_selected])
                self.complexity_ += 1
                if is_numeric_feature(X_t[index_selected]):
                    # split on this point
                    split_point = get_best_split(y_str, X_t[index_selected])

                    # build up children nodes
                    r_child_X = [[] for i in range(len(X_t))]
                    r_child_y = []
                    l_child_X = [[] for i in range(len(X_t))]
                    l_child_y = []
                    for i in range(len(y_str)):
                        if not X_t[index_selected][i] == "?":
                            if float(X_t[index_selected][i]) < float(split_point):
                                l_child_y.append(y_str[i])
                                for j in range(len(X_t)):
                                    l_child_X[j].append(X_t[j][i])
                            else:
                                r_child_y.append(y_str[i])
                                for j in range(len(X_t)):
                                    r_child_X[j].append(X_t[j][i])

                    # grow child nodes as well
                    if len(l_child_y) > 0 and len(r_child_y) > 0 and (
                            self.max_rules is None or
                            self.complexity_ <= self.max_rules
                    ):
                        p_l = float(len(l_child_y)) / (len(X_t[index_selected]) - X_t[index_selected].count("?"))
                        son = ET.SubElement(parent, name_selected,
                                            {'feature': str(split_point), "flag": "l", "p": str(round(p_l, 3))})
                        self.grow_tree(l_child_X, l_child_y, son, attrs_names)
                        son = ET.SubElement(parent, name_selected,
                                            {'feature': str(split_point), "flag": "r", "p": str(round(1 - p_l, 3))})
                        self.grow_tree(r_child_X, r_child_y, son, attrs_names)
                    else:
                        num_max = 0
                        for cat in set(y_str):
                            num_cat = y_str.count(cat)
                            if num_cat > num_max:
                                num_max = num_cat
                                most_cat = cat
                        parent.text = most_cat
                else:
                    # split on non-numeric variable (e.g. categorical)
                    # create a leaf for each unique value
                    for k in set(X_t[index_selected]):
                        if not k == "?" and (
                                self.max_rules is None or
                                self.complexity_ <= self.max_rules
                        ):
                            child_X = [[] for i in range(len(X_t))]
                            child_y = []
                            for i in range(len(y_str)):
                                if X_t[index_selected][i] == k:
                                    child_y.append(y_str[i])
                                    for j in range(len(X_t)):
                                        child_X[j].append(X_t[j][i])
                            son = ET.SubElement(parent, name_selected, {
                                'feature': k, "flag": "m",
                                'p': str(round(
                                    float(len(child_y)) / (
                                            len(X_t[index_selected]) - X_t[index_selected].count("?")),
                                    3))})
                            self.grow_tree(child_X, child_y, son, attrs_names)
        else:
            parent.text = y_str[0]

    @property
    def root(self):
        return self.dom_.childNodes[0]


import math


def prettify(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            prettify(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem


def is_numeric_feature(attr):
    for x in set(attr):
        if not x == "?":
            try:
                x = float(x)
                return isinstance(x, float)
            except ValueError:
                return False
    return True


def set_as_leaf_node(parent, y_str):
    num_max = 0
    for cat in set(y_str):
        num_cat = y_str.count(cat)
        if num_cat > num_max:
            num_max = num_cat
            most_cat = cat
    parent.text = most_cat


def entropy(x):
    ent = 0
    for k in set(x):
        p_i = float(x.count(k)) / len(x)
        ent = ent - p_i * math.log(p_i, 2)
    return ent


def gain_ratio(category, attr):
    s = 0
    cat = []
    att = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cat.append(category[i])
            att.append(attr[i])
    for i in set(att):
        p_i = float(att.count(i)) / len(att)
        cat_i = []
        for j in range(len(cat)):
            if att[j] == i:
                cat_i.append(cat[j])
        s = s + p_i * entropy(cat_i)
    gain = entropy(cat) - s
    ent_att = entropy(att)
    if ent_att == 0:
        return 0
    else:
        return gain / ent_att


def gain(category, attr):
    cats = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cats.append([float(attr[i]), category[i]])
    cats = sorted(cats, key=lambda x: x[0])

    cat = [cats[i][1] for i in range(len(cats))]
    att = [cats[i][0] for i in range(len(cats))]
    if len(set(att)) == 1:
        return 0
    else:
        gains = []
        div_point = []
        for i in range(1, len(cat)):
            if not att[i] == att[i - 1]:
                gains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1 - float(i) / len(cat)))
                div_point.append(i)
        gain = entropy(cat) - min(gains)

        p_1 = float(div_point[gains.index(min(gains))]) / len(cat)
        ent_attr = -p_1 * math.log(p_1, 2) - (1 - p_1) * math.log((1 - p_1), 2)
        return gain / ent_attr


def get_best_split(category, attr):
    cats = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cats.append([float(attr[i]), category[i]])
    cats = sorted(cats, key=lambda x: x[0])

    cat = [cats[i][1] for i in range(len(cats))]
    att = [cats[i][0] for i in range(len(cats))]
    gains = []
    split_point = []
    for i in range(1, len(cat)):
        if not att[i] == att[i - 1]:
            gains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1 - float(i) / len(cat)))
            split_point.append(i)
    return att[split_point[gains.index(min(gains))]]


def add(d1, d2):
    d = d1
    for i in d2:
        if d.has_key(i):
            d[i] = d[i] + d2[i]
        else:
            d[i] = d2[i]
    return d


def decision(root, obs, feature_names: list, p):
    if root.hasChildNodes():
        att_name = root.firstChild.nodeName
        if att_name == "#text":
            return decision(root.firstChild, obs, feature_names, p)
        else:
            att = obs[feature_names.index(att_name)]
            if att == "?":
                d = {}
                for child in root.childNodes:
                    d = add(d, decision(child, obs, feature_names, p * float(child.getAttribute("p"))))
                return d
            else:
                for child in root.childNodes:
                    if child.getAttribute("flag") == "m" and child.getAttribute("feature") == att or \
                            child.getAttribute("flag") == "l" and float(att) < float(child.getAttribute("feature")) or \
                            child.getAttribute("flag") == "r" and float(att) >= float(child.getAttribute("feature")):
                        return decision(child, obs, feature_names, p)
    else:
        return {root.nodeValue: root.parentNode.getAttribute('p')}
