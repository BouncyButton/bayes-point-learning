#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" aq_algorithm (doesn't work)...
Authors: Jakub Ciemięga, Klaudia Stpiczyńska
"""

import numpy as np
import pandas as pd
import pickle
import copy

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels


class AqClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, num_best=100, quality_index_type=0):
        self.num_best = num_best
        self.quality_index_type = quality_index_type

    def _more_tags(self):
        return {'X_types': ['2darray', 'string'], 'requires_y': True}

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        df = pd.DataFrame(X)
        df['class'] = y
        self.rules = induce_rules(df, self.num_best, self.quality_index_type)

        return self

    def predict(self, X):
        return predict_table(self.rules, X)


def covers(c: dict, row: pd.Series) -> bool:
    """ Check if given complex covers specific instance/row of pd.DataFrame

    :param c: complex in form: {'col_name1':[<list of possible values>], ...}. Empty complex returns True
    :type c: dict

    :param row: instance of a data frame
    :type row: pd.Series

    :return: True if complex covers the given row or False if it doesn't
    :rtype: bool
    """
    # if any column value isn't in the complex, row is not covered
    for col in c.keys():
        if not row[col] in c[col]:
            return False
    return True


def keep_covered(complexes: list, df: pd.DataFrame) -> pd.DataFrame:
    """ Return data frame containing only rows of df covered by any of the complexes

    :param complexes: complexes in form: [{'col_name1':[<list of possible values>], ...}, {...}].
    :type complexes: list of dict

    :param df: instance of a data frame
    :type df: pd.DataFrame

    :return df_new: data frame containing rows of df covered by any of the complexes
    :rtype df_new: pd.DataFrame
    """
    indexes = []
    for r in range(df.shape[0]):
        for c in complexes:
            if covers(c, df.iloc[r]):
                indexes.append(r)
                break
    return df.iloc[indexes]


def delete_covered(rule: dict, df: pd.DataFrame) -> pd.DataFrame:
    """ Return data frame containing only rows of df not covered by a rule

    :param rule: rule in form: {'col_name1':[<list of possible values>], ..., 'class_value':'<value>'}.
    :type rule: dict

    :param df: instance of a data frame
    :type df: pd.DataFrame

    :return df_new: data frame containing rows of df not covered by the rule
    :rtype df_new: pd.DataFrame
    """
    indexes = []
    # make complex from rule for covers() function
    # complex = copy.deepcopy(rule)
    c = rule.copy()
    del c['class_value']
    for r in range(df.shape[0]):
        if not covers(c, df.iloc[r]):
            indexes.append(r)
    return df.iloc[indexes]


def delete_non_general(complexes: list) -> list:
    """ Return a list of complexes containing only the most general ones

    :param complexes: list of complexes to perform the action on
    :type complexes: list

    :return complexes_new: list of complexes containing only general ones
    :rtype complexes_new: list of dict
    """
    for c in complexes.copy():
        # create list with all other complexes and compare them with the given complex to determine if c is most general
        complexes_rest = complexes.copy()
        del complexes_rest[complexes_rest.index(c)]
        for c_r in complexes_rest:
            # if c_r contains a key that c doesn't, c_r is more specific for this key, so it is not more general than c
            if any(key not in c.keys() for key in c_r.keys()):
                continue
            cont_flag = False
            # if the loop not interrupted, c has more or equal number of keys compared with c_r
            # now if there is a key in c_r with an element not in the same key of c, c_r is not more general than c
            for key in c_r.keys():
                if not all(el in c_r[key] for el in c[key]):
                    cont_flag = True
                    break
            if cont_flag:
                continue
            # if the loop not interrupted, c not general, so it can be deleted and doesn't have to be further compared
            del complexes[complexes.index(c)]
            break

    complexes_new = complexes
    return complexes_new


def keep_best(df_pos: pd.DataFrame, df_neg: pd.DataFrame, complexes: list, num_best: int = 2,
              quality_index_type: int = 0) -> list:
    """ Return a list containing only num_best best complexes

    :param df_pos: data frame containing only instances with the same class as positive kernel
    :type df_pos: pd.DataFrame

    :param df_neg: data frame containing only instances different class from positive kernel
    :type df_neg: pd.DataFrame

    :param complexes: list of complexes to perform the action on
    :type complexes: list

    :param num_best: number of complexes to keep in each iteration
    :type num_best: int

    :param quality_index_type: determines what kind of quality measurement to use. 0 - covering, 1 - accuracy. Formulas
                               according to project documentation
    :type quality_index_type: int

    :return complexes_new: list of complexes containing only best ones
    :rtype complexes_new: list of dict
    """
    # if not too many complexes, return all
    if len(complexes) <= num_best:
        return complexes

    scores = []
    for c in complexes:
        if 0 == quality_index_type:
            score = 0
            # add number of covered rows with correct class
            for r in range(df_pos.shape[0]):
                if covers(c, df_pos.iloc[r]):
                    score += 1
            # add number of not covered rows with incorrect class
            score += df_neg.shape[0]
            for r in range(df_neg.shape[0]):
                if covers(c, df_neg.iloc[r]):
                    score -= 1
        elif 1 == quality_index_type:
            numerator = 1
            # add number of covered rows with correct class
            for r in range(df_pos.shape[0]):
                if covers(c, df_pos.iloc[r]):
                    numerator += 1
            # create denominator for formula, num of classes in df_neg = 1 and cancels out -1 from formula
            denominator = numerator + df_neg.iloc[:, -1].unique().size
            for r in range(df_neg.shape[0]):
                if covers(c, df_neg.iloc[r]):
                    denominator += 1
            score = numerator / denominator
        scores.append(score)

    # find best complexes based on scores
    complexes_new = []
    for i in range(num_best):
        best_ind = scores.index(max(scores))
        complexes_new.append(complexes[best_ind])
        del scores[best_ind]
        del complexes[best_ind]

    return complexes_new


def aq_specialization(df: pd.DataFrame, num_best: int, quality_index_type: int) -> dict:
    """ Perform aq specialisation on a given dataset and return one rule

    :param df: dataframe to base the rule on
    :type df: pd.DataFrame

    :param num_best: number of complexes to keep in each iteration
    :type num_best: int

    :param quality_index_type: determines what kind of quality measurement to use. 0 - covering, 1 - accuracy. Formulas
                               according to project documentation
    :type quality_index_type: int

    :return rule: rule for df in form: {'target_class':<class>, 'col_name1':[<list of possible values>], ...}
    :rtype rule: dict
    """

    # choose positive kernel
    kernel_pos = df.index[0]
    target_class = df[df.columns[-1]][kernel_pos]

    # df with target class the same as positive kernel
    df_pos = df[df[df.columns[-1]] == target_class]
    # df with target class different from positive kernel
    df_neg = df[df[df.columns[-1]] != target_class]
    df_neg_original = df_neg.copy()

    # create new empty list of complexes
    complexes = [{}]
    # counter to make sure algorithm not stuck
    count = 0
    while not df_neg.empty:
        # print(f'iteration aq: {count}')
        count += 1
        # choose negative kernel
        kernel_neg = df_neg.index[0]
        # print(f'df_pos:\n {df_pos}')
        # print(f'df_neg:\n {df_neg}')

        # if all attributes of positive and negative kernel are the same, specialization not possible, skip kernel_neg
        if df.loc[kernel_pos].iloc[:-1].equals(df.loc[kernel_neg].iloc[:-1]):
            df_neg = df_neg.drop(index=kernel_neg)
            continue

        # specialization loop, c is a dict
        for c in copy.deepcopy(complexes):  # use copy since changing complexes in the loop
            if covers(c, df.loc[kernel_neg]):  # specialize only if kernel_neg covered by given complex
                del complexes[complexes.index(c)]  # delete complex since it will be replaced with specialization
                # crate specialization for each column where positive and negative kernels are different
                for col in df.columns[:-1]:
                    if df.loc[kernel_pos, col] != df.loc[kernel_neg, col]:
                        c_new = copy.deepcopy(c)
                        # if column wasn't in the complex, add it with all possible values
                        if col not in c_new.keys():
                            c_new[col] = list(df[col].unique())
                        # delete attribute of negative kernel from the complex specialization
                        c_new[col].remove(df.loc[kernel_neg, col])
                        # add specialized complex to complex list
                        complexes.append(c_new)

        print(complexes)
        # keep most general complexes
        complexes = delete_non_general(complexes)
        # print(complexes)
        # keep  only num_best complexes
        complexes = keep_best(df_pos, df_neg_original, complexes, num_best, quality_index_type)
        # print(complexes)
        # in df_neg only keep rows covered by current complexes
        df_neg = keep_covered(complexes, df_neg)

    # when no more negative seeds available, pick best (first) complex and make it a rule
    rule = complexes[0]
    # assign the class of kernel as class_value for rule
    rule['class_value'] = df[df.columns[-1]][kernel_pos]
    return rule


def induce_rules(df: pd.DataFrame, num_best: int, quality_index_type: int = 0) -> list:
    """ generate a set of rules using the aq algorithm based on df.

    :param df: dataframe to base the rules on
    :type df: pd.DataFrame

    :param num_best: number of complexes to keep in each iteration of the aq algorithm
    :type num_best: int

    :param quality_index_type: determines what kind of quality measurement to use in aq algorithm. 0 - covering,
                                1 - accuracy. Formulas according to project documentation
    :type quality_index_type: int

    :return rules: set of rules based on df
    :rtype rules: list of dict
    """
    # drop duplicates for rule induction
    df = df.drop_duplicates(ignore_index=True)

    # initialize empty rules list
    rules = []

    # counter to make sure algorithm not stuck
    count = 0

    while not df.empty:
        # print number of remaining rows to see have many rows covered by each rule
        if count % 5 == 0:
            print(f'rule induction iteration: {count}, df rows not covered: {df.shape[0]}')
        count += 1
        # create and append new rule
        new_rule = aq_specialization(df, num_best, quality_index_type)
        rules.append(new_rule)
        # delete rows from df covered by new rule
        df = delete_covered(new_rule, df)

    return rules


def use_aq_simple(fname: str, target_col_num: int, headers: bool, split: float, num_best: int,
                  quality_index_type: int) -> None:
    """ generate a set of rules using the aq algorithm based on data from a specified file and tests its accuracy

    :param fname: name of the file containing the data
    :type fname: str

    :param target_col_num: number of target column, starting from 0
    :type target_col_num: int

    :param headers: decides whether file under fname contains column names (True) or not (False)
    :type headers: bool

    :param split: decides what fraction of dataset used for training, the rest used for testing
    :type split: float

    :param num_best: number of complexes to keep in each iteration of the aq algorithm
    :type num_best: int

    :param quality_index_type: determines what kind of quality measurement to use in aq algorithm. 0 - covering,
                                1 - accuracy. Formulas according to project documentation
    :type quality_index_type: int

    :return rules: set of rules based on df
    :rtype rules: list of dict
    """
    # read df from file
    if headers:
        df = pd.read_csv(fname, header=0)
    else:
        df = pd.read_csv(fname, header=None)

    # move target column to last column
    columns = list(df.columns)
    try:
        target_col = columns[target_col_num]
    except IndexError:
        raise Exception('Requested target column index higher than total column number')
    del columns[target_col_num]
    columns.append(target_col)
    df = df[columns]

    # shuffle for better training
    df = df.sample(frac=1).reset_index(drop=True)

    print(df)

    # induce rules using the aq algorithm
    rules = induce_rules(df.loc[:int(df.shape[0] * split)], num_best, quality_index_type)

    df_pred = predict_table(rules, df.loc[int(df.shape[0] * split):], 'pred')
    # print(df_pred)
    print(f'Accuracy: {np.round(np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1]) / df_pred.shape[0] * 100, 3)}%')

    return


def predict_record(rules: list, row: pd.Series):
    """ predict outcome for a data series by using set of rules

    :param rules: rules induced with aq algorithm for classification
    :type rules: list of dict

    :param row: series to base the classification on
    :type row: pd.Series

    :return: predicted value for the series
    :rtype: dependent on predicted value
    """
    # for each rule check if it covers the row and if so, assign value of the rule
    for rule in rules:
        complex = rule.copy()
        del complex['class_value']
        if covers(complex, row):
            return rule['class_value']
    return np.NaN


# for confusion matrix
# def predict_table(possible_values, rules: list, df: pd.DataFrame, col_name: str = 'predicted') -> pd.DataFrame:
def predict_table(rules: list, df: pd.DataFrame, col_name: str = 'predicted') -> pd.DataFrame:
    """ Predict values for a data set using a set of rules

    :param rules: rules induced with aq algorithm for classification
    :type rules: list of dict

    :param df: data set to base the classification on
    :type df: pd.DataFrame

    :param col_name: name of the column for predicted values; if nonexistent, added
    :type col_name: str

    :return df_predict: df with added column of predictions
    :rtype df_predict: pd.DataFrame
    """
    # for confusion matrix
    # df1 = pd.DataFrame(0, index=possible_values, columns=possible_values)
    # create a copy of df and create new empty column with the target name
    df = df.copy()
    df[col_name] = np.nan
    # use rules to predict each record
    for i in range(df.shape[0]):
        df.iloc[i, -1] = predict_record(rules, df.iloc[i])
        if isinstance(df.iloc[i, -1], float):
            string_pred = str(int(df.iloc[i, -1]))
            str_actual = str(int(df.iloc[i, -2]))
        else:
            string_pred = str(df.iloc[i, -1])
            str_actual = str(df.iloc[i, -2])
        # df1.loc[string_pred, str_actual] += 1
    # print(df1)

    df_predict = df
    return df_predict


def save_rules_to_file(rules: list, fname: str = 'rules') -> None:
    """ save set of rules to a .pkl file

    :param rules: rules to save
    :type rules: list of dict

    :param fname: name of the file, the rules are saved in <fname>.pkl
    :type fname: str

    :return: None
    """
    fid = open(fname + ".pkl", "wb")
    pickle.dump(rules, fid)
    fid.close()


def load_rules_from_file(fname: str = 'rules') -> dict:
    """ load set of rules from a .pkl file

    :param fname: name of the file, the rules are loaded from <fname>.pkl
    :type fname: str

    :return rules: set of rules
    :rtype rules: list of dict
    """
    fid = open(fname + ".pkl", "rb")
    return pickle.load(fid)
