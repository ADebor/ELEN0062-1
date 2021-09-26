"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary


# (Question 1)

def boundaries_viewer(max_depth_values, data_fun):
    """Generates boundary plots.

    Parameters
    ----------
    max_depth_values : a list containing the depth values at which the tree is pruned
    data_fun : data generation function

    Return
    ------
    None

    """

    X_train, y_train, X_test, y_test = data_fun()

    for value in max_depth_values:
        clf = DecisionTreeClassifier(max_depth = value)
        clf = clf.fit(X_train, y_train)
        plot_boundary("{}_dt_{}".format(data_fun.__name__, str(value)), clf, X_test, y_test, mesh_step_size=0.1, title="Decision boundary of the tree, from {}, with depth {}".format(data_fun.__name__, str(value)), inline_plotting=True)


def get_acc(depth, data_fun):
    """Computes the accuracy score of the model for a certain depth and a certain data set.

    Parameters
    ----------
    depth : depth value at which the tree is pruned
    data_fun : data generation function

    Return
    ------
    accuracy : accuracy score of the model for a test set

    """

    X_train, y_train, X_test, y_test = data_fun()
    clf = DecisionTreeClassifier(max_depth = depth)
    clf = clf.fit(X_train, y_train)
    y_hat_test = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat_test)
    return accuracy


if __name__ == "__main__":
    pass # Make your experiments here

    max_depth_values = [1, 2, 4, 8, None]

    # Decision boundary

    boundaries_viewer(max_depth_values, make_data1)
    boundaries_viewer(max_depth_values, make_data2)

    # Test set accuracies & std

    acc = []

    print("\nAverage accuracies and standard deviations from the first data set, over 5 generations:")
    for value in max_depth_values:
        for i in range(5):
           acc.append(get_acc(value, make_data1))
        print("Depth = {} : Average accuracy = {} and Standard deviation = {}".format(str(value), str(np.mean(acc)), str(np.std(acc))))

    print("\nAverage accuracies and standard deviations from the second data set, over 5 generations:")
    acc = []
    for value in max_depth_values:
        for i in range(5):
           acc.append(get_acc(value, make_data2))
        print("Depth = {} : Average accuracy = {} and Standard deviation = {}".format(str(value), str(np.mean(acc)), str(np.std(acc))))
