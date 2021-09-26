"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


from data import make_data1, make_data2
from plot import plot_boundary


# (Question 2)

def boundaries_viewer(neighbors_values, data_fun):
    """Generate boundary plots

    Parameters
    ----------
    neighbors_values : a list containing the values of the number of neighbors for which the algorithm runs
    data_fun : data generation function

    Return
    ------
    None

    """

    X_train, y_train, X_test, y_test = data_fun()


    for value in neighbors_values:
        clf = KNeighborsClassifier(n_neighbors = value)
        clf = clf.fit(X_train, y_train)
        plot_boundary("{}_Knn_{}".format(data_fun.__name__, str(value)), clf, X_test, y_test, mesh_step_size=0.1, title="Decision boundary of the K-neighbors model, from {}\nand with {} neighbors".format(data_fun.__name__, str(value)), inline_plotting=True)

def fold(nb_splits, dataset):
    """Performs a n-fold CV on a particular data set, for a particular number of folds

    Parameters
    ----------
    nb_splits : number of folds
    data_set : data set on which the n-fold CV is performed

    Return
    ------
    index : list of indices of test and train data, under the form (test_index, train_index)

    """
    index = np.arange(np.shape(dataset)[0])
    splits = np.split(index, nb_splits)

    index = []

    for n_fold in np.arange(nb_splits):
        index.append((splits[n_fold].tolist(),(np.concatenate([x for i,x in enumerate(splits) if i!=n_fold])).tolist()))

    return index


def neighbor_optimizer(neighbors_values, nb_splits, data_fun):
    """Optimizes the value of the n_neighbors parameter using a nb_splits-fold CV strategy

    Parameters
    ----------
    neighbors_values : a list containing the values of the number of neighbors for which the algorithm runs (range of values in which the optimal one is supposed to be found)
    nb_splits : number of folds
    data_fun : data generation function

    Return
    ------
    tuple : (optimal value of the n_neighbors parameter, corresponding optimal mean accuracy over all folds of the CV)

    """

    X, y = data_fun()[0:2]

    index = fold(nb_splits, X)
    mean = []

    for value in neighbors_values:

        acc = []
        clf = KNeighborsClassifier(n_neighbors = value)

        # n-fold cross-validation

        for test_index, train_index in index:

            X_train, y_train = list(map(X.__getitem__, train_index)), list(map(y.__getitem__, train_index))
            X_test, y_test = list(map(X.__getitem__, test_index)), list(map(y.__getitem__, test_index))

            clf = clf.fit(X_train, y_train)
            acc.append(clf.score(X_test, y_test))

        # Mean accuracy over all folds of the CV

        mean.append(np.mean(acc))

    return neighbors_values[np.argmax(mean)], max(mean)


def accuracy_plot(LS_sizes, data_fun):
    """Generates the plots of the evolution of accuracy as a function of the n-neighbor parameter and of the optimal n_neighbors value as a function of the size of the LS

    Parameters
    ----------
    LS_sizes : list of learning sample sizes
    data_fun : data generation function

    Return
    ------
    blabla

    """

    opt_neigh = []

    #plot of optimal n_neighbors as a function of the LS size

    for size in LS_sizes:

        acc = []
        neighbors_values = np.arange(1,size+1,1)

        # For a given LS size, plots of accuracy(n_neighbors)

        for value in neighbors_values:

            X_train, y_train, X_test, y_test = data_fun(n_ts=500, n_ls=size)

            clf = KNeighborsClassifier(n_neighbors = value)
            clf = clf.fit(X_train, y_train)
            acc.append(clf.score(X_test,y_test))

        plt.figure()
        plt.plot(neighbors_values,acc, '.')
        plt.title("Evolution of accuracy as a function \nof n_neighbors for LS_size = {} samples, for {}.".format(size, data_fun.__name__))
        plt.savefig("acc(n_neigh)_{}_{}.pdf".format(size, data_fun.__name__))

        opt_neigh.append(np.argmax(acc)+1)

    plt.figure()
    plt.plot(LS_sizes, opt_neigh, '.')
    plt.title("Optimal n_neighbors as a function \nof the size of the learning sample, for {}.".format(data_fun.__name__))
    plt.savefig("opt_n_neigh(LS_size)_{}.pdf".format(data_fun.__name__))


if __name__ == "__main__":
    pass # Make your experiments here

    neighbors_values = [1, 5, 10, 75, 100, 150]
    LS_sizes = [50, 200, 250, 500]

    # Decision boundary

    boundaries_viewer(neighbors_values, make_data1)
    boundaries_viewer(neighbors_values, make_data2)

    # n_neighbors optimization using 5-fold CV

    n_fold = 5
    (opt_value, mean_score) = neighbor_optimizer(np.arange(1,10,1), n_fold, make_data2)
    print("The optimal n_neighbors value is {} with a mean accuracy_score of {} over the {}-fold CV.".format(opt_value, mean_score, str(n_fold)))

    # Plot of the optimal value of n_neighbors as a function of the size of the learning sample, for each data generation function

    accuracy_plot(LS_sizes, make_data1)
    accuracy_plot(LS_sizes, make_data2)
    plt.show()
