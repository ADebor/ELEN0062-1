"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary

from scipy import stats

class residual_fitting(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        """Fit a Residual fitting model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation

        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.classes = np.unique(y)

        # Prewithening

        for i in np.arange(np.shape(X)[1]):
                X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])


        X_prime = np.append(np.ones((np.shape(X)[0],1)),X, axis=1)

        # Initialization of the first weight

        self.weights = np.asarray(np.mean(y))

        index = np.arange(np.shape(X)[1])

        # Residual fitting algorithm

        for i in index:
            resid = y - (self.weights*X_prime[:,:i+1]).sum(axis=1)
            best_w = stats.pearsonr(X_prime[:,i+1], resid)[0]*np.std(resid)
            self.weights = np.append(self.weights, best_w)

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        # Prediction of the classes on the basis of the predicted class probabilities of the input samples

        proba = self.predict_proba(X)
        binary_predict = [np.argmax(item) for item in proba]
        y = [self.classes[i] for i in binary_predict]

        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # Linear approximation of the output

        X_prime = np.append(np.ones((np.shape(X)[0],1)),X, axis=1)
        y_hat = (self.weights*X_prime).sum(axis=1)

        # Predicted class probabilities

        p = np.asarray([[1-val, val] for val in y_hat])

        return p

        pass

def boundary_viewer(data_fun):
    """Generate boundary plots

    Parameters
    ----------
    data_fun : data generation function

    Returns
    -------
    None
    """

    X_train, y_train, X_test, y_test = data_fun()
    clf = residual_fitting().fit(X_train, y_train)

    plot_boundary("{}_resid_bound".format(data_fun.__name__), clf, X_test, y_test, mesh_step_size=0.1, title="Decision boundary of the R.F. model, from {}".format(data_fun.__name__), inline_plotting=True)
    print("Test accuracy for {}: {}".format(data_fun.__name__, clf.score(X_test, y_test)))

def modified_boundary_viewer(data_fun):
    """Generate boundary plots (for the extended data set)

    Parameters
    ----------
    data_fun : data generation function

    Returns
    -------
    None
    """

    X_train, y_train, X_test, y_test = data_fun()

    # Extension of the data set

    X_train = np.asarray(X_train)

    X_train = np.concatenate((X_train, (np.asarray([X_train[:,0]*X_train[:,0]])).T), axis=1)
    X_train = np.concatenate((X_train, (np.asarray([X_train[:,1]*X_train[:,1]])).T), axis=1)
    X_train = np.concatenate((X_train, (np.asarray([X_train[:,0]*X_train[:,1]])).T), axis=1)

    # Fitting of the model with the extended data

    clf = residual_fitting().fit(X_train, y_train)

    plot_boundary_extended("{}_resid_bound_extended".format(data_fun.__name__), clf, X_test, y_test, mesh_step_size=0.1, title="Decision boundary of the R.F. model, from {} \n(extended learning sample)".format(data_fun.__name__), inline_plotting=True)

    # Accuracy test of the fitted model and boundary plot

    X_test = np.concatenate((X_test, (np.asarray([X_test[:,0]*X_test[:,0]])).T), axis=1)
    X_test = np.concatenate((X_test, (np.asarray([X_test[:,1]*X_test[:,1]])).T), axis=1)
    X_test = np.concatenate((X_test, (np.asarray([X_test[:,0]*X_test[:,1]])).T), axis=1)

    print("Test accuracy for {} (extended learning sample): {}".format(data_fun.__name__, clf.score(X_test, y_test)))


if __name__ == "__main__":
    from data import make_data1,make_data2
    from plot import plot_boundary, plot_boundary_extended

    # Decision boundary for the initial data set

    boundary_viewer(make_data1)
    boundary_viewer(make_data2)

    # Decision boundary for the extended data set

    modified_boundary_viewer(make_data2)
