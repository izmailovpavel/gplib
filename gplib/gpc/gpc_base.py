"""
Base Gaussian Process Classification class.
"""
import numpy as np
from ..gp import GP
from abc import ABCMeta


class GPC(GP):
    """
    Gaussian Process Classifier abstract class
    """
    __metaclass__ = ABCMeta

    def __init__(self, cov_obj, mean_function=lambda x: 0):
        """
        :param cov_obj: object of the CovarianceFamily class
        :param mean_function: function, mean of the gaussian process
        """
        super(GPC, self).__init__(cov_obj, mean_function)

    def sample(self, X, seed=None):
        """
        Samples Gaussian process values and generates binary classification labels
        at given points X
        """
        targets = self._sample_normal(X, seed)
        targets = np.sign(targets)
        return targets[:, None]

    def generate_data(self, X_tr, X_test, seed=None):
        """
        Generates data for classification from a Gaussian process.
        :param dim: dimensions of the generated data
        :param x_tr: training data points
        :param x_test: testing data points
        :param seed: random seed
        :return: tuple (training data points, training labels or target values, test data points, test labels or target
        values)
        """
        targets = self.sample(np.vstack((X_tr, X_test)), seed)
        n = X_tr.shape[0]
        return targets[:n, :], targets[n:, :]

    @staticmethod
    def get_quality(y_true, y_pred):
        """
        Accuracy metric
        """
        return 1 - np.sum(y_true != y_pred) / y_true.size
