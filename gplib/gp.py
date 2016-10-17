import numpy as np
import numbers
from abc import ABCMeta, abstractmethod

from .covfun import CovarianceFamily


class GP:
    """
    An abstract class, a base class for Gaussian Process
    """
    __metaclass__ = ABCMeta

    def __init__(self, cov_obj, mean_function=lambda x: 0):
        """
        :param cov_obj: object of the CovarianceFamily class
        :param mean_function: function, mean of the gaussian process
        """
        if not isinstance(cov_obj, CovarianceFamily):
            raise TypeError("The covariance object cov_obj is of the wrong type")
        if not hasattr(mean_function, '__call__'):
            raise TypeError("mean_function must be callable")

        self.cov = cov_obj
        self.mean = mean_function

    def _sample_normal(self, X, seed=None):
        """
        Samples Gaussian process values at given points X
        :param X: points
        :param seed: random seed
        """
        cov_mat = self.cov(X, X)
        mean = np.array([self.mean(point) for point in X.tolist()]).reshape(-1)
        if not (seed is None):
            np.random.seed(seed)
        res = np.random.multivariate_normal(mean, cov_mat)
        return res

    @abstractmethod
    def sample(self, X, seed=None):
        """
        Samples values from the particular model at given points X
        :param X: points
        :param seed: random seed
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fits the Gaussian process to the data
        :param X: train data points
        :param y: target values
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predicts the answers at test points
        :param X_test: test data points
        """
        pass

