import numpy as np
from abc import ABCMeta, abstractmethod
from .utility import pairwise_distance, stationary_cov


class CovarianceFamily:
    """This is an abstract class, representing the concept of a family of covariance functions"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def covariance_function(self, x, y, w=None):
        """
        A covariance function
        :param x: vector
        :param y: vector
        :param w: hyper-parameters vector of the covariance functions' family
        :return: the covariance between the two vectors
        """
        pass

    @staticmethod
    @abstractmethod
    def get_bounds():
        """
        :return: The bouns on the hyper-parameters
        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        A setter function for the hyper-parameters
        :param params: a vector of hyper-parameters
        :return: CovarianceFamily object
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        A getter function for the hyper-parameters
        :param params: a vector of hyper-parameters
        :return: CovarianceFamily object
        """
        pass

    @abstractmethod
    def get_derivative_function_list(self, params):
        """
        :return: a list of functions, which produce the derivatives of the covariance matrix with respect to
        hyper-parameters except for the noise variance, when given to the covariance_matrix() function
        """
        pass

    @abstractmethod
    def covariance_derivative(self, x, y):
        """derivative wrt x"""

    def get_noise_derivative(self, points_num):
        """
        :return: the derivative of the covariance matrix w.r.t. to the noise variance.
        """
        return 2 * self.get_params()[-1] * np.eye(points_num)

    def __call__(self, x, y, w=None):
        return self.covariance_function(x, y, w)


class StationaryCovarianceFamily(CovarianceFamily):
    """This is an abstract class, representing the concept of a family of stationary covariance functions"""
    __metaclass__ = ABCMeta

    def covariance_function(self, x, y, w=None):
        return self.st_covariance_function(pairwise_distance(x, y), w)

    @abstractmethod
    def st_covariance_function(self, d, w=None):
        pass
