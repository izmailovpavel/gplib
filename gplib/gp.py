import numpy as np
import numbers
from abc import ABCMeta

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

    def sample(self, points, seed=None):
        """
        :param mean_func: mean function
        :param cov_func: covariance function
        :param points: data points
        :return: sample gaussian process values at given points
        """
        if not hasattr(self.cov, '__call__'):
            raise TypeError("cov must be callable")
        if not hasattr(self.mean, '__call__'):
            raise TypeError("mean must be callable")
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")

        cov_mat = self.cov(points, points)
        m_v = np.array([self.mean(point) for point in points.tolist()])
        mean_vector = m_v.reshape(-1)
        if not (seed is None):
            np.random.seed(seed)
        res = np.random.multivariate_normal(mean_vector, cov_mat)
        return res

    @staticmethod
    def sample_for_matrices(mean_vec, cov_mat, rnd=None):
        """
        :param mean_vec: mean vector
        :param cov_mat: cavariance matrix
        :return: sample gaussian process values at given points
        """
        if not (isinstance(mean_vec, np.ndarray) and
                isinstance(cov_mat, np.ndarray)):
            raise TypeError("points must be a numpy array")
        if rnd is None or (isinstance(rnd, bool) and rnd == False):
            upper_bound = mean_vec + 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
            lower_bound = mean_vec - 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
            return mean_vec, upper_bound, lower_bound
        else:
            if isinstance(rnd, numbers.Number):
                np.random.seed(rnd)
            y = np.random.multivariate_normal(mean_vec.reshape(-1), cov_mat)
            return y