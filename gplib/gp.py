import numpy as np
from abc import ABCMeta, abstractmethod
from GP.optim import _eig_val_correction
import numbers

class GP:
    """
    An abstract class, a base class for GPR and GPC
    """
    __metaclass__ = ABCMeta

    @staticmethod
    def sample(mean_func, cov_func, points, seed=None):
        """
        :param mean_func: mean function
        :param cov_func: covariance function
        :param points: data points
        :return: sample gaussian process values at given points
        """
        if not hasattr(cov_func, '__call__'):
            raise TypeError("cov_func must be callable")
        if not hasattr(mean_func, '__call__'):
            raise TypeError("mean_func must be callable")
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")

        cov_mat = cov_func(points, points)
        m_v = np.array([mean_func(point) for point in points.T.tolist()])
        mean_vector = m_v.reshape((m_v.size,))
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