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

    def generate_data(self, tr_points, test_points, seed=None):
        """
        Generates data for classification from a gaussian process.
        :param dim: dimensions of the generated data
        :param tr_points: training data points
        :param test_points: testing data points
        :return: tuple (training data points, training labels or target values, test data points, test labels or target
        values)
        """
        if not (seed is None):
            np.random.seed(seed)
        targets = self.sample(np.vstack((tr_points, test_points)), seed)
        targets = np.sign(targets)
        targets = targets.reshape((targets.size, 1))
        return targets[:tr_points.shape[0], :], targets[tr_points.shape[0]:, :]
