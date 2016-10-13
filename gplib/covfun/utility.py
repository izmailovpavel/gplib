"""
Utility functions, related to covariances
"""
import numpy as np
from scipy.spatial.distance import cdist

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(- x))


def delta(r):
    """Delta-function"""
    if np.all(np.diag(r) == 0):
        return np.eye(r.shape[1])
    return np.zeros((r.shape))


def gaussian_noise_term(noise_variance, r):
    return noise_variance**2 * delta(r)


def pairwise_distance(x, y):
    """
    Compute a matrix of pairwise distances between x and y
    :param x: array
    :param y: array
    :return: pairwise distances matrix
    """
    return cdist(x, y)


def stationary_cov(fun):
    def wrapper(self, x, y, *args, **kwargs):
        dists = pairwise_distance(x, y)
        return fun(self, dists, *args, **kwargs)
    return wrapper