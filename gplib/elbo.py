import numpy as np
import numbers
from abc import ABCMeta, abstractmethod

from .covfun import CovarianceFamily


class ELBO:
    """
    An abstract class representing the evidence lower bound. Defines the interface of 
    the classes, corresponding to different evidence lower bounds
    """
    __metaclass__ = ABCMeta

    def __init__(self, X, y):
        """
        :param X: data points (possibly a batch for some methods)
        :param y: target values
        """
        self.X = X
        self.y = y


    @abstractmethod
    def elbo(self, w):
        """
        Returns the value and the gradient (with shape (k,)) of the evidence lower bound for given parameters
        :param w: params, that are being optimized 
        """
        pass

    @abstractmethod
    def recompute_parameters(self, n_upd):
        """
        Recomputes parameters, using analytic formulas
        :n_upd: number of recomputes
        """
        pass

    @abstractmethod
    def get_params_opt(self):
        """
        Returns a vector of current values of (numerically optimized) parameters
        """
        pass

    @abstractmethod
    def get_bounds_opt(self):
        """
        Returns the bounds on parameters for optimization methods
        """
        pass

    @abstractmethod
    def set_params_opt(self, params):
        """
        Sets the (numerically optimized) parameters equal to the given parameter vector
        :param params: vector of parameters of the elbo
        """
        pass   

    @abstractmethod
    def set_state(self, params):
        """
        Sets all parameters equal to the given parameter vector
        :param params: vector of parameters of the elbo
        """
        pass  

    @abstractmethod
    def get_state(self, params):
        """
        Returns all parameters equal to the given parameter vector
        :param params: vector of parameters of the elbo
        """
        pass  

    @abstractmethod
    def get_inputs(self):
        """
        Returns the inducing input positions, mean and covariance matrix
        of the process at these points
        """
        pass


    # @abstractmethod
    # def get_prediction_quality(self, params, X_test, y_test):
    #     """
    #     Returns prediction quality (some metric)
    #     :param params: parameters
    #     :param X_test: test data points
    #     :param y_test: test target values
    #     """
    #     pass