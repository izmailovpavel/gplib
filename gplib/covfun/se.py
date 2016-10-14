import numpy as np

from .cov_base import StationaryCovarianceFamily
from .utility import pairwise_distance, stationary_cov, gaussian_noise_term


class SE(StationaryCovarianceFamily):
    """A class, representing the squared-exponential covariance function family."""

    def __init__(self, params):
        if params.size != 3:
            raise ValueError("Wrong parameters for SquaredExponential")

        self.sigma_f = params[0]
        self.l = params[1]
        self.sigma_l = params[2]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.sigma_l])

    @staticmethod
    def get_bounds():
        return (1e-2, 1e3), (1e-2, 1e3), (1e-2, 1e3)

    def set_params(self, params):
        if params.size != 3:
            raise ValueError("Wrong parameters for SquaredExponential")

        self.sigma_f = params[0]
        self.l = params[1]
        self.sigma_l = params[2]

    def st_covariance_function(self, r, w=None):
        if w is None:
            l = self.l
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            sigma_l = w[2]
        return np.exp(-r**2 / (2*(l**2))) * sigma_f**2 + gaussian_noise_term(sigma_l, r)


    def covariance_derivative(self, x, y):
        """derivative wrt x"""
        r = pairwise_distance(x, y)
        return - np.exp(-r**2 / (2 * self.l**2)) * self.sigma_f**2 * 2 * \
            (x[:, :, None] - y[:, None, :]) / (2 * self.l**2)

    @stationary_cov
    def _dse_dl(self, r):
        return (np.exp(-r**2 / (2*(self.l**2))) * self.sigma_f**2) * (r**2 / (self.l ** 3))

    @stationary_cov
    def _dse_dsigmaf(self, r):
        return 2 * self.sigma_f * np.exp(-r**2 / (2*(self.l**2)))

    def get_derivative_function_list(self, params):
        se = SE(params)
        return [se._dse_dsigmaf, se._dse_dl]