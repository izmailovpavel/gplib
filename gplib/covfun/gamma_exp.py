import numpy as np

from .cov_family import StationaryCovarianceFamily
from .utility import pairwise_distance, stationary_cov, gaussian_noise_term


class GammaExp(StationaryCovarianceFamily):
    """A class, representing the squared-exponential covariance functions family."""

    def __init__(self, params):
        self.sigma_f = params[0]
        self.l = params[1]
        self.gamma = params[2]
        self.sigma_l = params[3]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.gamma, self.sigma_l])

    @staticmethod
    def get_bounds():
        return (1e-2, None), (1e-2, None), (1e-2, 2), (1e-2, None)

    def set_params(self, params):
        self.sigma_f = params[0]
        self.l = params[1]
        self.gamma = params[2]
        self.sigma_l = params[3]

    def st_covariance_function(self, r, w=None):
        if w is None:
            l = self.l
            sigma_f = self.sigma_f
            g = self.gamma
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            g = w[2]
            sigma_l = w[3]
        return np.exp(-np.power((r / l), g)) * np.square(sigma_f) + gaussian_noise_term(sigma_l, r)

    @stationary_cov
    def _dge_dl(self, r):
        return np.exp(-(r/self.l)**self.gamma) * self.sigma_f**2 * (self.gamma * (r/self.l)**self.gamma) / self.l

    @stationary_cov
    def _dge_dsigmaf(self, r):
        return 2 * self.sigma_f * np.exp(-(r /self.l)**self.gamma)

    @stationary_cov
    def _dge_dgamma(self, r):
        loc_var = r/self.l
        loc_var_gamma = loc_var ** self.gamma
        loc_var[loc_var == 0] = 1 # A dirty hack to avoid log(0)
        res = -self.sigma_f**2 * loc_var_gamma * np.log(loc_var) * np.exp(-loc_var_gamma)
        return res

    def get_derivative_function_list(self, params):
        ge = GammaExponential(params)
        return [ge._dge_dsigmaf, ge._dge_dl, ge._dge_dgamma]