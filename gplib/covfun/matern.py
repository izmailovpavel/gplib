import numpy as np
from scipy.special import gamma, kv

from .cov_family import StationaryCovarianceFamily
from .utility import pairwise_distance, stationary_cov, gaussian_noise_term


class Matern(StationaryCovarianceFamily):
    def __init__(self, params):
        if params.size != 4:
            raise ValueError("Wrong parameters for Matern")
        self.sigma_f = params[0]
        self.l = params[1]
        self.nu = params[2]
        self.sigma_l = params[3]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.nu, self.sigma_l])

    def set_params(self, params):
        if params.size != 4:
            raise ValueError("Wrong parameters for Matern")
        self.sigma_f = params[0]
        self.l = params[1]
        self.nu = params[2]
        self.sigma_l = params[3]

    def st_covariance_function(self, r, w=None):
        if w is None:
            l = self.l
            nu = self.nu
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            nu = w[2]
            sigma_l = w[3]
        anc_var = np.sqrt(2.0 * nu) * r / l
        res = sigma_f**2 *(2.0 ** (1.0 - nu) / gamma(nu)) * (anc_var ** nu) * kv(nu, anc_var)
        res[r == 0] = sigma_f**2
        res += gaussian_noise_term(sigma_l, r)
        return res

    @staticmethod
    def get_bounds():
        return (1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None)

    @stationary_cov
    def _dm_dl(self, r):
        return 1e8 * (self.st_covariance_function(r, w=(self.get_params() + np.array([0, 1e-8, 0, 0]))) -
                      self.st_covariance_function(r))
    @stationary_cov
    def _dm_dnu(self, r):
        return 1e8 * (self.st_covariance_function(r, w=(self.get_params() + np.array([0, 0, 1e-8, 0]))) -
                      self.st_covariance_function(r))

    @stationary_cov
    def _dm_dsigmaf(self, r):
        anc_var = np.sqrt(2.0 * self.nu) * r / self.l
        res = 2 * self.sigma_f * (2.0 ** (1.0 - self.nu) / gamma(self.nu)) * (anc_var ** self.nu) * kv(self.nu,
                                                                                                        anc_var)
        res[r == 0] = 2 * self.sigma_f
        return res

    def get_derivative_function_list(self, params):
        m = Matern(params)
        return [m._dm_dsigmaf, m._dm_dl, m._dm_dnu]