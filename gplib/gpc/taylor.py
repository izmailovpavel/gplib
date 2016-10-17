import numpy as np
import copy
import time
import copy
from scipy.special import expit

from ..utility import _extract_and_delete, _get_inv_logdet_cholesky
from ..optim.utility import check_gradient
from ..optim.methods import scipy_wrapper
from ..gpres import GPRes
from .taylor_base import TaylorBase

class Taylor(TaylorBase):
	"""
	A class, representing vi_taylor method ELBO
	"""

	def __init__(self, X, y, inputs, cov):
		"""
		:param X: data points (possibly a batch for some methods)
        :param y: target values
        :param inputs: inducing inputs (positions)
        :param cov: covariance function
		"""
		TaylorBase.__init__(self, X, y, inputs, cov)
		m = inputs.shape[0]
		self.mu = np.zeros((m,1))
		self.Sigma = np.eye(m)

	def elbo(self, w):
		params = np.array(w.tolist() + self.xi.reshape(-1).tolist())
		fun, grad = self._elbo(params, with_xi=False)
		return -fun, -grad

	def get_params_opt(self):
		return self.cov.get_params()

	def set_params_opt(self, params):
		self.cov.set_params(params)

	def get_bounds_opt(self):
		return self.cov.get_bounds()

	def get_state(self):
		return np.copy(self.cov.get_params()), np.copy(self.mu), np.copy(self.Sigma)

	def set_state(self, params):
		theta, mu, Sigma = params
		self.cov.set_params(theta)
		self.mu = mu
		self.Sigma = Sigma