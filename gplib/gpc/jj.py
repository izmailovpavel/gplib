import numpy as np
import copy
import time
import copy
from scipy.special import expit

from ..utility import _extract_and_delete, _get_inv_logdet_cholesky
from ..optim.utility import check_gradient
from ..optim.methods import scipy_wrapper
from ..gpres import GPRes
from .jj_base import JJbase

class JJ(JJbase):
	"""
	A class, representing vi_jj method ELBO
	"""

	def __init__(self, X, y, inputs, cov):
		"""
		:param X: data points (possibly a batch for some methods)
        :param y: target values
        :param inputs: inducing inputs (positions)
        :param cov: covariance function
		"""
		JJbase.__init__(self, X, y, inputs, cov)
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

	def get_state(self):
		return np.copy(self.cov.get_params()), np.copy(self.mu), np.copy(self.Sigma)

	def set_state(self, params):
		theta, mu, Sigma = params
		self.cov.set_params(theta)
		self.mu = mu
		self.Sigma = Sigma

	def set_params(self, params):
		self.set_params_opt(params)
		K_nm = self.cov(self.X, self.inputs)
		K_mm = self.cov(self.inputs, self.inputs)
		K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)
		self._recompute_var_parameters(K_mm_inv, K_nm)

	def get_bounds_opt(self):
		return self.cov.get_bounds()

class JJfull(JJbase):
	"""
	A class, representing vi_jj_full method ELBO
	"""
	def __init__(self, X, y, inputs, cov):
		"""
		:param X: data points (possibly a batch for some methods)
        :param y: target values
        :param inputs: inducing inputs (positions)
        :param cov: covariance function
		"""
		JJbase.__init__(self, X, y, inputs, cov)
		m = inputs.shape[0]
		self.xi = np.ones(y.size)

	def elbo(self, w):
		fun, grad = self._elbo(w, with_xi=True)
		return -fun, -grad

	def get_params_opt(self):
		theta = self.cov.get_params()
		return np.array(theta.tolist() + self.xi.reshape(-1).tolist())

	def get_state(self):
		return self.get_params_opt()

	def set_state(self, params):
		self.set_params_opt(params)

	def set_params_opt(self, params):
		n_theta = self.cov.get_params().size
		self.cov.set_params(params[:n_theta])
		self.xi = params[n_theta:, None]
		K_nm = self.cov(self.X, self.inputs)
		K_mm = self.cov(self.inputs, self.inputs)
		K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)
		self._recompute_var_parameters(K_mm_inv, K_nm)

	def get_bounds_opt(self):
		bnds = list(self.cov.get_bounds())
		bnds = np.array(bnds+ [(1e-3, np.inf)] * self.xi.size)
		return bnds

class JJhybrid(JJbase):
	"""
	A class, representing vi_jj_hybrid method ELBO
	"""

	def __init__(self, X, y, inputs, cov):
		"""
		:param X: data points (possibly a batch for some methods)
        :param y: target values
        :param inputs: inducing inputs (positions)
        :param cov: covariance function
		"""
		JJbase.__init__(self, X, y, inputs, cov)
		m = inputs.shape[0]
		self.mu = np.zeros((m,1))
		self.Sigma = np.eye(m)

	def elbo(self, w):
		fun, grad = self._elbo(w, with_xi=True)
		return -fun, -grad

	def get_params_opt(self):
		theta = self.cov.get_params()
		return np.array(theta.tolist() + self.xi.reshape(-1).tolist())

	def set_params_opt(self, params):
		n_theta = self.cov.get_params().size
		self.cov.set_params(params[:n_theta])
		self.xi = params[n_theta:, None]

	def set_params(self, params):
		self.set_params_opt(params)
		K_nm = self.cov(self.X, self.inputs)
		K_mm = self.cov(self.inputs, self.inputs)
		K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)
		self._recompute_var_parameters(K_mm_inv, K_nm)

	def get_bounds_opt(self):
		bnds = list(self.cov.get_bounds())
		bnds = np.array(bnds+ [(1e-3, np.inf)] * self.y.size)
		return bnds

	def get_state(self):
		return np.copy(self.cov.get_params()), np.copy(self.mu), np.copy(self.Sigma)

	def set_state(self, params):
		theta, mu, Sigma = params
		self.cov.set_params(theta)
		self.mu = mu
		self.Sigma = Sigma



