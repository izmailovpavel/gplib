import numpy as np
import copy
import time
import copy
from abc import ABCMeta, abstractmethod
from scipy.special import expit

from ..utility import _get_inv_logdet_cholesky
from ..optim.utility import check_gradient
from ..optim.methods import scipy_wrapper
from ..gpres import GPRes
from ..elbo import ELBO


class JJbase(ELBO):
	"""
	Abstract base class for several methods, based on Jaakkola-Jordan lower bound
	"""
	__metaclass__ = ABCMeta

	def __init__(self, X, y, inputs, cov):
		"""
		:param X: data points (possibly a batch for some methods)
		:param y: target values
		:param inputs: inducing inputs (positions)
		:param cov: covariance function
		"""
		ELBO.__init__(self, X, y)
		self.cov = cov
		self.inputs = inputs
		self.xi = None
		self.mu = None
		self.Sigma = None

	def _recompute_xi(self, K_mm, K_mm_inv, K_nm, K_ii):
		"""
		Computes the optimal values of xi, given mu and Sigma
		"""
		K_mn = K_nm.T
		means = K_nm.dot(K_mm_inv.dot(self.mu))
		vars = K_ii + np.einsum('ij,ji->i', K_nm, 
								K_mm_inv.dot((self.Sigma - K_mm).dot(K_mm_inv.dot(K_mn))))[:, None]
		self.xi = np.sqrt(means**2 + vars)

	def _recompute_var_parameters(self, K_mm_inv, K_nm):
		"""
		Computes optimal mu and Sigma, given xi
		"""
		K_mn = K_nm.T
		Lambda_xi = self._lambda(self.xi)
		self.Sigma = np.linalg.inv(2 * K_mm_inv.dot(K_mn.dot((Lambda_xi * K_nm).dot(K_mm_inv))) + K_mm_inv)
		self.mu = self.Sigma.dot(K_mm_inv.dot(K_mn.dot(self.y))) / 2

	def recompute_parameters(self, n_upd=5):
		"""
		Updates xi, mu, Sigma
		:n_upd: number of updates for each parameter
		"""
		K_nm = self.cov(self.X, self.inputs)
		K_mm = self.cov(self.inputs, self.inputs)
		K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)
		K_ii = self.cov(self.X[:1, :], self.X[:1, :])
		for i in range(n_upd):
			self._recompute_xi(K_mm, K_mm_inv, K_nm, K_ii)
			self._recompute_var_parameters(K_mm_inv, K_nm)

	@staticmethod
	def _lambda(xi):
		return np.tanh(xi / 2) / (4 * xi)

	@staticmethod
	def _log_g(xi):
		return np.log(expit(xi))

	@staticmethod
	def _dlambda_dxi(xi):
		return (xi - np.sinh(xi)) / (4 * xi**2 * (np.cosh(xi) + 1))

	def _elbo(self, params_vec, with_xi):
		"""
		The evidence lower bound, used in the vi method.
		:param params: a vector (kernel hyper-parameters, xi)
		:param with_xi: wether or not to compute the derivatives wrt xi
		:return: the value and the gradient of the lower bound
		"""
		params_num = self.cov.get_params().size
		params = params_vec[:params_num]
		xi = params_vec[params_num:][:, None]

		y = self.y
		n = self.X.shape[0]
		m = self.inputs.shape[0]
		cov_fun = copy.deepcopy(self.cov)
		cov_fun.set_params(params)
		lambda_xi = self._lambda(xi)
		K_mm = cov_fun(self.inputs, self.inputs)
		K_mm_inv, K_mm_log_det = _get_inv_logdet_cholesky(K_mm)
		K_nm = cov_fun(self.X, self.inputs)
		K_mn = K_nm.T
		K_mnLambdaK_nm = K_mn.dot(lambda_xi*K_nm)
		K_ii = cov_fun(self.X[:1, :], self.X[:1, :])

		B = 2 * K_mnLambdaK_nm + K_mm

		B_inv, B_log_det = _get_inv_logdet_cholesky(B)

		fun = ((y.T.dot(K_nm.dot(B_inv.dot(K_mn.dot(y))))/8)[0, 0] + K_mm_log_det/2 - B_log_det/2
			   - np.sum(K_ii * lambda_xi) + np.einsum('ij,ji->', K_mm_inv, K_mnLambdaK_nm))
		if with_xi:
			fun -= np.sum(xi) / 2
			fun += np.sum(lambda_xi * xi**2)
			fun += np.sum(self._log_g(xi))

		gradient = []
		derivative_matrix_list = cov_fun.get_derivative_function_list(params)

		for param in range(len(params)):
			if param != len(params) - 1:
				func = derivative_matrix_list[param]
			else:
				func = lambda x, y: cov_fun.get_noise_derivative(1)
			if param != len(params) - 1:
				dK_mm = func(self.inputs, self.inputs)
				dK_nm = func(self.X, self.inputs)
				dK_mn = dK_nm.T
				dB = 4 * dK_mn.dot(lambda_xi*K_nm) + dK_mm
			else:
				dK_mm = np.eye(m) * func(self.inputs, self.inputs)
				dK_mn = np.zeros_like(K_mn)
				dK_nm = dK_mn.T
				dB = dK_mm
			dK_nn = func(np.array([[0]]), np.array([[0]]))
			derivative = np.array([[0]], dtype=float)
			derivative += y.T.dot(dK_nm.dot(B_inv.dot(K_mn.dot(y))))/4
			derivative -= y.T.dot(K_nm.dot(B_inv.dot(dB.dot(B_inv.dot(K_mn.dot(y))))))/8
			derivative += np.trace(K_mm_inv.dot(dK_mm))/2
			derivative -= np.trace(B_inv.dot(dB))/2
			derivative -= np.sum(lambda_xi * dK_nn)
			derivative += np.trace(2 * K_mm_inv.dot(K_mn.dot(lambda_xi*dK_nm)) -
								   K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(K_mnLambdaK_nm))))
			gradient.append(derivative[0, 0])

		if with_xi:
			xi_gradient = np.zeros((n, 1))
			dlambdaxi_dxi = self._dlambda_dxi(xi)
			anc_vec = B_inv.dot(K_mn.dot(y))
			xi_gradient += - 2 * np.einsum('ij,jk->jik', anc_vec.T.dot(K_mn), 
											dlambdaxi_dxi*K_nm.dot(anc_vec)).reshape(-1)[:, None] / 8


			xi_gradient += - np.einsum('ij,ji->j', B_inv.dot(K_mn), dlambdaxi_dxi*K_nm)[:, None]

			xi_gradient += - K_ii * dlambdaxi_dxi
			xi_gradient += (np.einsum('ij,ji->i', K_nm, K_mm_inv.dot(K_mn))[:, None] * dlambdaxi_dxi)
			xi_gradient += -1/2
			xi_gradient += dlambdaxi_dxi * xi**2 + lambda_xi * xi * 2
			xi_gradient += expit(-xi)

			gradient = gradient + xi_gradient.reshape(-1).tolist()

		return fun, np.array(gradient).reshape(-1)

	def get_inputs(self):
		"""
		Returns the inducing input positions, mean and covariance matrix
		of the process at these points
		"""
		if self.mu is None or self.Sigma is None:
			if self.xi is None:
				raise ValueError('Model is not fitted')
			K_mm = self.cov(self.inputs, self.inputs)
			K_nm = self.cov(self.X, self.inputs)

			K_mm_inv, _ = _get_inv_logdet_cholesky(K_mm)
			self._recompute_var_parameters(K_mm_inv, K_nm, self.y)

		return self.inputs, self.mu, self.Sigma

