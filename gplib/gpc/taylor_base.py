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


class TaylorBase(ELBO):
	"""
	Abstract base class for methods, based on Taylor lower bound
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

	@staticmethod
	def _phi(xi, y):
		return y * expit(-y * xi)
		# return y / (1 + np.exp(y * xi))

	@staticmethod
	def _psi(xi, y):
		return expit(y * xi) * expit(-y * xi)
		# return (1 / (1 + np.exp(y * xi))) / (1 + np.exp(-y * xi))

	@classmethod
	def _v(self, xi, y):
		return self._phi(xi, y) + 2 * self._psi(xi, y) * xi

	def recompute_parameters(self, n_upd=5):
		"""
		Updates xi, mu, Sigma
		:n_upd: number of updates for each parameter
		"""
		K_nm = self.cov(self.X, self.inputs)
		K_mn = K_nm.T
		K_mm = self.cov(self.inputs, self.inputs)
		K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)

		for i in range(n_upd):
			self.xi = K_nm.dot(K_mm_inv.dot(self.mu))
			B = 2 * K_mn.dot(self._psi(self.xi, self.y) * K_nm) + K_mm
			B_inv, _ = _get_inv_logdet_cholesky(B)
			self.Sigma = K_mm.dot(B_inv.dot(K_mm))
			self.mu = K_mm.dot(B_inv.dot(K_mn.dot(self._v(self.xi, self.y))))

	def _elbo(self, params_vec, with_xi=False):
		"""
		The evidence lower bound, used in the vi method.
		:param params: a vector (kernel hyper-parameters, xi)
		:param with_xi: wether or not to compute the derivatives wrt xi
		(Not yet implemented)
		:return: the value and the gradient of the lower bound
		"""
		params_num = self.cov.get_params().size
		params = params_vec[:params_num]
		xi = params_vec[params_num:][:, None]

		n = self.X.shape[0]
		m = self.inputs.shape[0]

		Psi_xi = self._psi(xi, self.y)

		cov_fun = copy.deepcopy(self.cov)
		cov_fun.set_params(params)

		K_mm = cov_fun(self.inputs, self.inputs)
		K_mm_inv, K_mm_log_det = _get_inv_logdet_cholesky(K_mm)
		K_nm = cov_fun(self.X, self.inputs)
		K_mn = K_nm.T
		K_ii = cov_fun(self.X[:1, :], self.X[:1, :])

		v_xi = self._v(xi, self.y)

		K_mnPsiK_nm = K_mn.dot(Psi_xi*K_nm)

		B = 2 * K_mnPsiK_nm + K_mm

		B_inv, B_log_det = _get_inv_logdet_cholesky(B)

		fun = ((v_xi.T.dot(K_nm.dot(B_inv.dot(K_mn.dot(v_xi))))/2) +
			   K_mm_log_det/2 - B_log_det/2
			   - np.sum(K_ii * Psi_xi) + np.einsum('ij,ji->', K_mm_inv, K_mnPsiK_nm))

		gradient = []
		derivative_matrix_list = cov_fun.get_derivative_function_list(params)
		for param in range(len(params)):
			if param != len(params) - 1:
				func = derivative_matrix_list[param]
			else:
				func = lambda x, y: cov_fun.get_noise_derivative(points_num=1)
			if param != len(params) - 1:
				dK_mm = func(self.inputs, self.inputs)
				dK_nm = func(self.X, self.inputs)
				dK_mn = dK_nm.T
				dB = 4 * dK_mn.dot(Psi_xi*K_nm) + dK_mm
			else:
				dK_mm = np.eye(m) * func(self.inputs, self.inputs)
				dK_mn = np.zeros_like(K_mn)
				dK_nm = dK_mn.T
				dB = dK_mm
			dK_nn = func(np.array([[0]]), np.array([[0]]))
			derivative = np.array([[0]], dtype=float)

			derivative += v_xi.T.dot(dK_nm.dot(B_inv.dot(K_mn.dot(v_xi))))
			derivative += - v_xi.T.dot(K_nm.dot(B_inv.dot(dB.dot(B_inv.dot(K_mn.dot(v_xi))))))/2

			derivative += np.trace(K_mm_inv.dot(dK_mm))/2
			derivative -= np.trace(B_inv.dot(dB))/2
			derivative -= np.sum(Psi_xi * dK_nn)
			derivative += np.trace(2 * K_mm_inv.dot(K_mn.dot(Psi_xi*dK_nm)) -
								   K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(K_mnPsiK_nm))))
			gradient.append(derivative[0, 0])
		return fun, np.array(gradient)

	def get_inputs(self):
		"""
		Returns the inducing input positions, mean and covariance matrix
		of the process at these points
		"""
		return self.inputs, self.mu, self.Sigma

	# @staticmethod
	# def get_prediction_quality(gp_obj, params, x_test, y_test):
	#     """
	#     Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
	#     method
	#     :param params: parameters
	#     :param x_test: test set points
	#     :param y_test: test set target values
	#     :return: prediction accuracy
	#     """
	#     new_gp = copy.deepcopy(gp_obj)
	#     theta, mu, Sigma = params
	#     new_gp.cov.set_params(theta)
	#     new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
	#     predicted_y_test = new_gp.predict(x_test)
	#     return 1 - np.sum(y_test != predicted_y_test) / y_test.size
