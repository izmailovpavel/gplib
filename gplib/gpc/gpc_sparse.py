import numpy as np
import scipy as sp
import numbers
import copy
import time
import scipy.optimize as op
from sklearn.cluster import KMeans

from .gpc_base import GPC
from ..covfun.utility import sigmoid
from ..utility import _extract_and_delete
from ..optim.utility import check_gradient
from ..gpres import GPRes
from ..gp_sparse import GPSparse
from .jj import JJ, JJfull, JJhybrid
from .taylor import Taylor
from .svi import SVI


class GPCSparse(GPC, GPSparse):
	"""
	Sparse GP-classification class
	"""
	
	def __init__(self, cov_obj, mean_function=lambda x: 0, inputs=None):
		"""
		:param cov_obj: object of the CovarianceFamily class
		:param mean_function: function, mean of the gaussian process
		:param inputs: number of inducing inputs or inputs themselves
		"""
		GPSparse.__init__(self, cov_obj, mean_function, inputs)
		self.elbo = None

	def predict(self, test_points):
		"""
		Predict new values given inducing points
		:param ind_points: inducing points
		:param expectation: expectation at inducing points
		:param covariance: covariance at inducing points
		:param test_points: test points
		:return: predicted values at inducing points
		"""
		ind_points, expectation, covariance = self.inputs
		# print(expectation[:3])
		if expectation is None or covariance is None:
			raise ValueError('Model is not fitted')
		K_xm = self.cov(test_points, ind_points)
		K_mm = self.cov(ind_points, ind_points)
		K_mm_inv = np.linalg.inv(K_mm)

		new_mean = K_xm.dot(K_mm_inv.dot(expectation))

		return np.sign(new_mean)

	def fit(self, X, y, method='vi_jj', options={}):
		"""
		Fit the sparse gpc model to the data
		:param X: training points
		:param y: training labels
		:param method: A string, representing the chosen method of training the model
			- 'vi_taylor'
			- 'vi_jj'
			- 'vi_jj_full'
			- 'vi_jj_hybrid'
			- 'svi'		
		"""

		self.init_inputs(X)

		mydisp = _extract_and_delete(options, 'mydisp', 5)
		if method == 'vi_jj':
			maxiter = _extract_and_delete(options, 'maxiter', 100)
			n_upd = _extract_and_delete(options, 'n_upd', 5)
			self.elbo = JJ(X, y, self.inputs[0], self.cov)
			res = self._fit_blockwise(self.elbo, maxiter, n_upd, options)
		elif method == 'vi_taylor':
			maxiter = _extract_and_delete(options, 'maxiter', 100)
			n_upd = _extract_and_delete(options, 'n_upd', 5)
			self.elbo = Taylor(X, y, self.inputs[0], self.cov)
			res = self._fit_blockwise(self.elbo, maxiter, n_upd, options)
		elif method == 'vi_jj_full':
			self.elbo = JJfull(X, y, self.inputs[0], self.cov)
			res = self._fit_simple(self.elbo, options)
		elif method == 'vi_jj_hybrid':
			maxiter = _extract_and_delete(options, 'maxiter', 100)
			n_upd = _extract_and_delete(options, 'n_upd', 5)
			self.elbo = JJhybrid(X, y, self.inputs[0], self.cov)
			res = self._fit_blockwise(self.elbo, maxiter, n_upd, options)
		elif method == 'svi':
			batch_size = _extract_and_delete(options, 'batch_size', y.size/100)
			self.elbo = SVI(X, y, self.inputs[0], self.cov, batch_size)
			options['train_size'] = y.size
			res = self._fit_simple(self.elbo, options, method='AdaDelta')
		else:
			raise ValueError('Unknown method: ' + str(method))

		return res




