import numpy as np
import scipy as sp
import numbers
import copy
import time
import scipy.optimize as op
from sklearn.cluster import KMeans

from .gpc_base import GPC
from ..covfun.utility import sigmoid
from .gpc_svi import SVIMethod
from .gpc_vi_jj import VIJJMethod
from .gpc_vi_taylor import VITaylorMethod
from .gpc_vi_jj_full import VIJJFullMethod
from ..gpres import GPRes


class GPCSparse(GPC):
	"""
	A basic method for GP-classification, based on Laplace integral approximation
	"""
	
	def __init__(self, cov_obj, mean_function=lambda x: 0, inputs=None):
		"""
		:param cov_obj: object of the CovarianceFamily class
		:param mean_function: function, mean of the gaussian process
		:param inputs: number of inducing inputs or inputs themselves
		"""
		super(GPCSparse, self).__init__(cov_obj, mean_function)

		# A tuple: inducing inputs, and parameters of gaussian distribution at these points (mean and covariance)
		if isinstance(inputs, np.ndarray):
			self.inducing_inputs = inputs, None, None
			self.m = inputs.shape[0]
		elif isinstance(inputs, numbers.Integral):
			self.inducing_inputs = None
			self.m = inputs
		else:
			raise TypeError('inputs must be either a number or a numpy.ndarray')

		self.method = None

	def _get_method(self, method_name, method_options):
		if not method_name in ['vi_taylor', 'vi_jj', 'vi_jj_full', 'vi_jj_hybrid', 'svi']:
			raise ValueError('Unknown method: ' + str(method))

		if method_name == 'svi':
			method = SVIMethod(self.cov, method_options)
		elif method_name == 'vi_jj':
			method = VIJJMethod(self.cov, method_options)
		elif method_name == 'vi_taylor':
			method = VITaylorMethod(self.cov, method_options)
		elif method_name == 'vi_jj_full':
			method = VIJJFullMethod(self.cov, method_options, method_type='full')
		elif method_name == 'vi_jj_hybrid':
			method = VIJJFullMethod(self.cov, method_options, method_type='hybrid')
		return method

	def predict(self, test_points):
		"""
		Predict new values given inducing points
		:param ind_points: inducing points
		:param expectation: expectation at inducing points
		:param covariance: covariance at inducing points
		:param test_points: test points
		:return: predicted values at inducing points
		"""
		ind_points, expectation, covariance = self.inducing_inputs
		cov_fun = self.cov
		K_xm = cov_fun(test_points, ind_points)
		K_mm = cov_fun(ind_points, ind_points)
		K_mm_inv = np.linalg.inv(K_mm)

		new_mean = K_xm.dot(K_mm_inv.dot(expectation))

		return np.sign(new_mean)

	def fit(self, X, y, method='vi_jj', method_options={}):
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

		# if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
		if self.inducing_inputs is None:
			means = KMeans(n_clusters=self.m)
			means.fit(X)
			self.inducing_inputs = means.cluster_centers_, None, None

		# Initializing required variables

		self.method = self._get_method(method, method_options)

		inducing_inputs, theta, res = self.method.fit(X, y, self.inducing_inputs[0])
		self.inducing_inputs = inducing_inputs
		self.cov.set_params(theta)
		return res

	def get_prediction_quality(self, params, X_test, y_test):
		"""
		Returns prediction quality on the test set for the given parameters for the
		method
		:param params: parameters
		:param X_test: test set points
		:param y_test: test set target values
		:return: prediction accuracy on test data
		"""
		if self.method is None:
			raise ValueError('Model should be fitted first, as method should be specified')
		return self.method.get_prediction_quality(self, params, X_test, y_test)



