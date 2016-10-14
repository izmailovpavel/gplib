import numpy as np
import scipy as sp
import copy
import time
import scipy.optimize as op
from sklearn.cluster import KMeans

from .gpc_base import GPC
from ..covfun.utility import sigmoid
from .gpc_svi import SVIMethod
from ..gpres import GPRes


class GPCSparse(GPC):
	"""
	A basic method for GP-classification, based on Laplace integral approximation
	"""
	
	def __init__(self, cov_obj, mean_function=lambda x: 0, method='vi_jj'):
		"""
		:param cov_obj: object of the CovarianceFamily class
		:param mean_function: function, mean of the gaussian process
		:param method: A string, representing the chosen method of training the model
			- 'vi_taylor'
			- 'vi_jj'
			- 'vi_jj_full'
			- 'vi_jj_hybrid'
			- 'svi'
		"""
		super(GPCSparse, self).__init__(cov_obj, mean_function)

		# A tuple: inducing inputs, and parameters of gaussian distribution at these points (mean and covariance)
		self.inducing_inputs = None

		self.method = method
		if not self.method in ['vi_taylor', 'vi_jj', 'vi_jj_full', 'vi_jj_hybrid', 'svi']:
			raise ValueError('Unknown method: ' + str(method))

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

	def fit(self, X, y, num_inputs=0, inputs=None, method_options={}):
		"""
		Fit the sparse gpc model to the data
		:param X: training points
		:param y: training labels
		:param num_inputs: number of inducing inputs
		:param inputs: 
		"""

		# if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
		if inputs is None:
			means = KMeans(n_clusters=num_inputs)
			means.fit(X)
			inputs = means.cluster_centers_

		# Initializing required variables
		m = inputs.shape[0]
		n = y.size

		if self.method == 'svi':
			method = SVIMethod(self.cov, method_options)
			inducing_inputs, theta, res = method.fit(X, y, inputs)
			self.inducing_inputs = inducing_inputs
		
		self.cov.set_params(theta)
		return res



