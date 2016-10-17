import numpy as np
import scipy as sp
import numbers
import copy
import time
import scipy.optimize as op
from sklearn.cluster import KMeans
from abc import ABCMeta, abstractmethod

from .gpres import GPRes
from .gp import GP
from .elbo import ELBO
from .optim.utility import check_gradient
from .optim.methods import scipy_wrapper, climin_wrapper


class GPSparse(GP):
	"""
	An abstract base class for sparse GP methods
	"""
	__metaclass__ = ABCMeta

	def __init__(self, cov_obj, mean_function=lambda x: 0, inputs=None):
		"""
		:param cov_obj: object of the CovarianceFamily class
		:param mean_function: function, mean of the gaussian process
		:param inputs: number of inducing inputs or inputs themselves
		"""
		GP.__init__(self, cov_obj, mean_function)

		# A tuple: inducing inputs, and parameters of gaussian distribution at these points (mean and covariance)
		if isinstance(inputs, np.ndarray):
			self.inputs = inputs, None, None
			self.m = inputs.shape[0]
		elif isinstance(inputs, numbers.Integral):
			self.inputs = None
			self.m = inputs
		else:
			raise TypeError('inputs must be either a number or a numpy.ndarray')

		self.method = None

	def init_inputs(self, X):
		"""
		Initialize inducing input positions with K-means clustering
		"""

		if self.inputs is None:
			means = KMeans(n_clusters=self.m)
			means.fit(X)
			self.inputs = means.cluster_centers_, None, None

	def _fit_simple(self, elbo, options, method='L-BFGS-B'):
		"""
		Fit model to the data using maximization of the ELBO
		:param X: train data points
        :param y: train target values
        :param elbo: ELBO object 
		"""
		if not isinstance(elbo, ELBO):
			raise TypeError('elbo should be an instance of ELBO class')

		w0 = elbo.get_params_opt()
		if method == 'L-BFGS-B':
			bnds = elbo.get_bounds_opt()
			res, w_list, t_list = scipy_wrapper(elbo.elbo, w0, method='L-BFGS-B', mydisp=False, bounds=bnds,
	                                                           options=options)
			elbo.set_params_opt(res['x'])
		elif method == 'AdaDelta':
			res, w_list, t_list = climin_wrapper(elbo.elbo, w0, options=options)
			elbo.set_params_opt(res)
		else:
			raise ValueError('Unnknown method' + method)
		self.cov = elbo.cov
		self.inputs = elbo.get_inputs()
		return GPRes(param_lst=w_list, time_lst=t_list)

	def _fit_blockwise(self, elbo, maxiter, n_upd, options):
		"""
		Fit model to the data using maximization of the ELBO
		:param X: train data points
        :param y: train target values
        :param elbo: ELBO object 
		"""
		if not isinstance(elbo, ELBO):
			raise TypeError('elbo should be an instance of ELBO class')

		bnds = elbo.get_bounds_opt()
		w_list, t_list = [], []
		start = time.time()
		for i in range(maxiter):
			elbo.recompute_parameters(n_upd)
			w = elbo.get_params_opt()
			res, _, _ = scipy_wrapper(elbo.elbo, w, method='L-BFGS-B', mydisp=False, bounds=bnds,
                                                           options=options)
			w = res['x']
			elbo.set_params_opt(np.copy(w))
			w_list.append(elbo.get_state())
			t_list.append(time.time() - start)
		self.cov = elbo.cov
		self.inputs = elbo.get_inputs()
		return GPRes(param_lst=w_list, time_lst=t_list)

	@staticmethod
	@abstractmethod
	def get_quality(y_true, y_pred):
		"""
		Quality metric for particular model
		"""
		pass

	def get_prediction_quality(self, x_test, y_test, params):
		"""
		Returns prediction accuracy on the test set for the given parameters for the
		method
		:param params: parameters
		:param X_test: test set points
		:param y_test: test set target values
		:return: prediction accuracy on test data
		"""
		if self.elbo is None:
			raise ValueError('Model is not fitted')
		self.elbo.set_state(params)
		predictive_gp = type(self)(copy.deepcopy(self.elbo.cov), self.mean, self.inputs[0])
		predictive_gp.inputs = self.elbo.get_inputs()
		y_pred = predictive_gp.predict(x_test)

		return self.get_quality(y_test, y_pred)

