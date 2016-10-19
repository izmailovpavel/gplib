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
from ..optim.methods import LBFGS, AdaDelta, ProjNewton, FGD


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

	def fit(self, X, y, method='JJ', options={}):
		"""
		Fit the sparse gpc model to the data
		:param X: training points
		:param y: training labels
		:param method: A string, representing the chosen method of training the model
			- 'Taylor'
			- 'JJ'
			- 'JJ_full'
			- 'JJ_hybrid'
			- 'svi'
		:param options: A fictionary with fields 
			- 'optimizer' — OptimizerBase object or name of an Optimization method
				- 'L-BFGS-B'
				— 'Projected Newton'
				— 'FGD'
				- 'AdaDelta' — Only for 'svi'
			If a string is passed, method is used with default parameters for the given method.
			If you want to choose specific parameters of the methods pass an OptimizerBas
			- 'n_upd'=5 — number of recomputes (only for 'JJ_hybrid', 'JJ', 'Taylor')
			- 'maxiter'=100 — maximum number of outter iterations (only for 'JJ_hybrid', 
			'JJ', 'Taylor')
			- 'batch_size'=y.size/100 — batch size (only for svi+AdaDelta)
			- 'disp'=False — number (frequency of printing) or bool, indicating wether or not
			to print progress
		"""
		batch_size = None
		if method in ['JJ_full', 'JJ_hybrid', 'JJ', 'Taylor']:
			optimizer = _extract_and_delete(options, 'optimizer', 'L-BFGS-B')
		elif method in ['svi']:
			optimizer = _extract_and_delete(options, 'optimizer', 'AdaDelta')
		else:
			raise ValueError('Unknown method name:'+method)
		if isinstance(optimizer, str):
			if optimizer == 'L-BFGS-B':
				optimizer = LBFGS(maxfun=5)
			elif optimizer == 'Projected Newton':
				optimizer = ProjNewton(maxiter=5)
			elif optimizer == 'FGD':
				optimizer = FGD(maxiter=5)
			elif optimizer == 'AdaDelta':
				batch_size = _extract_and_delete(options, 'batch_size', int(y.size/100))
				optimizer = AdaDelta(n_epoch=100, iter_per_epoch=y.size/batch_size)
			else:
				raise ValueError('Unknown optimizer name:'+optimizer)
	
		n_upd = _extract_and_delete(options, 'n_upd', 5)
		maxiter = _extract_and_delete(options, 'maxiter', 100)
		disp = _extract_and_delete(options, 'disp', False)
		
		self.init_inputs(X)

		if method == 'JJ':
			self.elbo = JJ(X, y, self.inputs[0], self.cov)
			res = self._fit_blockwise(self.elbo, optimizer, n_upd, maxiter, disp)
		elif method == 'Taylor':
			self.elbo = Taylor(X, y, self.inputs[0], self.cov)
			res = self._fit_blockwise(self.elbo, optimizer, n_upd, maxiter, disp)
		elif method == 'JJ_full':
			self.elbo = JJfull(X, y, self.inputs[0], self.cov)
			res = self._fit_simple(self.elbo, optimizer)
		elif method == 'JJ_hybrid':
			self.elbo = JJhybrid(X, y, self.inputs[0], self.cov)
			res = self._fit_blockwise(self.elbo, optimizer, n_upd, maxiter, disp)
		elif method == 'svi':
			if batch_size is None:
				batch_size = int(y.size / optimizer.iter_per_epoch)
			self.elbo = SVI(X, y, self.inputs[0], self.cov, batch_size)
			res = self._fit_simple(self.elbo, optimizer)
		else:
			raise ValueError('Unknown method: ' + str(method))

		return res




