import numpy as np
import copy
import time
import copy
from scipy.special import expit

from ..utility import _extract_and_delete, _get_inv_logdet_cholesky
from ..optim.utility import check_gradient
from ..optim.methods import scipy_wrapper
from ..gpres import GPRes


class VITaylorMethod:
    """
    Class, implementing vi_jj method for GP-classification.
    """

    def __init__(self, cov, method_options):
        """
        :param cov: CovarianceFamily object; it's hyper-parameters are being fitted
        :param method_options:
            - mydisp: wether or not to print progress info 
            - max_out_iter: maximum number of outter iterations
            - num_updates: number of updates of parameters xi, mu, sigma at each iteration
            - LBFGS-B optimizer options from scipy
        """
        self.cov = copy.deepcopy(cov)
        self.method_options = method_options
        self.name = 'vi_taylor'
        self.mydisp = _extract_and_delete(self.method_options, 'mydisp', False)
        self.max_out_iter = _extract_and_delete(self.method_options, 'max_out_iter', 20)
        self.num_updates = _extract_and_delete(self.method_options, 'num_updates', 3)

    def _phi(self, xi, y):
        return y * expit(-y * xi)
        # return y / (1 + np.exp(y * xi))

    def _psi(self, xi, y):
        return expit(y * xi) * expit(-y * xi)
        # return (1 / (1 + np.exp(y * xi))) / (1 + np.exp(-y * xi))

    def _v(self, xi, y):
        return self._phi(xi, y) + 2 * self._psi(xi, y) * xi

    def _update_xi(self, params, data_points, target_values, inputs, mu, Sigma, n_iter=5):

        y = target_values
        cov_obj = copy.deepcopy(self.cov)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_nm = cov_fun(data_points, inputs)
        K_mn = K_nm.T
        K_mm = cov_fun(inputs, inputs)
        K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)
        xi = 0

        for i in range(n_iter):
            xi = K_nm.dot(K_mm_inv.dot(mu))
            B = 2 * K_mn.dot(self._psi(xi, y) * K_nm) + K_mm
            B_inv, _ = _get_inv_logdet_cholesky(B)
            Sigma = K_mm.dot(B_inv.dot(K_mm))
            mu = K_mm.dot(B_inv.dot(K_mn.dot(self._v(xi, y))))

        return xi, mu, Sigma

    def fit(self, X, y, inputs):
        """
        An experimental method for optimizing hyper-parameters (for fixed inducing points), based on variational
        inference and Second order Taylor approximation to the logistic function. See the review.
        :param data_points: training set objects
        :param target_values: training set answers
        :param inputs: inducing inputs
        :param num_inputs: number of inducing points to generate. If inducing points are provided, this parameter is
        ignored
        :param optimizer_options: options for the optimization method
        :return:
        """

        # Initializing required variables
        m = inputs.shape[0]
        n = y.size

        # Initializing variational (normal) distribution parameters
        mu = np.zeros((m, 1), dtype=float)
        Sigma = np.eye(m)

        def oracle(w):
            fun, grad = self._elbo(X, y, w, inputs, xi)
            return -fun, -grad

        bnds = self.cov.get_bounds()
        params = self.cov.get_params()
        w_list, time_list = [(params, mu, Sigma)], [0]
        start = time.time()

        for iteration in range(self.max_out_iter):
            xi, mu, Sigma = self._update_xi(params, X, y, inputs, mu, Sigma,
                                                      n_iter=self.num_updates)
            res, _, _ = scipy_wrapper(oracle, params, method='L-BFGS-B', mydisp=self.mydisp, 
                                         bounds=bnds, options=self.method_options)

            params = res['x']
            w_list.append((params, np.copy(mu), np.copy(Sigma)))
            time_list.append(time.time() - start)
            if self.mydisp:
                print('\tHyper-parameters at outter iteration', iteration, ':', params)

        inducing_inputs = inputs, mu, Sigma
        self.cov.set_params(params)
        return inducing_inputs, params, GPRes(param_lst=w_list, time_lst=time_list)

    def _elbo(self, points, targets, params, ind_points, xi):
        """
        The evidence lower bound, used in the vi method.
        :param points: data points
        :param targets: target values
        :param params: hernel hyper-parameters
        :param ind_points: inducing input positions
        :param xi: variational parameters xi
        :return: the value and the gradient of the lower bound
        """
        y = targets
        m = ind_points.shape[0]
        cov_obj = copy.deepcopy(self.cov)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        Psi_xi = self._psi(xi, y)

        K_mm = cov_fun(ind_points, ind_points)
        K_mm_inv, K_mm_log_det = _get_inv_logdet_cholesky(K_mm)
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        K_ii = cov_fun(points[:1, :], points[:1, :])

        v_xi = self._v(xi, y)

        K_mnPsiK_nm = K_mn.dot(Psi_xi*K_nm)

        B = 2 * K_mnPsiK_nm + K_mm

        B_inv, B_log_det = _get_inv_logdet_cholesky(B)

        fun = ((v_xi.T.dot(K_nm.dot(B_inv.dot(K_mn.dot(v_xi))))/2) +
               K_mm_log_det/2 - B_log_det/2
               - np.sum(K_ii * Psi_xi) + np.einsum('ij,ji->', K_mm_inv, K_mnPsiK_nm))

        gradient = []
        derivative_matrix_list = cov_obj.get_derivative_function_list(params)
        for param in range(len(params)):
            if param != len(params) - 1:
                func = derivative_matrix_list[param]
            else:
                func = lambda x, y: cov_obj.get_noise_derivative(points_num=1)
            if param != len(params) - 1:
                dK_mm = func(ind_points, ind_points)
                dK_nm = func(points, ind_points)
                dK_mn = dK_nm.T
                dB = 4 * dK_mn.dot(Psi_xi*K_nm) + dK_mm
            else:
                dK_mm = np.eye(m) * func(ind_points, ind_points)
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

    @staticmethod
    def get_prediction_quality(gp_obj, params, x_test, y_test):
        """
        Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
        method
        :param params: parameters
        :param x_test: test set points
        :param y_test: test set target values
        :return: prediction accuracy
        """
        new_gp = copy.deepcopy(gp_obj)
        theta, mu, Sigma = params
        new_gp.cov.set_params(theta)
        new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
        predicted_y_test = new_gp.predict(x_test)
        return 1 - np.sum(y_test != predicted_y_test) / y_test.size

