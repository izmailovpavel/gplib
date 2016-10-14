import numpy as np
import copy
import time
import copy
from scipy.special import expit

from ..utility import _extract_and_delete, _get_inv_logdet_cholesky
from ..optim.utility import check_gradient
from ..optim.methods import scipy_wrapper
from ..gpres import GPRes


class VIJJFullMethod:
    """
    Class, implementing vi_jj method for GP-classification.
    """

    def __init__(self, cov, method_options, method_type='hybrid'):
        """
        :param cov: CovarianceFamily object; it's hyper-parameters are being fitted
        :param method_options:
            - mydisp: wether or not to print progress info 
            - max_out_iter: maximum number of outter iterations (only for hybrid)
            - num_updates: number of updates of parameters xi, mu, sigma at each iteration (only for hybrid)
            - LBFGS-B optimizer options from scipy
        :param type: hybrid or full
        """
        self.cov = copy.deepcopy(cov)
        self.method_options = method_options
        self.mydisp = _extract_and_delete(self.method_options, 'mydisp', False)
        self.max_out_iter = _extract_and_delete(self.method_options, 'max_out_iter', 20)
        self.num_updates = _extract_and_delete(self.method_options, 'num_updates', 3)
        if method_type == 'hybrid':
            self.is_hybrid = True
        elif method_type == 'full':
            self.is_hybrid = False
        else:
            raise ValueError('Unknown method type')

    @staticmethod
    def _lambda(xi):
        return np.tanh(xi / 2) / (4 * xi)

    @staticmethod
    def _recompute_xi(K_mm, K_mm_inv, K_nm, K_ii, mu, Sigma):
        K_mn = K_nm.T
        means = K_nm.dot(K_mm_inv.dot(mu))
        vars = K_ii + np.einsum('ij,ji->i', K_nm, K_mm_inv.dot((Sigma - K_mm).dot(K_mm_inv.dot(K_mn))))[:, None]
        return np.sqrt(means**2 + vars)

    def _update_xi(self, params, data_points, target_values, inputs, mu, Sigma, n_iter=5):
        cov_obj = copy.deepcopy(self.cov)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_nm = cov_fun(data_points, inputs)
        K_mm = cov_fun(inputs, inputs)
        K_mm_inv, K_log_det = _get_inv_logdet_cholesky(K_mm)
        K_ii = cov_fun(data_points[:1, :], data_points[:1, :])
        for i in range(n_iter):
            xi = self._recompute_xi(K_mm, K_mm_inv, K_nm, K_ii, mu, Sigma)
            mu, Sigma = self._recompute_var_parameters(K_mm_inv, K_nm, xi, target_values)
        return xi, mu, Sigma

    def _recompute_var_parameters(self, K_mm_inv, K_nm, xi, y):
        K_mn = K_nm.T
        Lambda_xi = self._lambda(xi)
        Sigma = np.linalg.inv(2 * K_mm_inv.dot(K_mn.dot((Lambda_xi * K_nm).dot(K_mm_inv))) + K_mm_inv)
        mu = Sigma.dot(K_mm_inv.dot(K_mn.dot(y))) / 2
        return mu, Sigma

    def _dlambda_dxi(self, xi):
        return (xi - np.sinh(xi)) / (4 * xi**2 * (np.cosh(xi) + 1))
        # return np.tanh(xi / 2) / (4 * xi)

    def fit(self, X, y, inputs):
        """
        An experimental method for optimizing hyper-parameters (for fixed inducing points), based on variational
        inference and Second order Taylor approximation to the logistic function. See the review.
        :param X: training set objects
        :param y: training set answers
        :param inputs: inducing inputs
        :return:
        """

        # Initializing required variables
        m = inputs.shape[0]
        n = y.size

        def oracle(w):
            fun, grad = self._elbo(X, y, inputs, w)
            return -fun, -grad

        bnds = self.cov.get_bounds()
        params = self.cov.get_params()

        xi = np.ones(n)

        bnds = np.array(list(bnds)+ [(1e-3, np.inf)] * xi.size)

        maxfun = 5

        if self.is_hybrid:
            if self.mydisp:
                print('Hybrid mode'+#\n\tAnalytic updates frequency:'+str(freq_updates)+
                      '\n\tAnalytic updates number:'+str(self.num_updates))
            sum_iter = 0
            out_iter = 0
            mu, Sigma = np.zeros((m,1)), np.eye(m)
            vec = np.vstack([params.reshape(-1)[:, None], xi.reshape(-1)[:, None]]).reshape(-1)
            w_list, t_list = [(params, np.copy(mu), np.copy(Sigma))], [0.]
            start = time.time()
            while out_iter < self.max_out_iter:
                if self.mydisp:
                    print('\t Outter iteration '+str(out_iter))

                res, it_w_list, it_t_list = scipy_wrapper(oracle, vec, method='L-BFGS-B', 
                                                             mydisp=self.mydisp, bounds=bnds,
                                                             options=self.method_options)

                point = res['x']
                params = point[:params.size]
                xi = point[params.size:][:, None]
                sum_iter += res['nit']
                out_iter += 1

                cov_obj = copy.deepcopy(self.cov)
                cov_obj.set_params(params)
                cov_fun = cov_obj.covariance_function
                K_mm = cov_fun(inputs, inputs)
                K_nm = cov_fun(X, inputs)

                K_mm_inv, _ = _get_inv_logdet_cholesky(K_mm)
                mu, Sigma = self._recompute_var_parameters(K_mm_inv, K_nm, xi, y)

                xi, mu, Sigma = self._update_xi(params, X, y, inputs, mu, Sigma,
                                                      n_iter=self.num_updates)
                vec = np.vstack([params.reshape(-1)[:, None], xi.reshape(-1)[:, None]]).reshape(-1)
                w_list.append((params, np.copy(mu), np.copy(Sigma)))
                t_list.append(time.time() - start)

        else:
            # self._need_compute_mu_Sigma = True
            if mydisp:
                print('Standard mode')
            vec = np.vstack([params.reshape(-1)[:, None], xi.reshape(-1)[:, None]]).reshape(-1)
            res, w_list, t_list = scipy_wrapper(oracle, vec, method='L-BFGS-B', mydisp=mydisp, bounds=bnds,
                                                           options=self.method_options)
            point = res['x']
            params = point[:params.size]
            xi = point[params.size:][:, None]
            cov_obj = copy.deepcopy(self.cov)
            cov_obj.set_params(params)
            cov_fun = cov_obj.covariance_function
            K_mm = cov_fun(inputs, inputs)
            K_nm = cov_fun(X, inputs)

            K_mm_inv, _ = _get_inv_logdet_cholesky(K_mm)
            mu, Sigma = self._recompute_var_parameters(K_mm_inv, K_nm, xi, y)

        inducing_inputs = inputs, mu, Sigma
        self.cov.set_params(params)
        return inducing_inputs, params, GPRes(param_lst=w_list, time_lst=t_list)

    @classmethod
    def _log_g(self, xi):
        return np.log(expit(xi))

    def _elbo(self, points, targets, ind_points, params_vec):
        """
        The evidence lower bound, used in the vi method.
        :param points: data points
        :param targets: target values
        :param params: hernel hyper-parameters
        :param ind_points: inducing input positions
        :param xi: variational parameters xi
        :return: the value and the gradient of the lower bound
        """
        # start = time.time()
        params_num = self.cov.get_params().size
        params = params_vec[:params_num]
        xi = params_vec[params_num:][:, None]

        y = targets
        n = points.shape[0]
        m = ind_points.shape[0]
        cov_obj = copy.deepcopy(self.cov)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        lambda_xi = self._lambda(xi)
        K_mm = cov_fun(ind_points, ind_points)
        K_mm_inv, K_mm_log_det = _get_inv_logdet_cholesky(K_mm)
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        K_mnLambdaK_nm = K_mn.dot(lambda_xi*K_nm)
        K_ii = cov_fun(points[:1, :], points[:1, :])

        B = 2 * K_mnLambdaK_nm + K_mm

        B_inv, B_log_det = _get_inv_logdet_cholesky(B)

        fun = (y.T.dot(K_nm.dot(B_inv.dot(K_mn.dot(y))))/8)[0, 0] + \
              (K_mm_log_det/2 - B_log_det/2) - np.sum(K_ii * lambda_xi) + np.einsum('ij,ji->', K_mm_inv, K_mnLambdaK_nm)
        fun -= np.sum(xi) / 2
        fun += np.sum(lambda_xi * xi**2)
        fun += np.sum(self._log_g(xi))

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
                dB = 4 * dK_mn.dot(lambda_xi*K_nm) + dK_mm
            else:
                dK_mm = np.eye(m) * func(ind_points, ind_points)
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

        xi_gradient = np.zeros((n, 1))
        dlambdaxi_dxi = self._dlambda_dxi(xi)
        # dB_dxi = 2 * np.einsum('ij,jk->jik', K_mn, dlambdaxi_dxi*K_nm)
        anc_vec = B_inv.dot(K_mn.dot(y))
        # xi_gradient += - anc_vec.T.dot(dB_dxi.dot(anc_vec)).reshape(-1)[:, None] / 8
        xi_gradient += - 2 * np.einsum('ij,jk->jik', anc_vec.T.dot(K_mn), dlambdaxi_dxi*K_nm.dot(anc_vec)).reshape(-1)[:, None] / 8
        # xi_gradient += - np.trace(dB_dxi.dot(B_inv), axis1=1, axis2=2)[:, None] / 2


        xi_gradient += - np.einsum('ij,ji->j', B_inv.dot(K_mn), dlambdaxi_dxi*K_nm)[:, None]

        xi_gradient += - K_ii * dlambdaxi_dxi
        xi_gradient += (np.einsum('ij,ji->i', K_nm, K_mm_inv.dot(K_mn))[:, None] * dlambdaxi_dxi)
        xi_gradient += -1/2
        xi_gradient += dlambdaxi_dxi * xi**2 + lambda_xi * xi * 2
        xi_gradient += expit(-xi)

        gradient = gradient + xi_gradient.reshape(-1).tolist()

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

    # def _get_prediction_quality(self, params, train_points, train_targets, test_points, test_targets):
    #     """
    #     Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
    #     method
    #     :param params: parameters
    #     :param test_points: test set points
    #     :param test_targets: test set target values
    #     :return: prediction MSE
    #     """
    #     new_gp = deepcopy(self)
    #     num_params = self.cov.get_params().size
    #     theta, xi = params[:num_params], params[num_params:][:, None]
    #     new_gp.cov.set_params(theta)
    #     cov_fun = new_gp.cov.covariance_function
    #     inputs = self.inducing_inputs[0]
    #     K_mm = cov_fun(inputs, inputs)
    #     K_nm = cov_fun(train_points, inputs)

    #     K_mm_inv, _ = _get_inv_logdet_cholesky(K_mm)
    #     mu, Sigma = self._recompute_var_parameters(K_mm_inv, K_nm, xi, train_targets)

    #     new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
    #     predicted_y_test = new_gp.predict(test_points)
    #     return 1 - np.sum(test_targets != predicted_y_test) / test_targets.size