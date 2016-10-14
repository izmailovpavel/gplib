import numpy as np
import copy
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

from ..utility import _extract_and_delete, _lower_triang_mat_to_vec, _lower_triang_vec_to_mat
from ..optim.methods import climin_wrapper, sgd, scipy_wrapper
from ..gpres import GPRes


class SVIMethod():
    def __init__(self, cov, method_options):
        self.cov = copy.deepcopy(cov)
        hermgauss_deg = _extract_and_delete(method_options, 'hermgauss_deg', 100)
        self.gauss_hermite = self.gauss_hermite_precompute(hermgauss_deg)
        self.options = method_options

    def _get_parameter_vector(self, theta, eta_1, eta_2):
        """
        Transform given parameters to a vector, according to the used parametrization
        :param theta: vector or list
        :param eta_1: vector
        :param eta_2: matrix
        :return: a vector
        """
        theta = np.array(theta).reshape(-1)[:, None]
        eta_1 = eta_1.reshape(-1)[:, None]
        eta_2 = _lower_triang_mat_to_vec(eta_2)[:, None]
        return np.vstack((theta, eta_1, eta_2))[:, 0]

    def _get_parameters(self, vec):
        """
        Retrieve the parameters from the parameter vector vec of the same form, as the one, _get_parameter_vector
        produces.
        :param vec: a vector
        :return: a tuple of parameters
        """
        theta_len = len(self.cov.get_params())
        vec = vec[:, None]
        theta = vec[:theta_len, :]
        mat_size = np.int(np.sqrt(2 * (vec.size - theta_len) + 9 / 4) - 3 / 2)
        eta_1 = vec[theta_len:theta_len+mat_size, :].reshape((mat_size, 1))
        eta_2 = _lower_triang_vec_to_mat(vec[theta_len+mat_size:, :])
        return theta, eta_1, eta_2

    def _get_bounds(self, m):
        bnds = list(self.cov.get_bounds())
        bnds += [(None, None)] * m
        sigma_bnds = _lower_triang_mat_to_vec(np.eye(m)).tolist()
        for elem in sigma_bnds:
            if elem == 0:
                bnds.append((None, None))
            else:
                bnds.append((1e-2, None))
        return tuple(bnds)

    def fit(self, X, y, inputs):
        """
        A method for optimizing hyper-parameters (for fixed inducing points), based on stochastic variational inference
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
        mu = np.zeros((m, 1))
        sigma_L = np.eye(m)  # Cholesky factor of sigma

        theta = self.cov.get_params()
        param_vec = self._get_parameter_vector(theta, mu, sigma_L)

        bnds = self._get_bounds(m)

        def fun(w, i=None):
            full = False
            if i is None:
                full = True
                i = range(target_values.size)
            fun, grad = self._elbo_batch_approx_oracle(X, y, inputs, parameter_vec=w,
                                       indices=i)
            if full:
                return -fun, -grad[:, 0]
            else:
                return -grad[:, 0]

        def adadelta_fun(w, X, y):
            _, grad = self._elbo_batch_approx_oracle(X, y, inputs, parameter_vec=w,
                                       indices=range(y.size), N=n)
            return -grad[:, 0]

        opts = copy.deepcopy(self.options)
        mydisp = _extract_and_delete(opts, 'mydisp', False)
        mode = _extract_and_delete(opts, 'mode', 'adadelta')

        if mode == 'full':
            res, w_list, time_list = scipy_wrapper(fun, param_vec, method='L-BFGS-B', mydisp=mydisp,
                                                      bounds=bnds, jac=True, options=opts)
            res = res['x']
        elif mode == 'batch':
            res, w_list, time_list = sgd(oracle=fun, n=n, point=param_vec, bounds=bnds,
                                                                 options=opts)
        elif mode == 'adadelta':
            res, w_list, time_list = climin_wrapper(oracle=adadelta_fun, point=param_vec, train_points=X,
                                                    train_targets=y, options=opts, method='AdaDelta')

        theta, mu, sigma_L = self._get_parameters(res)
        sigma = sigma_L.dot(sigma_L.T)

        inducing_inputs = (inputs, mu, sigma)
        return inducing_inputs, theta, GPRes(param_lst=w_list, time_lst=time_list)

    # def _get_prediction_quality(self, params, test_points, test_targets):
    #     """
    #     Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
    #     method
    #     :param params: parameters
    #     :param test_points: test set points
    #     :param test_targets: test set target values
    #     :return: prediction MSE
    #     """
    #     new_gp = deepcopy(self)
    #     theta, mu, Sigma_L = new_gp._get_parameters(params)
    #     Sigma = Sigma_L.dot(Sigma_L.T)
    #     # theta = params[:len(new_gp.cov.get_params())]
    #     new_gp.cov.set_params(theta)
    #     new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
    #     predicted_y_test = new_gp.predict(test_points)
    #     return 1 - np.sum(test_targets != predicted_y_test) / test_targets.size
    #     # return f1_score(test_targets, predicted_y_test)

    def _elbo_batch_approx_oracle(self, data_points, target_values, inducing_inputs, parameter_vec,
                                       indices, N=None):
        """
        The approximation of Evidence Lower Bound (L3 from the article 'Scalable Variational Gaussian Process
        Classification') and it's derivative wrt kernel hyper-parameters and variational parameters.
        The approximation is found using a mini-batch.
        :param data_points: the array of data points
        :param target_values: the target values at these points
        :param inducing_inputs: an array of inducing inputs
        :param mu: the current mean of the process at the inducing points
        :param sigma: the current covariance of the process at the inducing points
        :param theta: a vector of hyper-parameters and variational parameters, the point of evaluation
        :param indices: a list of indices of the data points in the mini-batch
        :return: ELBO and it's gradient approximation in a tuple
        """
        if N is None:
            N = target_values.size
        m = inducing_inputs.shape[0]
        theta, mu, sigma_L = self._get_parameters(parameter_vec)
        sigma = sigma_L.dot(sigma_L.T)
        self.cov.set_params(theta)

        l = len(indices)
        i = indices
        y_i = target_values[i]
        x_i = data_points[i, :]

        # Covariance function and it's parameters
        cov_fun = self.cov
        params = self.cov.get_params()

        # Covariance matrices
        K_mm = cov_fun(inducing_inputs, inducing_inputs)
        try:
            L = np.linalg.cholesky(K_mm)
        except:
            print(params)
            exit(0)

        L_inv = np.linalg.inv(L)
        K_mm_inv = L_inv.T.dot(L_inv)
        k_i = cov_fun(inducing_inputs, x_i)
        K_ii = cov_fun(x_i[:1, :], x_i[:1, :])

        # Derivatives
        derivative_matrix_list = self.cov.get_derivative_function_list(params)
        d_K_mm__d_theta_lst = [fun(inducing_inputs, inducing_inputs) for fun in derivative_matrix_list]
        d_k_i__d_theta_lst = [fun(inducing_inputs, data_points[i, :]) for fun in derivative_matrix_list]
        d_K_mm__d_sigma_n = self.cov.get_noise_derivative(points_num=m)
        d_k_i__d_sigma_n = np.zeros((m, l))
        d_K_mm__d_theta_lst.append(d_K_mm__d_sigma_n)
        d_k_i__d_theta_lst.append(d_k_i__d_sigma_n)


        #f_i marginal distribution parameters

        m_i = k_i.T.dot(K_mm_inv.dot(mu))
        S_i = np.sqrt((K_ii + np.einsum('ij,ji->i', k_i.T, K_mm_inv.dot((sigma - K_mm).dot(K_mm_inv.dot(k_i))))).T)

        # Variational Lower Bound, estimated by the mini-batch
        loss = self._compute_log_likelihood_expectation(m_i, S_i, y_i)
        loss += - np.sum(np.log(np.diag(L))) * l / N
        loss += np.sum(np.log(np.diag(sigma_L))) * l / N
        loss += - np.trace(sigma.dot(K_mm_inv)) * l / (2*N)
        loss += - mu.T.dot(K_mm_inv.dot(mu)) * l / (2*N)


        # Gradient
        grad = np.zeros((len(theta,)))
        mu_expectations = self._compute_mu_grad_expectation(m_i, S_i, y_i)
        dL_dmu = (np.sum(mu_expectations * K_mm_inv.dot(k_i), axis=1)[:, None]
                  - K_mm_inv.dot(mu) * l / N)

        sigma_expectations = self._compute_sigma_l_grad_expectation(m_i, S_i, y_i)
        dL_dsigma_L = K_mm_inv.dot((k_i * sigma_expectations).dot(k_i.T.dot(K_mm_inv.dot(sigma_L)))) + \
                      np.eye(m) * l / (N * np.diag(sigma_L)) - (K_mm_inv.dot(sigma_L)) * l / N
        dL_dsigma_L = _lower_triang_mat_to_vec(dL_dsigma_L)

        for param in range(len(theta)):
            if param != len(theta) - 1:
                cov_derivative = derivative_matrix_list[param]
            else:
                cov_derivative = lambda x, y: self.cov.get_noise_derivative(points_num=1)

            d_K_mm__d_theta = d_K_mm__d_theta_lst[param]
            d_k_i__d_theta = d_k_i__d_theta_lst[param]
            grad[param] += np.einsum('ij,ji->', (d_k_i__d_theta * mu_expectations).T, K_mm_inv.dot(mu))
            grad[param] += - np.einsum('ij,ji->', (k_i * mu_expectations).T, K_mm_inv.dot(d_K_mm__d_theta.
                                                                                        dot(K_mm_inv.dot(mu))))
            grad[param] += cov_derivative(x_i[:1, :], x_i[:1, :]) * np.sum(sigma_expectations) / 2

            grad[param] += np.einsum('ij,ji->', d_k_i__d_theta.T, K_mm_inv.dot((sigma_L.dot(sigma_L.T) - K_mm).dot(
                K_mm_inv.dot(k_i * sigma_expectations))))
            grad[param] += - np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(
                (sigma_L.dot(sigma_L.T) - K_mm).dot(K_mm_inv.dot(k_i * sigma_expectations))))))
            grad[param] += - 1/2 * np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(d_K_mm__d_theta.dot(
                K_mm_inv.dot(k_i * sigma_expectations))))
            grad[param] += - np.trace(K_mm_inv.dot(d_K_mm__d_theta)) * l / (2*N)
            grad[param] += np.trace(sigma.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv)))) * l / (2*N)
            grad[param] += mu.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(mu)))) * l / (2*N)

        grad = grad[:, None]
        grad = np.vstack((grad, dL_dmu.reshape(-1)[:, None]))
        grad = np.vstack((grad, dL_dsigma_L.reshape(-1)[:, None]))
        return loss, grad

    @staticmethod
    def gauss_hermite_precompute(hermgauss_degree):
        """
        Precompute weights and points for Gauss-Hermite quadrature
        :return: None
        """
        points, weights = hermgauss(hermgauss_degree)
        return points, weights

    def _compute_expectation(self, variable, mu, sigma):
        """
        Computes the approximate expectation of a one-dimensional random variable with respect to a normal distribution
        with given parameters, using gauss-hermite quadrature.
        :param variable: the random variable under expectation
        :param mu: mean of the distribution
        :param sigma: std of the distribution
        :return: expectation
        """
        points, weights = self.gauss_hermite
        expectation = 0
        sqrt_two = np.sqrt(2)
        for weight, point in zip(weights, points):
            expectation += weight * variable(sqrt_two * point * sigma + mu)
        return expectation / np.sqrt(np.pi)

    def _compute_log_likelihood_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        mat = - (np.sqrt(2) * points * stds + means) * targets
        mat = np.log(expit(np.abs(mat))) - mat * (np.sign(mat) == 1)
        mat *= weights
        return np.sum(mat) / np.sqrt(np.pi)

    def _compute_mu_grad_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        # mat = targets / (1 + np.exp(targets * (np.sqrt(2) * points * stds + means)))
        mat = targets * (np.sqrt(2) * points * stds + means)
        mat = expit(-mat) * targets
        mat *= weights
        return (np.sum(mat, axis=1) / np.sqrt(np.pi))[None, :]

    def _compute_sigma_l_grad_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        anc_mat = targets * (np.sqrt(2) * points * stds + means)
        mat = -targets**2 * expit(anc_mat) * expit(-anc_mat)
        mat *= weights
        return (np.sum(mat, axis=1) / np.sqrt(np.pi))[None, :]
