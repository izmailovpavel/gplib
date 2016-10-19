import numpy as np
import copy
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from climin.util import iter_minibatches

from ..utility import _extract_and_delete, _lower_triang_mat_to_vec, _lower_triang_vec_to_mat, \
                      _get_inv_logdet_cholesky
from ..gpres import GPRes
from ..elbo import ELBO


class SVI(ELBO):
    """
    Class representing the svi method's ELBO
    """

    def __init__(self, X, y, inputs, cov, batch_size, hermgauss_deg=100):
        """
        :param X: data points (possibly a batch for some methods)
        :param y: target values
        :param inputs: inducing inputs (positions)
        :param cov: covariance function
        :param batch_size: size of the mini batch on which the ELBO is computed
        :param hermgauss_deg: degree of Gauss-Hermite approximation
        """
        ELBO.__init__(self, X, y, 'svi')
        self.cov = cov
        self.inputs = inputs
        m = inputs.shape[0]
        self.mu = np.zeros((m, 1))
        self.SigmaL = np.eye(m)
        self.batches = (i for i in iter_minibatches([X, y], batch_size, [0, 0]))
        self.gauss_hermite = hermgauss(hermgauss_deg)

    @staticmethod
    def _get_parameter_vector(theta, eta_1, eta_2):
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

    def elbo(self, params_vec, return_fun=False):
        """
        The approximation of Evidence Lower Bound (L3 from the article 'Scalable Variational Gaussian Process
        Classification') and it's derivative wrt kernel hyper-parameters and variational parameters.
        The approximation is found using a mini-batch.
        :param params_vec: vector of parameters of the model 
        :return: ELBO and it's gradient approximation in a tuple
        """
        N = self.y.size
        m = self.inputs.shape[0]

        theta, mu, sigma_L = self._get_parameters(params_vec)
        sigma = sigma_L.dot(sigma_L.T)
        self.cov.set_params(theta)

        X_batch, y_batch = next(self.batches)
        l = y_batch.size

        # Covariance function and it's parameters
        cov_fun = self.cov
        params = self.cov.get_params()

        # Covariance matrices
        K_mm = cov_fun(self.inputs, self.inputs)
        
        K_mm_inv, K_mm_logdet = _get_inv_logdet_cholesky(K_mm)


        k_i = cov_fun(self.inputs, X_batch)
        K_ii = cov_fun(X_batch[:1, :], X_batch[:1, :])

        # Derivatives
        derivative_matrix_list = self.cov.get_derivative_function_list(params)
        d_K_mm__d_theta_lst = [fun(self.inputs, self.inputs) for fun in derivative_matrix_list]
        d_k_i__d_theta_lst = [fun(self.inputs, X_batch) for fun in derivative_matrix_list]
        d_K_mm__d_sigma_n = self.cov.get_noise_derivative(points_num=m)
        d_k_i__d_sigma_n = np.zeros((m, l))
        d_K_mm__d_theta_lst.append(d_K_mm__d_sigma_n)
        d_k_i__d_theta_lst.append(d_k_i__d_sigma_n)


        #f_i marginal distribution parameters

        m_i = k_i.T.dot(K_mm_inv.dot(mu))
        S_i = np.sqrt((K_ii + np.einsum('ij,ji->i', k_i.T, K_mm_inv.dot((sigma - K_mm).dot(K_mm_inv.dot(k_i))))).T)

        # Variational Lower Bound, estimated by the mini-batch
        loss = self._compute_log_likelihood_expectation(m_i, S_i, y_batch)
        # loss += - np.sum(np.log(np.diag(L))) * l / N
        loss += - K_mm_logdet * l / (2 * N)
        loss += np.sum(np.log(np.diag(sigma_L))) * l / N
        loss += - np.trace(sigma.dot(K_mm_inv)) * l / (2*N)
        loss += - mu.T.dot(K_mm_inv.dot(mu)) * l / (2*N)


        # Gradient
        grad = np.zeros((len(theta,)))
        mu_expectations = self._compute_mu_grad_expectation(m_i, S_i, y_batch)
        dL_dmu = (np.sum(mu_expectations * K_mm_inv.dot(k_i), axis=1)[:, None]
                  - K_mm_inv.dot(mu) * l / N)

        sigma_expectations = self._compute_sigma_l_grad_expectation(m_i, S_i, y_batch)
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
            grad[param] += cov_derivative(X_batch[:1, :], X_batch[:1, :]) * np.sum(sigma_expectations) / 2

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
        if return_fun:
            return -loss, -grad.reshape(-1)
        return -grad.reshape(-1)

    def recompute_parameters(self, n_upd=5):
        """
        Does nothing
        """
        pass

    def get_params_opt(self):
        theta = self.cov.get_params()
        param_vec = self._get_parameter_vector(np.copy(theta), np.copy(self.mu), np.copy(self.SigmaL))
        return param_vec

    def set_params_opt(self, params):
        theta, self.mu, self.SigmaL = self._get_parameters(params)
        self.cov.set_params(theta)

    def get_bounds_opt(self):
        # bnds = list(self.cov.get_bounds())
        # bnds = np.array(bnds+ [(1e-3, np.inf)] * self.get_params().size)
        return None

    def get_inputs(self):
        """
        Returns the inducing input positions, mean and covariance matrix
        of the process at these points
        """
        return self.inputs, self.mu, self.SigmaL.dot(self.SigmaL.T)

    def get_state(self):
        return self.get_params_opt()

    def set_state(self, params):
        self.set_params_opt(params)


