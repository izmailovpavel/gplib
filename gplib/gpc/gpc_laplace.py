import numpy as np
import scipy as sp
import copy
import time
import scipy.optimize as op

from .gpc_base import GPC
from ..covfun.utility import sigmoid


class GPCLaplace(GPC):
    """
    A basic method for GP-classification, based on Laplace integral approximation
    """
    
    def __init__(self, cov_obj, mean_function=lambda x: 0):
        """
        :param cov_obj: object of the CovarianceFamily class
        :param mean_function: function, mean of the gaussian process
        """
        super(GPCLaplace, self).__init__(cov_obj, mean_function)

    @staticmethod
    def _get_ml(f_opt, b, cov_inv):
        """
        :param f_opt: posterior mode
        :param b: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        :param cov_inv: inverse covariance matrix at data points
        :return: the Laplace estimation of log marginal likelihood
        """
        l_b = np.linalg.cholesky(b)
        return -((f_opt.T.dot(cov_inv)).dot(f_opt) + 2 * np.sum(np.log(np.diag(l_b))))/2

    @staticmethod
    def _get_b(points_cov, hess_opt):
        """
        :param points_cov: covariance matrix
        :param hess_opt: -\nabla\nabla log p(y|f)|_{f_opt}
        :return: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        """
        w = sp.linalg.sqrtm(hess_opt)
        b = (w.dot(points_cov)).dot(w)
        return np.eye(b.shape[0]) + b

    @staticmethod
    def _get_laplace_approximation(labels, cov_inv, cov_l, max_iter=1000):
        """
        :param labels: label vector
        :param cov_inv: inverse covariance matrix at data points
        :param cov_l: cholesky decomposition matrix of the covariance matrix
        :param max_iter: maximum number of iterations for the optimization
        :return: tuple (posterior mode f_opt, unnormalized posterior hessian at f_opt)
        """
        def loss(f):
            """log p(y|f)"""
            f = f.reshape((f.size, 1))
            return (np.sum(np.log(np.exp(-labels * f) + np.ones(labels.shape))) + (f.T.dot(cov_inv)).dot(f)/2 +
                    2 * np.sum(np.log(np.diag(cov_l)))/2)

        def hessian(f):
            """Hessian of the log p(y|f)"""
            f = f.reshape((f.size, 1))
            diag_vec = (-np.exp(f) / np.square(np.ones(f.shape) + np.exp(f)))
            return -np.diag(diag_vec.reshape((diag_vec.size, ))) + cov_inv

        def grad(f):
            f = f.reshape((f.size, 1))
            return (-((labels + np.ones(labels.shape))/2 - sigmoid(f)) + cov_inv.dot(f)).reshape((f_0.size,))

        f_0 = np.zeros(labels.shape)
        f_0 = f_0.reshape((f_0.size,))
        f_res = op.minimize(loss, f_0, args=(), method='L-BFGS-B', jac=grad,
                            options={'gtol': 1e-5, 'disp': False, 'maxiter': max_iter})
        f_opt = f_res['x']
        return f_opt, hessian(f_opt) - cov_inv

    @staticmethod
    def _get_ml_partial_derivative(f_opt, k_inv, ancillary_mat, dk_dtheta_mat):
        """
        :param f_opt: posterior mode
        :param k_inv: inverse covariance matrix at data points
        :param ancillary_mat: (W^(-1) + K)^(-1), where W is hess_opt
        :param dk_dtheta_mat: the matrix of partial derivatives of the covariance function at data points
        with respect to hyper-parameter theta
        :return: marginal likelihood partial derivative with respect to hyper-parameter theta
        """
        return ((((f_opt.T.dot(k_inv)).dot(dk_dtheta_mat)).dot(k_inv)).dot(f_opt) -
                np.trace(ancillary_mat.dot(dk_dtheta_mat)))/2

    def _get_ml_grad(self, points, cov_inv, f_opt, anc_mat, params):
        """
        !! If the covariance function does not provide a derivative w.r.t. to the noise variance (the last parameter of
        the covariance function), it is assumed to be equal to 2 * noise variance * I. Else it is assumed to be
        derivative_matrix_list[-1] * I.
        :param points: data points array
        :param cov_inv: inverse covariance matrix
        :param params: hyper-parameters vector
        :param f_opt: posterior mode
        :param anc_mat: ancillary matrix
        :return: marginal likelihood gradient with respect to hyper-parameters
        """
        derivative_matrix_list = self.cov.get_derivative_function_list(params)
        noise_derivative = self.cov.get_noise_derivative(points.shape[0])
        # print(noise_derivative.shape)
        return np.array([self._get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               func(points, points))
                         for func in derivative_matrix_list] +
                        [self._get_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative)])

    def _oracle(self, points, f_opt, hess_opt, params):
        """
        :param points: data points array
        :param f_opt: posterior mode
        :param hess_opt: some weird hessian (-\nabla \nabla \log p(y|f_opt))
        :param params: hyper-parameters vector
        """
        cov_obj = copy.deepcopy(self.cov)
        cov_obj.set_params(params)
        # cov_fun = cov_obj
        points_cov = cov_obj(points, points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)

        b_mat = self._get_b(points_cov, hess_opt)
        marginal_likelihood = self._get_ml(f_opt, b_mat, points_cov_inv)

        anc_mat = np.linalg.inv(np.linalg.inv(hess_opt) + points_cov)
        gradient = self._get_ml_grad(points, points_cov_inv, f_opt, anc_mat, params)

        return marginal_likelihood, gradient

    def fit(self, points, labels, max_iter=10):
        """
        optimizes the self.covariance_obj hyper-parameters
        :param points: data points
        :param labels: class labels at data points
        :param max_iter: maximim number of iterations
        :return: a list of hyper-parameters values at different iterations and a list of times iteration-wise
        """
        cov_obj = copy.deepcopy(self.cov)

        bnds = self.cov.get_bounds()
        w0 = self.cov.get_params()
        w_list = []
        time_list = []

        def func(w):
            loss, grad = self._oracle(points, f_opt, hess_opt, w)
            return -loss, -grad

        start = time.time()
        for i in range(max_iter):
            points_cov = cov_obj(points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)

            f_opt, hess_opt = self._get_laplace_approximation(labels, points_cov_inv, points_l, max_iter=20)

            w_res = op.minimize(func, w0, args=(), method='L-BFGS-B', jac=True, bounds=bnds,
                                options={'ftol': 1e-5, 'disp': False, 'maxiter': 20})
            w0 = w_res['x']
            if not(i % 10):
                print("Iteration ", i)
                print("Hyper-parameters at iteration ", i, ": ", w0)

            w_list.append(w0)
            time_list.append(time.time() - start)
            cov_obj.set_params(w0)
        self.cov = copy.deepcopy(cov_obj)
        return w_list, time_list

    def predict(self, test_points, training_points, training_labels):
        """
        :param test_points: test data points
        :param training_points: training data points
        :param training_labels: class labels at training points
        :return: prediction of class labels at given test points
        """
        cov_fun = self.cov
        points_cov = cov_fun(training_points, training_points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)
        f_opt, hess_opt = self._get_laplace_approximation(training_labels, points_cov_inv, points_l)
        k_test_x = cov_fun(test_points, training_points)

        f_test = np.dot(np.dot(k_test_x, points_cov_inv), f_opt)
        loc_y_test = np.sign(f_test.reshape((f_test.size, 1)))

        return loc_y_test
