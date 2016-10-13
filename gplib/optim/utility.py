"""
Utility functions, related to optimization
"""
import numpy as np


def project_into_bounds(point, bounds):
    """
    Project the given point into the given bounds
    :param bounds:
    :param point:
    :return:
    """
    if bounds is None:
        return point
    low_bounds = [bound[0] for bound in bounds]
    high_bounds = [bound[1] for bound in bounds]
    proj = np.copy(point)
    i = 0
    for coord, l_bound, h_bound in list(zip(point, low_bounds, high_bounds)):
        if not(l_bound is None):
            if coord < l_bound:
                proj[i] = l_bound
        if not(h_bound is None):
            if coord > h_bound:
                proj[i] = h_bound
        i += 1
    return proj


def eig_val_correction(mat, eps=1e-2):
    """
    Corrects the matrix, so that it becomes simmetric positive-definite, based on eigenvalue decomposition.
    :param mat: matrix to be corrected
    :param eps: the minimum eigenvalue allowed
    :return: a positive-definite simmetric matrix
    """
    mat = (mat + mat.T)/2
    w, v = np.linalg.eigh(mat)
    w[w < eps] = eps
    new_mat = v.dot(np.diag(w).dot(np.linalg.inv(v)))
    return (new_mat + new_mat.T)/2


def generate_constraint_matrix(bounds, x_old=None):
    """
    Generates a constraint matrix and right-hand-side vector for the cvxopt qp-solver.
    :param bounds: list of bounds on the optimization variables
    :param x_old: the vector of values, that have to be substracted from the bounds
    :return: the matrix G and the vector h, such that the constraints are equivalent to G x <= h.
    """
    if bounds is None:
        return None, None
    num_variables = len(bounds)
    if x_old is None:
        x_old = np.zeros((num_variables, 1))
    elif len(x_old.shape) == 1:
        x_old = x_old[:, None]
    G = np.zeros((1, num_variables))
    h = np.zeros((1, 1))
    for i in range(num_variables):
        bound = bounds[i]
        a = bound[0]
        b = bound[1]
        if not (a is None):
            new_line = np.zeros((1, num_variables))
            new_line[0, i] = -1
            G = np.vstack((G, new_line))
            h = np.vstack((h, np.array([[-a + x_old[i, 0]]])))
        if not (b is None):
            new_line = np.zeros((1, num_variables))
            new_line[0, i] = 1
            G = np.vstack((G, new_line))
            h = np.vstack((h, np.array([[b - x_old[i, 0]]])))
    if G.shape[0] == 1:
        return None, None
    G = G[1:, :]
    h = h[1:, 0]
    return G, h


def check_gradient(oracle, point, hess=False, print_diff=False, delta=1e-6, indices=None):
    """
    Prints the gradient, calculated with the provided function
    and approximated via a finite difference.
    :param oracle: a function, returning the loss and it's grad given point
    :param point: point of calculation
    :param hess: a boolean, showing weather or not to check the hessian
    :param print_diff: a boolean. If true, the method prints all the entries of the true and approx.
    gradients
    :return:
    """
    if not indices:
        indices = range(point.size)
    fun, grad = oracle(point)[:2]
    app_grad = np.zeros(grad.shape)
    if print_diff:
        print('Gradient')
        print('Approx.\t\t\t\t Calculated')
    for i in indices:
        point_eps = np.copy(point)
        point_eps[i] += delta
        app_grad[i] = (oracle(point_eps)[0] - fun) / delta
        if print_diff:
            print(app_grad[i], '\t', grad[i])
    print('\nDifference between calculated and approximated gradients')
    print(np.linalg.norm(app_grad.reshape(-1) - grad.reshape(-1)))

    if hess:
        fun, grad, hess = oracle(point)
        app_hess = _approximate_hessian(oracle, point)
        if print_diff:
            print('Hessian')
            print('Approx.\t\t\t\t Calculated')
        if print_diff:
            for i in range(point.size):
                print(app_hess[:, i], '\t', hess[:, i])
        print('\nDifference between calculated and approximated hessians')
        print(np.linalg.norm(app_hess.reshape(-1) - hess.reshape(-1)))


def approximate_hessian(oracle, point):
    app_hess = np.zeros((point.size, point.size))
    fun, grad = oracle(point)[:2]
    for i in range(point.size):
        point_eps = np.copy(point)
        point_eps[i] += 1e-6
        if len(grad.shape) == 2:
            app_hess[:, i] = ((oracle(point_eps)[1] - grad) * 1e6)[:, 0]
        else:
            app_hess[:, i] = ((oracle(point_eps)[1] - grad) * 1e6)
        app_hess = (app_hess + app_hess.T)/2
    return app_hess