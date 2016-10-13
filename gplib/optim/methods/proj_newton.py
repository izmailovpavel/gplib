import numpy as np
import time
import cvxopt

from ..linesearch import armiho
from ..utility import project_into_bounds, eig_val_correction, approximate_hessian, generate_constraint_matrix

def proj_newton(oracle, point, bounds=None, options=None):
    """
    Projected Newton method for bound-constrained problems.
    :param oracle: Oracle function, returning the function value, the gradient and the hessian the given point. If it
    doesn't provide a hessian, a finite difference is used to approximate it.
    :param point: starting point for the method
    :param bounds: bounds on the variables
    :param options: a dictionary, containing some of the following fields
        'maxiter': maximum number of iterations
        'verbose': a boolean, showing weather or not to print the convergence info
        'print_freq': the frequency of the convergence messages
        'g_tol': the tolerance wrt gradient. If the gradient at the current point is
        smaller than the tolerance, the method stops
        'step_tol': the tolerance wrt the step length. If the step length at current
        iteration is less than tolerance, the method stops.
        'maxstep': the maximum allowed step length
        'qp_abstol': the tolerance for the qp sub-problem
    default options: {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1.0}
    :return:
    """
    default_options = {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1, 'qp_abstol':1e-5}
    if not options is None:
        default_options.update(options)
        if 'print_freq' in options.keys():
            default_options['verbose'] = True
    options = default_options

    step = 1.0
    x = np.copy(point)[:, None]
    loss_fun = lambda w: oracle(w)[0]
    x_lst = [np.copy(x)]
    time_lst = [0]
    start = time.time()

    for i in range(options['maxiter']):
        x = project_into_bounds(x, bounds)
        oracle_answer = oracle(x)

        if len(oracle_answer) == 3:
            loss, grad, hess = oracle_answer
        elif len(oracle_answer) == 2:
            loss, grad = oracle_answer
            hess = eig_val_correction(approximate_hessian(oracle, point), eps=1e-5)
        else:
            raise ValueError('Oracle must return 2 or 3 values')

        if np.linalg.norm(grad) < options['g_tol']:
            if options['verbose']:
                print("Gradient norm reached the stopping criterion")
            break

        hess = hess.astype(float)
        grad = grad.astype(float)

        # The qp-subproblem
        P = hess
        q = grad
        G, h = generate_constraint_matrix(bounds, x)
        P, q = cvxopt.matrix(P), cvxopt.matrix(q)
        if not (G is None):
            G, h = cvxopt.matrix(G), cvxopt.matrix(h)
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = options['maxiter']
        cvxopt.solvers.options['abstol'] = options['qp_abstol']
        if not (G is None):
            solution = cvxopt.solvers.qp(P, q, G, h)
        else:
            solution = cvxopt.solvers.qp(P, q)
        dir = np.array(solution['x'])

        # step
        x, step = armiho(fun=loss_fun, gradient=grad, point_loss=loss, bounds=bounds, point=x,
                                     step_0=1.0, maxstep=options['maxstep'], direction=dir)
        x_lst.append(np.copy(x))
        time_lst.append(time.time() - start)
        if step < options['step_tol']:
            if options['verbose']:
                print("Step length reached the stopping criterion")
            break

        if not (i % options['print_freq']) and options['verbose']:
            print("Iteration ", i, ":")
            print("\tGradient norm", np.linalg.norm(grad))
            print("\tFunction value", loss)

    return x, x_lst, time_lst