"""
Full gradient descent
"""
import numpy as np
import time
from ..linesearch import armiho
from ..utility import project_into_bounds

def fgd(oracle, point, bounds=None, options=None):
    """
    Gradient descent optimization method
    :param oracle: oracle function, returning the function value and it's gradient, given point
    :param point: point
    :param bounds: bounds on the variables
    :param options: a dictionary, containing some of the following fields
        'maxiter': maximum number of iterations
        'verbose': a boolean, showing weather or not to print the convergence info
        'print_freq': the frequency of the convergence messages
        'g_tol': the tolerance wrt gradient. If the gradient at the current point is
        smaller than the tolerance, the method stops
        'step_tol': tolerance wrt the step length. If the step length at current
        iteration is less than tolerance, the method stops.
        'maxstep': the maximum allowed step length
    default options: {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1.0}
    :return: the point with the minimal function value found
    """
    default_options = {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1.0}
    if not options is None:
        default_options.update(options)
        if 'print_freq' in options.keys():
            default_options['verbose'] = True
    options = default_options

    step = 1.0
    x = point
    loss_fun = lambda w: oracle(w)[0]
    x_lst = [np.copy(x)]
    time_lst = [0]
    start = time.time()

    for i in range(options['maxiter']):
        x = project_into_bounds(x, bounds)
        loss, grad = oracle(x)
        if np.linalg.norm(grad) < options['g_tol']:
            if options['verbose']:
                print("Gradient norm reached the stopping criterion")
            break
        x, step = armiho(fun=loss_fun, gradient=grad, point_loss=loss, bounds=bounds, point=x,
                                     step_0=step, maxstep=options['maxstep'])
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