"""
Wrappers of different optimization libraries. Currently, climin and scipy.optimize.minimize
"""

import time
import numpy as np
import scipy.optimize as op
import climin
from ..utility import eig_val_correction, generate_constraint_matrix, project_into_bounds


def scipy_wrapper(oracle, point, mydisp=False, print_freq=1, jac=True, **kwargs):
    """
    A wrapper function for scipy.optimize.minimize
    :param oracle: function, being optimized or a tuple (function, gradient)
    :param point: initial point of optimization
    """
    aux = {'start': time.time(), 'total': 0., 'it': 0}

    def callback(w):
        aux['total'] += time.time() - aux['start']
        if mydisp and not (aux['it'] % print_freq):
            print("Hyper-parameters at iteration", aux['it'], ":", w.reshape(-1)[:5])
            # if w.size > 5:
            #     print('...')
        time_list.append(aux['total'])
        w_list.append(np.copy(w))
        aux['it'] += 1
        aux['start'] = time.time()
        # if aux['it'] == kwargs['options']['maxiter']:
        #     return np.copy(w), w_list, time_list
    w_list = []
    time_list = []
    callback(point)

    out = op.minimize(oracle, point, jac=jac, callback=callback, **kwargs)

    return out, w_list, time_list


def climin_wrapper(oracle, point, options, method='AdaDelta'):
    """
    A wrapper function for climin optimization library
    :param oracle: function, being optimized or a tuple (function, gradient)
    :param point: initial point of optimization
    """
    default_options = {'maxiter': 1000, 'print_freq':1, 'verbose': False, 'g_tol': 1e-5,
                       'batch_size': 10, 'step_rate': 0.1}
    if not options is None:
        default_options.update(options)
        if 'print_freq' in options.keys():
            default_options['verbose'] = True
    options = default_options

    w = point.copy()
    # data = ((i, {}) for i in iter_minibatches([train_points, train_targets], options['batch_size'], [0, 0]))

    if method == 'AdaDelta':
        opt = climin.Adadelta(wrt=w, fprime=oracle, step_rate=options['step_rate'])
    elif method == 'SG':
        opt = climin.GradientDescent(wrt=w, fprime=oracle, step_rate=options['step_rate'])
    else:
        raise ValueError('Unknown optimizer')

    w_lst = [w.copy()]
    time_lst = [0.]
    start = time.time()
    n_epochs = options['maxiter']
    train_size = options['train_size']
    n_iterations = int(n_epochs * train_size / options['batch_size'])
    print_freq = int(options['print_freq'] * train_size / options['batch_size'])

    if options['verbose']:
        print('Using ' + method + ' optimizer')
    for info in opt:
        i = info['n_iter']
        if i > n_iterations:
            break
        if not (i % print_freq) and options['verbose']:
            grad = info['gradient']
            print("Iteration ", int(i * options['batch_size'] / train_size), ":")
            print("\tGradient norm", np.linalg.norm(grad))
        if not i % int(train_size / options['batch_size']):
            w_lst.append(w.copy())
            time_lst.append(time.time() - start)

    return w.copy(), w_lst, time_lst
