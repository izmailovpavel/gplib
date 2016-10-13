"""
Stochastic gradient descent
"""
import numpy as np
import time

from ..utility import project_into_bounds


def sgd(oracle, point, n, bounds=None, options=None):
    """
    Stochastic gradient descent optimization method for finite sums
    :param oracle: an oracle function, returning the gradient approximation by one data point,
    given it's index and the point
    :param point:
    :param n: number of training examples
    :param bounds: bounds on the variables
    :param options: a dictionary, containing the following fields
        'maxiter': maximum number of iterations
        'verbose': a boolean, showing weather or not to print the convergence info
        'print_freq': the frequency of the convergence messages
        'batch_size': the size of the mini-batch, used for gradient estimation
        'step0': initial step of the method
        'gamma': a parameter of the step length rule. It should be in (0.5, 1). The smaller it
        is, the more aggressive the method is
        'update_rate': the rate of shuffling the data points
    default options: {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'batch_size': 1,
                      'step0': 0.1, 'gamma': 0.55, 'update_rate':1}
    :return: optimal point
    """
    default_options = {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'batch_size': 1,
                      'step0': 0.1, 'gamma': 0.55, 'update_rate':1}
    if not options is None:
        default_options.update(options)
        if 'print_freq' in options.keys():
            default_options['verbose'] = True
    options = default_options

    batch_size = options['batch_size']
    step0 = options['step0']
    gamma = options['gamma']

    batch_num = int(n / batch_size)
    if n % batch_size:
        batch_num += 1
    update_rate = options['update_rate']

    indices = np.random.random_integers(0, n-1, (update_rate * batch_num * batch_size,))
    step = step0
    x = point
    x = project_into_bounds(x, bounds)
    x_lst = [np.copy(x)]
    time_lst = [0]
    start = time.time()
    for epoch in range(options['maxiter']):
        for batch in range(batch_num):
            new_indices = indices[range(batch_size*batch, (batch + 1)*batch_size)]
            grad = oracle(x, new_indices)
            x -= grad * step
            x = project_into_bounds(x, bounds)
        x_lst.append(np.copy(x))
        time_lst.append(time.time() - start)

        if not (epoch % update_rate):
            indices = np.random.random_integers(0, n-1, (update_rate * batch_num * batch_size,))

        if not (epoch % options['print_freq']) and options['verbose']:
            print("Epoch ", epoch, ":")
            print("\tStep:", step)
            print("\tParameters", x[:2])
        step = step0 / np.power((epoch+1), gamma)
    return x, x_lst, time_lst
