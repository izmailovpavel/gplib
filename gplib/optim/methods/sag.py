"""
Stochastic average gradient method
"""
import numpy as np
import time
from ..linesearch import armiho
from ..utility import project_into_bounds


def sag(oracle, point, n, bounds=None, options=None):
    """
    Stochastic average gradient (SAG) optimization method for finite sums
    :param oracle: an oracle function, returning the gradient approximation by one data point,
    given it's index and the point
    :param point: initial point of optimization
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
    default options: {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'batch_size': 1,
                      'step0': 0.1, 'gamma': 0.55}
    :return: optimal point
    """
    default_options = {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'batch_size': 1,
                      'step0': 0.1, 'gamma': 0.55}
    if not options is None:
        default_options.update(options)
        if 'print_freq' in options.keys():
            default_options['verbose'] = True
    options = default_options

    batch_size = options['batch_size']
    l = 1.0
    eps = 0.5

    def update_lipschitz_const (l, point, cur_loss=None, cur_grad=None):
        if cur_loss is None or cur_grad is None:
            cur_loss, cur_grad = batch_oracle(point)
        l *= np.power(2.0, - 1 / batch_oracle.batch_num)
        if l <= 1:
            l = 1
        new_point = point - cur_grad / l
        new_point = project_into_bounds(new_point, bounds)
        new_loss, _ = batch_oracle(new_point)

        while new_loss > cur_loss - eps * cur_grad.T.dot(cur_grad) / l:
            l *= 2
            new_point = point - cur_grad / l
            new_point = project_into_bounds(new_point, bounds)
            new_loss, _ = batch_oracle(new_point)
            if l > 1e16:
                print('Abnormal termination in linsearch')
                return 0
        return l

    class BatchOracle:
        def __init__(self, n, batch_size):
            self.num_funcs = n
            self.batch_size = batch_size
            self.batch_num = int(n / batch_size)
            if n % batch_size:
                self.batch_num += 1
            self.gradients = np.zeros((self.batch_num, point.size))
            self.current_grad = np.zeros(point.shape)
            self.cur_index = 0
            self.cur_batch_index = 0

        def update_gradients(self, new_grad):
            self.current_grad += (new_grad - self.gradients[self.cur_batch_index].reshape(point.shape)) / self.batch_num
            self.gradients[self.cur_batch_index] = new_grad.reshape(point.shape[0], )
            self.cur_index += batch_size
            self.cur_batch_index += 1
            if self.cur_batch_index > self.batch_num - 1:
                self.cur_index = 0
                self.cur_batch_index = 0
            return self.current_grad

        def __call__(self, eval_point):
            if self.cur_index + self.batch_size < n:
                indices = range(self.cur_index, self.cur_index + self.batch_size)
            else:
                indices = list(range(self.cur_index, n-1)) + list(range(self.cur_index + self.batch_size - n + 1))

            new_loss, new_grad = oracle(eval_point, indices)
            return new_loss, new_grad

    x = point
    x = project_into_bounds(x, bounds)
    batch_oracle = BatchOracle(n=n, batch_size=batch_size)
    x_lst = [np.copy(x)]
    time_lst = [0]
    start = time.time()

    for epoch in range(options['maxiter']):
        for i in range(batch_oracle.batch_num):
            loss, grad = batch_oracle(x)
            l = update_lipschitz_const(l, x, cur_loss=loss, cur_grad=grad)
            direction = batch_oracle.update_gradients(grad)
            if l == 0:
                return x
            x -= direction / l#(16 * l)
            x = project_into_bounds(x, bounds)
        x_lst.append(np.copy(x))
        time_lst.append(time.time() - start)
        if not (epoch % options['print_freq']) and options['verbose']:
            print("Epoch ", epoch, ":")
            print("\tLipschitz constant estimate:", l)
            print("\t", x[:2])    # print(x_lst)
    return x, x_lst, time_lst