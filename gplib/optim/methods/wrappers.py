"""
Wrappers of different optimization libraries. Currently, climin and scipy.optimize.minimize
"""

import time
import numpy as np
import scipy.optimize as op
import climin

from ..utility import eig_val_correction, generate_constraint_matrix, project_into_bounds
from .optimizer_base import OptimizerBase

class LBFGS(OptimizerBase):
    """
    A wrapper-class for L-BFGS-B method from scipy.optimize. Requires function value and gradient.
    """
    def __init__(self, disp=False, maxiter=15000, maxfun=15000, gtol=1e-5, ftol=2.220446049250313e-09, 
                 maxls=20, maxcor=10):
        """
        :param disp: determines the method output.
        Other parameters are the options of scipy.optimize.minimize method for L-BFGS-B
        :param maxiter: maximum number of iterations
        :param maxfun: maximum number of function calls
        :param gtol: method stops, when projection of each component of the gradient
                on the set is less than gtol.
        :param frol: the method stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
        :param maxcor: maximum number of variable metric corrections used to define the limited memory matrix
        """
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.gtol = gtol
        self.ftol = ftol
        self.maxcor = maxcor
        OptimizerBase.__init__(self, disp)

    def minimize(self, fun, x_0, bounds=None):

        aux = {'start': time.time(), 'total': 0., 'it': 0}

        def callback(x):
            aux['total'] += time.time() - aux['start']
            if self.disp and not (aux['it'] % self.print_freq):
                print('Iteration', aux['it'], ':')
                print('\tx', x.reshape(-1)[:5])
            time_list.append(aux['total'])
            x_list.append(np.copy(x))
            aux['it'] += 1
            aux['start'] = time.time()

        x_list = []
        time_list = []
        callback(x_0)
        opts = {'disp': False, 'iprint': -1, 'gtol': self.gtol, 'eps': 1e-08, \
                'maxiter': self.maxiter, 'ftol': self.ftol, 'maxfun': self.maxfun}
        out = op.minimize(fun, x_0, method='L-BFGS-B', jac=True, callback=callback, options=opts)

        x = out.x
        stat_dict = {'time_lst': time_list, 'x_lst': x_list, 'fun': out.fun, 'time': time_list[-1], 
                     'info': out}
        return x.copy(), stat_dict


class AdaDelta(OptimizerBase):
    """
    A wrapper-class for AdaDelta method from climin library. Requires gradient estimation.
    """
    def __init__(self, disp=False, iter_per_epoch=1, n_epoch=1000, step_rate=1., decay=0.9, 
        momentum=0., offset=1e-4):
        """
        :param iter_per_epoch: number of iteration per epoch
        :param n_epoch: maximum number of epochs (or iterations if no sample_size is provided)
        The names of the other parameters are the same as in the corresponding climin method 
        :param step_rate: step size of the method
        :param decay: decay of the moving avrage; must lie in [0, 1)
        :param momentum: momentum
        :param offset: added before taking the sqrt of running averages
        """
        OptimizerBase.__init__(self, disp)
        self.n_epoch = n_epoch
        self.iter_per_epoch = iter_per_epoch
        self.maxiter = int(self.n_epoch * self.iter_per_epoch)
        self.print_freq = int(self.print_freq * self.iter_per_epoch)
        self.step_rate = step_rate
        self.decay = decay
        self.momentum = momentum
        self.offset = offset

    def minimize(self, fun, x_0, bounds=None):
        """
        Does not take bounds into account
        """
        x = np.copy(x_0).reshape(-1)
        opt = climin.Adadelta(wrt=x, fprime=fun, step_rate=self.step_rate, momentum=self.momentum,
                              decay=self.decay, offset=self.offset)

        x_list = [x.copy()]
        time_list = [0.]
        start = time.time()

        for info in opt:
            i = info['n_iter']
            if i > self.maxiter:
                break
            
            if self.disp and not (i % self.print_freq):
                    grad = info['gradient']
                    print('Epoch', int(i / self.iter_per_epoch), ':')
                    print('\tx', x.reshape(-1)[:5])
                    print("\tGradient norm", np.linalg.norm(grad))
            
            if not i % int(self.iter_per_epoch):
                x_list.append(x.copy())
                time_list.append(time.time() - start)

        stat_dict = {'time_lst': time_list, 'x_lst': x_list, 'fun': None, 'time': time_list[-1], 
                     'info': info}

        return x.copy(), stat_dict
