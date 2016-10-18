"""
Full gradient descent
"""
import numpy as np
import time
from ..linesearch import armiho
from ..utility import project_into_bounds
from .optimizer_base import OptimizerBase

class FGD(OptimizerBase):
    """
    Full gradient descent method with Armiho line search rule
    """

    def __init__(self, disp, maxiter=15000, gtol=1e-5, step_tol=1e-16, maxstep=1e3):
        """
        :param maxiter: maximum number of iterations
        :param gtol: method stops, when projection of each component of the gradient
                on the set is less than gtol.
        :param step_tol: method stops, when the step (chosen by linesearch) becomes 
                smaller than step_tol
        :param maxstep: maximum step length of the method
        """
        OptimizerBase.__init__(self, disp)
        self.maxiter = maxiter
        self.gtol = gtol
        self.step_tol = step_tol
        self.maxstep = maxstep


    def minimize(self, fun, x_0, bounds=None):

        step = 1.0
        x = np.copy(x_0).reshape(-1)
        loss_fun = lambda w: fun(w)[0]
        x_lst = [np.copy(x)]
        time_lst = [0]
        start = time.time()

        for i in range(self.maxiter):
            x = project_into_bounds(x, bounds)
            loss, grad = fun(x)
            if np.max(np.abs(project_into_bounds(x + grad, bounds) - x)) < self.gtol:
                if self.disp:
                    print("Gradient projection reached the stopping criterion")
                break
            
            x, step = armiho(fun=loss_fun, gradient=grad, point_loss=loss, bounds=bounds, point=x,
                                         step_0=step, maxstep=self.maxstep)
            x_lst.append(np.copy(x))
            time_lst.append(time.time() - start)
            if step < self.step_tol:
                if self.disp:
                    print("Step length reached the stopping criterion")
                break
            if not (i % self.print_freq) and self.disp:
                print("Iteration ", i, ":")
                print("\tGradient projection norm", 
                      np.linalg.norm(project_into_bounds(x + grad, bounds) - x))
                print("\tFunction value", loss)
        
        stat_dict = {'time_lst': time_lst, 'x_lst': x_lst, 'fun': loss, 'time': time_lst[-1], 
                     'info': {'grad_proj_norm': np.linalg.norm(project_into_bounds(x + grad, bounds) - x),
                               'grad': np.copy(grad), 'step': step}}

        return x.copy(), stat_dict
