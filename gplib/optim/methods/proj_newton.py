import numpy as np
import time
import cvxopt

from ..linesearch import armiho
from ..utility import project_into_bounds, eig_val_correction, approximate_hessian, generate_constraint_matrix
from .optimizer_base import OptimizerBase

class ProjNewton(OptimizerBase):
    """
    Projected Newton method for bound-constrained problems. Uses finite-difference Hessian
    approximation if it is not provided. Uses Armiho rule for line search.
    """

    def __init__(self, disp=False, maxiter=1000, gtol=1e-5, step_tol=1e-16, maxstep=1e3, 
                 qp_abstol=1e-5):
        """
        :param maxiter: maximum number of iterations
        :param gtol: method stops, when projection of each component of the gradient
                on the set is less than gtol.
        :param step_tol: method stops, when the step (chosen by linesearch) becomes 
                smaller than step_tol
        :param maxstep: maximum step length of the method
        :param qp_abstol: the tolerance for the qp sub-problem
        """
        OptimizerBase.__init__(self, disp)
        self.maxiter = maxiter
        self.gtol = gtol
        self.step_tol = step_tol
        self.maxstep = maxstep
        self.qp_abstol = qp_abstol

    def minimize(self, fun, x_0, bounds=None):
        step = 1.0
        x = np.copy(x_0)
        x = x.reshape(x.size, 1)
        x_lst = [np.copy(x)]
        time_lst = [0]
        start = time.time()

        def new_fun(w):
            ans = list(fun(w.reshape(-1)))
            ans[1] = ans[1][:, None]
            return ans[1]

        loss_fun = lambda w: new_fun(w)[0]

        for i in range(self.maxiter):
            x = project_into_bounds(x, bounds)
            x = x.astype(float)
            oracle_answer = new_fun(x)

            if len(oracle_answer) == 3:
                loss, grad, hess = oracle_answer
            elif len(oracle_answer) == 2:
                loss, grad = oracle_answer
                hess = eig_val_correction(approximate_hessian(new_fun, x), eps=1e-5)
            else:
                raise ValueError('Oracle must return 2 or 3 values')

            if np.max(np.abs(project_into_bounds(x + grad, bounds) - x)) < self.gtol:
                if self.disp:
                    print("Gradient projection reached the stopping criterion")
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
            # cvxopt.solvers.options['maxiters'] = options['maxiter']
            cvxopt.solvers.options['abstol'] = self.qp_abstol
            if not (G is None):
                solution = cvxopt.solvers.qp(P, q, G, h)
            else:
                solution = cvxopt.solvers.qp(P, q)
            direction = np.array(solution['x'], dtype=float)

            # step
            x, step = armiho(fun=loss_fun, gradient=grad, point_loss=loss, bounds=bounds, point=x,
                                         step_0=1.0, maxstep=self.maxstep, direction=direction)
            x_lst.append(np.copy(x))
            time_lst.append(time.time() - start)
            
            if step < self.step_tol:
                if self.disp:
                    print("Step length reached the stopping criterion")
                break

            if self.disp and not (i % self.print_freq):
                print("Iteration ", i, ":")
                print("\tGradient projection norm", 
                      np.linalg.norm(project_into_bounds(x + grad, bounds) - x))
                print("\tFunction value", loss)

        stat_dict = {'time_lst': time_lst, 'x_lst': x_lst, 'fun': loss, 'time': time_lst[-1], 
                     'info': {'grad_proj_norm': np.linalg.norm(project_into_bounds(x + grad, bounds) - x),
                               'grad': np.copy(grad), 'step': step}}

        return x.copy(), stat_dict
