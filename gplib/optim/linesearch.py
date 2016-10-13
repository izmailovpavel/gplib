"""
Module, implementing linesearch methods for optimisation
"""
import numpy as np
from .utility import project_into_bounds

def armiho(fun, point, gradient, bounds=None, step_0=1.0, theta=0.5, eps=1e-2,
                       direction=None, point_loss=None, maxstep=np.inf):
    """
    Line search using Armiho rule
    :param fun: function, being optimized or a tuple (function, gradient)
    :param point: the point of evaluation
    :param direction: direction of the optimization
    :param gradient: gradient at the point point
    :param step_0: initial step length
    :param theta: theta parameter for updating the step length
    :param eps: parameter of the armiho rule
    :param point_loss: fun value at the point point
    :return: (new point, step) — a tuple, containing the chosen step length and
    the next point for the optimization method
    """
    if point_loss is None:
        current_loss = fun(point)
    else:
        current_loss = point_loss
    if direction is None:
        direction = -gradient

    step = step_0/theta
    while step > maxstep:
        step *= theta
    new_point = point + step * direction
    new_point = project_into_bounds(new_point, bounds)
    while fun(new_point) > current_loss + eps * step * direction.T.dot(gradient):
        step *= theta
        new_point = point + step * direction
        new_point = project_into_bounds(new_point, bounds)
    return new_point, step