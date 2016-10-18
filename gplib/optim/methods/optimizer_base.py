import numbers
from abc import ABCMeta, abstractmethod

class OptimizerBase:
	"""
	Abstract base class, defining the interface of optimization methods
	"""
	
	def __init__(self, disp):
		"""
		:param disp: determines the output of the method progress. If False, doesn't print anything. If True, 
				prints output at each iteration. If equal to an integer number l, prints output every 
				l iterations. 
		"""
		self.disp = disp
		self.print_freq = int(disp)
		if isinstance(self.disp, numbers.Integral):
			self.print_freq = disp
			self.disp = True

	@abstractmethod
	def minimize(self, fun, x_0, bounds=None):
		"""
		Minimizes given function, starting from x_0
		:param fun: function; should return the optimized function value and/or
				it's gradient, given point; shapes of the argument and the gradient 
				should be (k,)
		:param x_0: initial point (numpy array of shape (k,))
		:paran bounds: bounds on variables, a list of tuples [(p1_min, p1_max), ..., (pk_min, pk_max)]
				if a bound (e.g. pi_min) is None, than it's considered, that there is no bound on 
				the corresponding variable. If the hole bounds parameter is None, than it's considered, 
				that there are no bounds at all.

		:return point: the final point
		:return stat_dict: dictionary, containing the following fields
				- time_lst — list, containing time stamps of all iterations of the method
				- x_lst — list, containing the parameter values at all iterations
				- fun — function value at termination point, if avalible
				- time — time, consumed by the method
				- info — other info, returned by the method
		"""
		pass

	def maximize(self, fun, x_0, bounds=None):
		def new_fun(x):
			fun_ans = fun(x)
			if hasattr(fun_ans, '__iter__'):
				return [-elem for elem in fun(x)]
			else:
				return -fun_ans
		x, stat = self.minimize(new_fun, x_0, bounds)
		if 'fun' in stat.keys():
			stat['fun'] *= -1
		if 'grad' in stat.keys():
			stat['grad'] *= -1
		return x, stat


