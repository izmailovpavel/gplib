class GPRes:
    """A class, for the objects, returned by gaussian process fitting methods"""

    def __init__(self, param_lst, iteration_lst=None, time_lst=None):
        self.params = param_lst
        self.iters = iteration_lst
        if iteration_lst is None:
            self.iters = range(len(self.params))
        self.times = time_lst

    def __str__(self):
        ans = '\nParameter values list:\n'
        ans += str(self.params)
        ans += '\nIteration numbers:\n'
        ans += str(list(self.iters))
        ans += '\nTimes at these iterations:\n'
        ans += str(self.times)
        ans += '\n'
        return ans

    def plot_performance(self, metrics, it_time='i', freq=1, verbose=False, print_freq=1):
        y_lst=[]
        j = 0
        for i in range(len(self.params)):
            if not (i % freq):
                y_lst.append(metrics(self.params[i]))
                j += 1
                if verbose and not (j % print_freq):
                    print('Processing parameter number%d/%d'%(i+1, int(len(self.params) / print_freq)))

        if it_time == 'i':
            x_lst = [self.iters[i] for i in range(len(self.iters)) if not(i%freq)]
        elif it_time == 't':
            x_lst = [self.times[i] for i in range(len(self.times)) if not(i%freq)]
        elif it_time == 'it':
            x_lst_i = [self.iters[i] for i in range(len(self.iters)) if not(i%freq)]
            x_lst_t = [self.times[i] for i in range(len(self.times)) if not(i%freq)]
            return x_lst_i, x_lst_t, y_lst
        else:
            raise ValueError('Wrong it_time')
        return x_lst, y_lst
