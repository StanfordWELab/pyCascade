import time
import numpy as np
from IPython.core.debugger import set_trace

class Helper:
    def get_input(self, input):
        if isinstance(input, str):
            input = eval(input)
        return input
        
def ax_index(ax, i, j):
    n_dims = np.array(ax).ndim
    if n_dims == 0:
        sub_ax = ax
    elif n_dims == 1:
        sub_ax = ax[max(i, j)]
    else:
        sub_ax = ax[i, j]
    return sub_ax

def eval_tuple(value):
    if isinstance(value, tuple):
        function, arg = value  # retrieve data reading function and data path
        # read in the data, and assign it to the dict value
        value = function(arg)
    return value

def sort_and_remove_duplicates(l):
    l = [*set(l)]
    l.sort()
    return l

def dict_apply(f):
    return lambda d: {k: f(v) for k, v in d.items()}

def start_timer(description = None):
    if description != None:
        print(description)
    return time.time()

def end_timer(st, description):
    et = time.time()
    elapsed_time = round(et - st)
    print(f"{description} took {elapsed_time} seconds")


def last_unique(a):
    n_ind = len(a)
    a, ind_unique = np.unique(np.flip(a), return_index = True)
    ind_unique = n_ind-1-ind_unique
    
    return a, ind_unique
    

# class MyLazyDict(dict):
#     '''
#     Create a lazy dictionary by modifying the __getitem__ attribute. New dictionary dynamically reads in data as it is accessed,
#     and memorizes data once it has been read in.
#     '''

#     def __getitem__(self, item):
#         # retrieve the current dictionary value
#         value = dict.__getitem__(self, item)
#         if isinstance(value, tuple):  # check if data has been read in
#             # print('reading in data')
#             value = eval_tuple(value)
#             # reset the dictionary value to the data
#             dict.__setitem__(self, item, value)
#         return value

