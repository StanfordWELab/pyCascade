import pandas as pd
import glob
import time

def read_pcd(filename):
    return pd.read_csv(filename, delim_whitespace=True)
            
class MyProbeDict(dict):
    '''
    Create a lazy dictionary by modifying the __getitem__ attribute. New dictionary dynamically reads in data as it is accessed,
    and memorizes data once it has been read in.
    '''
    def __getitem__(self, item):
        value=dict.__getitem__(self, item) # retrieve the current dictionary value
        if not isinstance(value, pd.core.frame.DataFrame): # check if data has been read in
            # print('reading in probe data')
            function, arg = value # retrieve data reading function and data path
            value = function(arg) # read in the data, and assign it to the dict value
            dict.__setitem__(self, item, value) # reset the dictionary value to the data
        return value


class Probes:
    def __init__(self, directory):

        my_dict= {}
        path_generator = glob.iglob(f'{directory}/*.pcd') # create a generator to iterate over probe paths

        for path in path_generator:

            file_name = path.split('/')[-1] # get the local file name
            probe_info = file_name.split('.')
            probe_name, probe_number, _ = probe_info[:]
            probe_number = int(probe_number)

            # create a dictionary for each probe name if it does not exist
            if probe_name not in my_dict.keys():
                my_dict[probe_name] = {}

            my_dict[probe_name][probe_number] = (read_pcd, path) # store the pcd path and pcd reader function

        #iterate through the upper data dict
        for probe_name, name_dict in my_dict.items():
            my_dict[probe_name] = MyProbeDict(name_dict) # modify the getter or the lower-level dicts to lazily read in data
        self.data = my_dict


st = time.time()
test = Probes('./CHARLES/probes/probesOut')
et = time.time()

print(et-st)
print(test.data['x_1over6'][79743]['comp(u,0)'])
print(test.data['x_1over6'][79742]['comp(u,0)'])