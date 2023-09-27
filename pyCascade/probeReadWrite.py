from pyCascade import utils
import pandas as pd
from dask import dataframe as dd
import os
import shutil
import numpy as np

def read_pointcloud_probes(filename):
    return dd.read_csv(filename, delim_whitespace=True)  # read as dataframe

def read_probes_csv(filename):
    return dd.read_csv(filename, delimiter = ' ', comment = "#",header = None, assume_missing=True, encoding = 'utf-8')

def read_probes(filename, file_type = 'csv'):
    if file_type == 'csv':
        ddf = read_probes_csv(filename)
    elif file_type == 'parquet':
        ddf = dd.read_parquet(filename)
    else:
        raise(f'file type {file_type} not supported')
    step_index = ddf.iloc[:, 0] #grab the first column for the indixes
    time_index = ddf.iloc[:, 1] #grab the second column for the times
    ddf = ddf.iloc[:, 3:] #take the data less the index rows

    _, n_cols = ddf.shape
    ddf = ddf.rename(columns=dict(zip(ddf.columns, np.arange(0, n_cols)))) #reset columns to integer 0 indexed
    ddf.columns.name = 'Stack'
    ddf.index = step_index
    return ddf, step_index, time_index

def read_locations(filename):
    locations = pd.read_csv(filename, delim_whitespace=True, comment = "#", names=['probe', 'x', 'y', 'z'], index_col = 'probe')
    probes = locations.index.values
    _, probe_ind = utils.last_unique(probes)
    return locations.iloc[probe_ind]

def csv_to_parquet(csv_path, parquet_path, overwrite = True):
    isParquet = os.path.exists(parquet_path)
    if isParquet:
        if overwrite:
            shutil.rmtree(parquet_path)
        else:
            return
    ddf = read_probes_csv(csv_path)
    ddf.index.name = "steps"
    st = utils.start_timer()
    ddf.columns = ddf.columns.astype(str)
    ddf.to_parquet(parquet_path) 
    return
    

class ProbesRW(utils.Helper):
    def __init__(self, directory, probe_type = "PROBES", file_type = "csv"):
        """
        File info is stored in a tuple-indexed dictionary. Once data is access, it is read in as a (nested) tuple-indexed dictionary.
        For POINTCLOUD_PROBES data is indexed as self.data[(name,step)][(stack, quant)]. For PROBES data is indexed as 
        self.data[(name, quant)][(stack, step)]. This format mimics the multiIndex DataFrame created in self.slice_into_df.
        """

        self.probe_type = probe_type
        self.file_type = file_type
        self.directory = directory
        self.directory_parquet = f'{directory}/../probesOut_parquet'
        
        if self.file_type == "csv":
            isExist = os.path.exists(self.directory_parquet)
            if not isExist:
                os.makedirs(self.directory_parquet)

        
        my_dict = {}  # this will be a tuple indexed 1-level dictionary.
        # create a generator to iterate over probe paths
        path_generator = glob.iglob(f'{directory}/*.*')
        probe_names = []
        probe_tbd1s = []
        probe_stack = np.array([])

        for path in path_generator:

            if self.probe_type == "POINTCLOUD_PROBES":
                if ".pcb" not in path:
                    continue
                file_name = path.split('/')[-1]  # get the local file name
                probe_info = file_name.split('.')
                probe_name, probe_tbd1, _ = probe_info[:]
                probe_tbd1 = int(probe_tbd1)
                # store the pcd path and pcd reader function
            elif self.probe_type == "PROBES":
                if "README" in path:
                    continue
                file_name = path.split('/')[-1]  # get the local file name
                probe_info = file_name.split('.')
                probe_name, probe_tbd1 = probe_info[:]
                # store the pcd path and pcd reader function
                if self.file_type == 'parquet':
                    path = f"{self.directory_parquet}/{probe_name}.{probe_tbd1}.parquet"
                    
                my_dict[(probe_name, probe_tbd1)], tbd2s, time = probeReadWrite.read_probes(path, self.file_type)
                    
                if 'col' in probe_name: # assuming the cols are run in all runs
                    probe_tbd2s = tbd2s
                    probe_time = time


            probe_names.append(probe_name)
            probe_tbd1s.append(probe_tbd1) 
       
        probe_tbd2s = probe_tbd2s.compute().values
        self.steps_written = len(probe_tbd2s)
        probe_tbd2s, unique_steps_indexes = utils.last_unique(probe_tbd2s)
        probe_times = probe_time.compute().values[unique_steps_indexes]

        self.data = my_dict

        self.probe_names = utils.sort_and_remove_duplicates(probe_names)  # remove duplicates
        # remove duplicates and sort
        probe_tbd1s = utils.sort_and_remove_duplicates(probe_tbd1s)

        # get the all quants and (max) stack across all probes
        if self.probe_type == "POINTCLOUD_PROBES":
            for name in self.probe_names:
                representative_df = my_dict[(name, probe_tbd1s[0])].compute()
                probe_stack = np.append(probe_stack, representative_df.columns.values)
                probe_tbd2s = np.append(probe_tbd2s, representative_df.index.values)

        # sort and remove duplicates
        probe_tbd2s = utils.sort_and_remove_duplicates(probe_tbd2s)
        # sort and remove duplicates
        self.probe_stack = utils.sort_and_remove_duplicates(probe_stack)
        if self.probe_type == "POINTCLOUD_PROBE":
            self.probe_steps = probe_tbd1s
            self.probe_quants = probe_tbd2s
        elif self.probe_type == "PROBES":
            self.probe_steps = probe_tbd2s
            self.probe_steps = [int(step) for step in self.probe_steps]
            self.probe_quants = probe_tbd1s
            self.probe_times = probe_times
            self.unique_steps_indexes = unique_steps_indexes


        self.data = my_dict
        
        self.data = my_dict

        self.dt = self.probe_times[-1]-self.probe_times[-2]
