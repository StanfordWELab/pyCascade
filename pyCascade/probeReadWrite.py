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
    
def readPointCloudProbes(pathGenerator):
    """
    This is the remenants of old functionality preserved for future use. This will need to be updated/fixed before use.
    """
    probe_names = []
    probe_steps = []
    
    # Initialize data structre
    my_dict = {}  # this will be a tuple indexed 1-level dictionary.

    for path in pathGenerator:
        if ".pcb" not in path:
            continue
        file_name = path.split('/')[-1]  # get the local file name
        probe_info = file_name.split('.')
        probe_name, probe_step, _ = probe_info[:]
        
        probe_step = int(probe_step)
        my_dict[(probe_name, probe_step)], _, _ = read_probes(path) # this function does not currently work for pointcloud probes

        probe_names.append(probe_name)
        probe_steps.append(probe_step) 

    probe_names = utils.sort_and_remove_duplicates(probe_names)
    
    # get the all quants and (max) stack across all probes
    for name in probe_names:
        representative_df = my_dict[(name, probe_steps[0])].compute()
        probe_stack = np.append(probe_stack, representative_df.columns.values)
        probe_quants = np.append(probe_quants, representative_df.index.values)

    return my_dict, probe_names, probe_steps, probe_quants, probe_stack


def readPointProbes(pathGenerator, file_type = 'csv', directory_parquet = None):
    probe_names = []
    probe_quants = []
    
    # Initialize data structre
    my_dict = {}  # this will be a tuple indexed 1-level dictionary.

    for path in pathGenerator:
        if "README" in path or ".pcp" in path or ".fp" in path:
            continue
        file_name = path.split('/')[-1]  # get the local file name
        probe_info = file_name.split('.')
        probe_name, probe_quant = probe_info[:]
        # store the pcd path and pcd reader function
        if file_type == 'parquet':
            path = f"{directory_parquet}/{probe_name}.{probe_quant}.parquet"
            
        my_dict[(probe_name, probe_quant)], step, time = read_probes(path, file_type)
            
        if 'col' in probe_name: # assuming the cols are run in all runs
            probe_steps = step
            probe_times = time
            probe_stack = my_dict[(probe_name, probe_quant)].columns.values

        probe_names.append(probe_name)
        probe_quants.append(probe_quant) 

    probe_steps = probe_steps.compute().values
    return my_dict, probe_names, probe_steps, probe_quants, probe_stack, probe_times