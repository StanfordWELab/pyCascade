from pyCascade import utils
import pandas as pd
from dask import dataframe as dd
import os
import shutil
import numpy as np
from IPython.core.debugger import set_trace


def read_pointcloud_probes(filename):
    return dd.read_csv(filename, delim_whitespace=True)  # read as dataframe

def read_probes_file_switch(filename, file_type = 'csv'):
    if file_type == 'csv':
        ddf = dd.read_csv(filename, delimiter = ' ', comment = "#",header = None, assume_missing=True, encoding = 'utf-8')
    elif file_type == 'parquet':
        ddf = dd.read_parquet(filename)
    else:
        raise(f'file type {file_type} not supported')
    return ddf 

def read_probes(filename, file_type = 'csv'):
    ddf = read_probes_file_switch(filename, file_type)
    step_index = ddf.iloc[:, 0] #grab the first column for the indixes
    time_index = ddf.iloc[:, 1] #grab the second column for the times
    ddf = ddf.iloc[:, 3:] #take the data less the index rows

    _, n_cols = ddf.shape
    ddf = ddf.rename(columns=dict(zip(ddf.columns, np.arange(0, n_cols)))) #reset columns to integer 0 indexed
    ddf.columns.name = 'Stack'
    ddf.index = step_index
    return ddf, step_index, time_index

def read_flux_probes(filename, file_type = 'csv', quants = None):
    ddf = read_probes_file_switch(filename, file_type)
    step_index = ddf.iloc[:, 1] #grab the first column for the indixes
    time_index = ddf.iloc[:, 2] #grab the second column for the times
    location = ddf.iloc[:, 3:6] #grab the locations
    location.columns = ['x', 'y', 'z']
    area = ddf.iloc[:, 6] #grab the area
    ddf = ddf.iloc[:, 8::2] #take the data less other rows
    if quants == None:
        _, n_cols = ddf.shape
        ddf = ddf.rename(columns=dict(zip(ddf.columns, np.arange(0, n_cols)))) #reset columns to integer 0 indexed
    else:
        ddf = ddf.rename(columns=dict(zip(ddf.columns, quants)))

    ddf.columns.name = 'Flux Quants'
    ddf.index = step_index
    return ddf, step_index, time_index, location, area

def read_vol_probes(filename, file_type = 'csv', quants = None):
    ddf = read_probes_file_switch(filename, file_type)
    step_index = ddf.iloc[:, 0] #grab the first column for the indixes
    time_index = ddf.iloc[:, 1] #grab the second column for the times
    ddf = ddf.iloc[:, 3::2] #take the data less other rows
    if quants == None:
        _, n_cols = ddf.shape
        ddf = ddf.rename(columns=dict(zip(ddf.columns, np.arange(0, n_cols)))) #reset columns to integer 0 indexed
    else:
        ddf = ddf.rename(columns=dict(zip(ddf.columns, quants)))

    ddf.columns.name = 'Flux Quants'
    ddf.index = step_index
    return ddf, step_index, time_index

def read_locations(filename, file_type):
    if file_type == 'csv':
        locations = pd.read_csv(filename, delim_whitespace=True, comment = "#", header = None)
    elif file_type == 'parquet':
        locations = pd.read_parquet(filename)
    else:
        raise(f'file type {file_type} not supported')
    locations.columns = ['probe', 'x', 'y', 'z']
    locations.set_index('probe', inplace = True)
    probes = locations.index.values
    _, probe_ind = utils.last_unique(probes)
    return locations.iloc[probe_ind]

def csv_to_parquet(csv_path, parquet_path, overwrite = False):
    isParquet = os.path.exists(parquet_path)
    if isParquet:
        if overwrite:
            shutil.rmtree(parquet_path)
        else:
            return
    ddf = read_probes_file_switch(csv_path)
    ddf.index.name = "steps"
    ddf.columns = ddf.columns.astype(str)
    ddf.to_parquet(parquet_path) 
    return

    
def readPointCloudProbes(pathGenerator):
    """
    This is the remenants of old functionality preserved for future use. This will need to be updated/fixed before use.
    """
    probe_names = []
    probe_steps = []
    probe_paths = []
    
    # Initialize data structre
    my_dict = {}  # this will be a tuple indexed 1-level dictionary.

    for path in pathGenerator:
        path = path.replace(".parquet", '')
        if ".pcb" not in path:
            continue
        file_name = path.split('/')[-1]  # get the local file name
        probe_info = file_name.split('.')
        probe_name, probe_step, _ = probe_info[:]
        
        probe_step = int(probe_step)
        my_dict[(probe_name, probe_step)] = read_pointcloud_probes(path)

        probe_names.append(probe_name)
        probe_steps.append(probe_step) 
        probe_paths.append(path)

    probe_names = utils.sort_and_remove_duplicates(probe_names)
    
    # get the all quants and (max) stack across all probes
    for name in probe_names:
        representative_df = my_dict[(name, probe_steps[0])].compute()
        probe_stack = np.append(probe_stack, representative_df.columns.values)
        probe_quants = np.append(probe_quants, representative_df.index.values)

    return my_dict, probe_names, probe_steps, probe_quants, probe_stack, probe_paths


def readPointProbes(pathGenerator, file_type = 'csv', directory_parquet = None):
    probe_names = []
    probe_quants = []
    probe_paths = []
    
    # Initialize data structre
    my_dict = {}  # this will be a tuple indexed 1-level dictionary.
    locations = {}

    for path in pathGenerator:
        if ".pcp" in path or ".fp" in path:
            continue
        probe_paths.append(path)
        file_name = path.split('/')[-1]  # get the local file name
        probe_info = file_name.replace(".parquet", '')
        probe_info = probe_info.split('.')
        probe_name, probe_quant = probe_info[:]
        if probe_quant == 'README':
            locations[probe_name] = read_locations(path, file_type)
            continue
        # store the pcd path and pcd reader function
            
        my_dict[(probe_name, probe_quant)], step, time = read_probes(path, file_type)
            
        if 'col' in probe_name: # assuming the cols are run in all runs
            probe_steps = step
            probe_times = time
            probe_stack = my_dict[(probe_name, probe_quant)].columns.values

        probe_names.append(probe_name)
        probe_quants.append(probe_quant) 

    probe_steps = probe_steps.compute().values
    probe_names = utils.sort_and_remove_duplicates(probe_names)

    return my_dict, probe_names, probe_steps, probe_quants, probe_stack, probe_times, locations, probe_paths


def readBulkProbes(pathGenerator, file_type = 'csv', directory_parquet = None, quants = None, probe_type = "FLUX_PROBES"):
    if probe_type == "FLUX_PROBES":
        probe_file_type = '.fp'
    elif probe_type == "VOLUMETRIC_PROBES":
        probe_file_type = '.svp'
    probe_names = []
    probe_quants = []
    probe_paths = []
    
    # Initialize data structre
    my_dict = {}  # this will be a tuple indexed 1-level dictionary.
    locations = {}
    areas = {}
    
    gotTimes = False

    for path in pathGenerator:
        if probe_file_type not in path:
            continue
        file_name = path.split('/')[-1]  # get the local file name
        probe_info = file_name.replace(".parquet", '')
        probe_info = probe_info.split('.')
        probe_name = probe_info[0]
        # store the pcd path and pcd reader function
            
        if probe_type == "FLUX_PROBES":
            ddf, step, time, locations[probe_name], areas[probe_name] = read_flux_probes(path, file_type, quants)
        elif probe_type == "VOLUMETRIC_PROBES":
            ddf, step, time = read_vol_probes(path, file_type, quants)
            
        if gotTimes == False:
            probe_steps = step.compute().values
            probe_times = time
            probe_quants = ddf.columns.values
            gotTimes = True
        
        for quant in ddf.columns.values:
            my_dict[(probe_name, quant)] = ddf[[quant]]

        probe_names.append(probe_name)
        probe_paths.append(path)
        
    if gotTimes == False:
        raise Exception("Probes not found")

    return my_dict, probe_names, probe_steps, probe_quants, probe_times, locations, areas, probe_paths