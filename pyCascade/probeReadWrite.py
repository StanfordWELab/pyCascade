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