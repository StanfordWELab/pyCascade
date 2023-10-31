from pyCascade import utils, quantities, probeReadWrite

import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from dask import dataframe as dd
from pandarallel import pandarallel
import scipy as sp
import os
from IPython.core.debugger import set_trace


def ddf_to_MIseries(ddf):
    """
    Function to read in data directly instead of accessing it through indexing the lazy dictionary. 
    This speeds up parrallel processing because less infromation is passed to subprocesses.
    """
    return ddf.compute().transpose().stack()

def ddf_to_pdf(df):
    if isinstance(df, (dd.core.DataFrame, dd.core.Series, dd.core.Scalar)): 
        df = df.compute()
    return df


def mean_convergence(data_dict, t_data = None):
    def df_func(data_df):
        time_sum = data_df.cumsum(axis='index')
        cum_avg = time_sum.div(time_sum.index, axis='index')  # cumumlative averge
        last_time = cum_avg.index[-1]
        last_avg = cum_avg.loc[last_time]
        data_diff = cum_avg - last_avg.values
        data_diff_norm = np.abs(data_diff/last_avg.values)
        return data_diff_norm

    return utils.dict_apply(df_func)(data_dict)


def time_average(data_dict, t_data = None):
    df_func = lambda df: df.mean(axis='index')
    return utils.dict_apply(df_func)(data_dict)

def time_rms(data_dict, t_data = None):
    def df_func(data_df):
        mean = data_df.mean(axis='index')
        norm_data = data_df - mean
        diff_squared = (norm_data**2)
        rms = np.sqrt(diff_squared.mean())
        return rms

    return utils.dict_apply(df_func)(data_dict)


def ClenshawCurtis_Quadrature(data_dict, t_data=None):
    N = 10
    interval = 2.5
    xs = [np.cos((2*(N-k)-1)*np.pi/(2*N)) for k in range(N)]
    A = np.array([[xs[i]**j for i in range(N)] for j in range(N)])
    b = [(1+(-1)**k)/(k+1) for k in range(N)]
    ws = np.linalg.solve(A,b)
    CC_weights = np.squeeze(ws[:,None])

    Quad_weights = np.tile(CC_weights, 10) * np.repeat(CC_weights, 10) * (interval/2)**2

    def df_func(data_df):
        wighted_data = data_df*Quad_weights
        integrated_data = wighted_data.sum(axis=1)
        return integrated_data

    return utils.dict_apply(df_func)(data_dict)

def linear_quadrature(data_dict, t_data=None):
    # for k, v in data_dict.items():
    #     name = k[0]
    #     probe_locations = locations_dict[name]
    #     # Sort the probe locations in the correct order
    #     probe_locations = sorted(probe_locations, key=lambda x: (x[1], x[0]))

    #     # Calculate the intervals between the probe locations
    #     intervals = [np.sqrt((probe_locations[i+1][0]-probe_locations[i][0])**2 + (probe_locations[i+1][1]-probe_locations[i][1])**2) for i in range(len(probe_locations)-1)]

    #     # Calculate the massflow at each interval
    #     massflows = []
    #     for i in range(len(intervals)):
    #         start_probe = probe_locations[i]
    #         end_probe = probe_locations[i+1]
    #         start_data = data_dict[("PROBES", start_probe)]
    #         end_data = data_dict[("PROBES", end_probe)]
    #         massflow = (end_data - start_data) / intervals[i]
    #         massflows.append(massflow)

    #     # Calculate the integrated massflow
    #     integrated_massflow = sum(massflows)

    # Define a function to apply to each data frame
    def df_func(data_df):
        return data_df.mean(axis='columns')

    # Apply the function to the data dictionary
    return utils.dict_apply(df_func)(data_dict)

# use to define lambda function with mul preset
def mul_names(data_dict, names, mul, t_data=None):
    for k, v in data_dict.items():
        name, _ = k
        if name in names:
            data_dict[k] = mul*v
    return data_dict

def quick_dict_apply(df_func):
    return lambda data_dict, t_data=None: utils.dict_apply(df_func)(data_dict)

def quick_apply(func):
    return lambda data_dict, t_data=None: func(data_dict)

class Probes(utils.Helper):
    def __init__(self, directory, probe_type = "PROBES", file_type = "csv", flux_quants = None):
        """
        File info is stored in a tuple-indexed dictionary. Once data is access, it is read in as a (nested) tuple-indexed dictionary.
        For POINTCLOUD_PROBES data is indexed as self.data[(name,step)][(stack, quant)]. For PROBES data is indexed as 
        self.data[(name, quant)][(stack, step)].
        """
        
        # Set class properties  
        self.probe_type = probe_type
        self.file_type = file_type
        self.directory = directory
        self.directory_parquet = f'{directory}/../probesOut_parquet'

        # Prepare paruet directory if reading csvs 
        if self.file_type == "csv":
            isExist = os.path.exists(self.directory_parquet)
            if not isExist:
                os.makedirs(self.directory_parquet)

        # create a generator to iterate over probe paths
        
        path_generator = glob.iglob(f'{directory}/*.*')

        # get data dict and associated info 
        if self.probe_type == "POINTCLOUD_PROBES":
            self.data, probe_names, probe_steps, probe_quants, probe_stack, self.probe_paths = probeReadWrite.readPointCloudProbes(path_generator)
        elif self.probe_type == "PROBES":
            self.data, probe_names, probe_steps, probe_quants, probe_stack, probe_times, self.locations, self.probe_paths = probeReadWrite.readPointProbes(path_generator, self.file_type, self.directory_parquet)
        elif self.probe_type == "FLUX_PROBES":
            self.data, probe_names, probe_steps, probe_quants, probe_times, self.locations, self.areas, self.probe_paths = probeReadWrite.readFluxProbes(path_generator, self.file_type, self.directory_parquet, quants = flux_quants)
            probe_stack = []
            
        self.steps_written = len(probe_steps)
        # remove steps and times that were written twice during run restarts
        if self.probe_type == "PROBES" or self.probe_type == "FLUX_PROBES":
            probe_steps, self.unique_steps_indexes = utils.last_unique(probe_steps)
            self.probe_times = probe_times.compute().iloc[self.unique_steps_indexes]
            self.probe_times.index = probe_steps
            self.probe_steps = [int(step) for step in probe_steps]
            self.dt = self.probe_times.iloc[-1]-self.probe_times.iloc[-2]
        else:
            self.probe_steps = utils.sort_and_remove_duplicates(probe_steps)

        # remove duplicates and sort
        self.probe_names = utils.sort_and_remove_duplicates(probe_names)
        self.probe_quants = utils.sort_and_remove_duplicates(probe_quants)
        self.probe_stack = utils.sort_and_remove_duplicates(probe_stack)
        
    def to_parquet(
        self,
        overwrite = False,
        names = "self.probe_names",
        quants = "self.probe_quants"):

        quants, names = [self.get_input(input) for input in [quants, names]]
        
        #write parquet files
        if self.file_type == "csv":
            for csv_path in self.probe_paths:
                local_path = csv_path.split('/')[-1]
                parquet_path = f"{self.directory_parquet}/{local_path}.parquet"
                st = utils.start_timer()
                probeReadWrite.csv_to_parquet(csv_path, parquet_path, overwrite)
                if overwrite: 
                    utils.end_timer(st, f"writing {parquet_path}")

    def get_flux_probe_loc_area(self, name):
        location = self.locations[name]
        area = self.areas[name]
        if isinstance(location, (dd.core.DataFrame, dd.core.Series, dd.core.Scalar)):
            location = location.loc[0].compute() #compute for only step 0
            location = location.iloc[-1] #get values from last step 0 (in case of restart)
            location.index = ['x', 'y', 'z'] # set location index
            self.locations[name] = location
        if isinstance(area, (dd.core.DataFrame, dd.core.Series, dd.core.Scalar)):
            area = area.loc[0].compute()
            area = area.iloc[-1]
            self.areas[name] = area
        
        
    def process_data(
        self, 
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "np.s_[::]",
        processing = None):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]
        t_data = self.probe_times.loc[steps]
        st = utils.start_timer()
        processed_data  = {}
        for name in names:
            for quant in quants:
                ddf = self.data[(name, quant)]
                df = ddf.compute()
                processed_data[(name, quant)] = df[stack]#.loc[steps[0]:steps[-1]]

            if self.probe_type == "FLUX_PROBES":
                self.get_flux_probe_loc_area(name)

        def index_unique_steps(df):
            if isinstance(df, (pd.core.frame.DataFrame, pd.core.series.Series)):
                shift = self.steps_written - len(df.index)
                unique_indexes = np.array(self.unique_steps_indexes - shift)
                unique_indexes = unique_indexes[unique_indexes >= 0]
                unique_indexes = unique_indexes.tolist()
                
                df = df.iloc[unique_indexes]
                df = df.loc[steps]
            return df
    
        processed_data = utils.dict_apply(index_unique_steps)(processed_data)

        if processing is not None:
            for i, process_step in enumerate(processing):
                # set_trace()
                processed_data = process_step(processed_data, t_data)
        
        utils.end_timer(st, 'processing data')
        return processed_data
    
    def create_qty_dict(
            self, 
            theta_wind = 0,
            names = "self.probe_names", 
            steps = "self.probe_steps", 
            quants = "self.probe_quants", 
            stack = "np.s_[::]", 
            processing = None):
        
        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]
        
        qty_dict = {}
        for name in names:
            qty = quantities.Qty()
            processed_data = self.process_data([name], steps, quants, stack, processing)
            qty.computeQty(processed_data,
                           self.probe_times[steps],
                           theta_wind = theta_wind,
                           u_str = (name, 'comp(u,0)'), 
                           v_str = (name, 'comp(u,1)'), 
                           w_str = (name, 'comp(u,2)'), 
                           p_str = (name, 'p'), 
                           calc_stats = True)
            qty.set_y(self.locations[name]['y'].values[stack])
            qty_dict[name] = qty
        return qty_dict


    def contour_plots(
        self,
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "np.s_[::]",
        parrallel = False,
        processing = None,
        plot_params={}
    ):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]

        n_names = len(names)
        n_quants = len(quants)

        processed_data = self.process_data(names, steps, quants, stack, processing)

        st = utils.start_timer()

        # plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(n_names, n_quants, constrained_layout =True)

        for j, quant in enumerate(quants):
            if 'plot_levels' in plot_params and quant in plot_params['plot_levels']:
                plot_levels = plot_params['plot_levels'][quant]
            else:
                plot_levels = 1000

            ax_list = []
            im_list = []
            vmins = [] # for colorbar
            vmaxs = []
            for i, name in enumerate(names):
                plot_df = processed_data[(name, quant)]
                # plot_df = self.data[(name, quant)].compute().iloc[steps]
                plot_df = plot_df.transpose()
                plot_df = plot_df.dropna()
                sub_ax = utils.ax_index(ax, i, j)
                
                xPlot = plot_df.columns
                if 'horizontal spacing' in plot_params:
                    if hasattr(plot_params['horizontal spacing'], "__len__"):
                        x_ind = [int(x) for x in xPlot]
                        xPlot = plot_params['horizontal spacing'][x_ind]
                    else:
                        xPlot *= plot_params['horizontal spacing']
                yPlot = plot_df.index
                if hasattr(self, 'locations') and 'stack span' in plot_params:
                    location = self.locations[name]
                    yAxis = location[plot_params['stack span']]
                    yPlot = yAxis[yPlot]

                if 'veritcal scaling' in plot_params:
                   yPlot*=plot_params['veritcal scaling']

                if 'plot_every' in plot_params:  # usefull to plot subset of timesteps but run calcs across all timesteps
                    plot_df = plot_df.iloc[:,::plot_params['plot_every']]
                    xPlot =xPlot[::plot_params['plot_every']]

                # x_mesh, y_mesh = np.meshgrid(xPlot, yPlot)
                im = sub_ax.contourf(xPlot, yPlot, plot_df, levels=plot_levels)
                ax_list.append(sub_ax)
                im_list.append(im)

                if j > 0:
                    sub_ax.yaxis.set_visible(False)
                else:
                    sub_ax.set_ylabel(name)
                if i < n_names-1:
                    sub_ax.xaxis.set_visible(False)
                else:
                    sub_ax.set_xlabel(quant)

                vmin, vmax = im.get_clim()
                vmins.append(vmin)
                vmaxs.append(vmax)
                # fig.colorbar(im, ax = sub_ax)

            vmin = min(vmins)
            vmax = max(vmaxs)
            if 'ColorNorm' in plot_params:
                if plot_params['ColorNorm'] == "TwoSlope":
                    norm = colors.TwoSlopeNorm(0,vmin,vmax)
                elif plot_params['ColorNorm'] == "Centered":
                    vmagmax = max(np.abs((vmin, vmax)))
                    norm = colors.TwoSlopeNorm(0,-vmagmax,vmagmax)

            else:
                norm = colors.Normalize(vmin,vmax)
            for im in im_list:
                im.set_norm(norm)
            fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax_list)
        
        if 'xlabel' in plot_params:
            fig.supxlabel(plot_params['xlabel'])
        if 'ylabel' in plot_params:
            fig.supylabel(plot_params['ylabel'])
        utils.end_timer(st, "plotting")
        # plt.figure()
        # plt.contourf(xPlot, yPlot, plot_data, plot_levels = plot_params['plot_levels'])
        # fig.show()

        return fig, ax

    def profile_plots(
        self,
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "np.s_[::]",
        processing = None,
        parrallel = False,
        plot_params = {}
        ):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]

        processed_data = self.process_data(names, steps, quants, stack, processing)

        fig, ax = plt.subplots(1, 1, constrained_layout =True)
        for j, quant in enumerate(quants):
            for i, name in enumerate(names):
                plot_df = ddf_to_pdf(processed_data[(name, quant)])
                plot_df = plot_df.dropna()
                # if isinstance(plot_df, pd.DataFrame):
                #     plot_df = plot_df.droplevel('name', axis='columns')

                yPlot = plot_df.index
                if hasattr(self, 'locations') and 'stack span' in plot_params:
                    location = self.locations[name]
                    yAxis = location[plot_params['stack span']]
                    yPlot = yAxis[yPlot]

                if 'veritcal scaling' in plot_params:
                   yPlot*=plot_params['veritcal scaling']

                ax.plot(plot_df.values, yPlot, label=f'{name}: {quant}')
                if 'xlabel' in plot_params:
                    fig.supxlabel(plot_params['xlabel'])
                if 'ylabel' in plot_params:
                    fig.supylabel(plot_params['ylabel'])
                ax.legend()

    
    
    def time_plots(
        self,
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "np.s_[::]",
        parrallel = False,
        processing = None,
        plot_params={}
    ):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]

        processed_data = self.process_data(names, steps, quants, stack, processing)

        st = utils.start_timer()

        # plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(1, 1, constrained_layout =True)

        for j, quant in enumerate(quants):
            for i, name in enumerate(names):
                plot_df = ddf_to_pdf(processed_data[(name, quant)])
                
                xPlot = plot_df.index
                if 'horizontal spacing' in plot_params:
                    if hasattr(plot_params['horizontal spacing'], "__len__"):
                        x_ind = [int(x) for x in xPlot]
                        xPlot = plot_params['horizontal spacing'][x_ind]
                    else:
                        xPlot *= plot_params['horizontal spacing']

                if 'plot_every' in plot_params:  # usefull to plot subset of timesteps but run calcs across all timesteps
                    plot_df = plot_df.iloc[:,::plot_params['plot_every']]
                    xPlot =xPlot[::plot_params['plot_every']]

                yPlot =  np.squeeze(plot_df.values)
                if 'veritcal scaling' in plot_params:
                   yPlot*=plot_params['veritcal scaling']

                ax.plot(xPlot, yPlot, label=f'{name}: {quant}')
                if 'xlabel' in plot_params:
                    fig.supxlabel(plot_params['xlabel'])
                if 'ylabel' in plot_params:
                    fig.supylabel(plot_params['ylabel'])
                ax.legend()
        

        return fig, ax

    def statistics(
        self,
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "np.s_[::]",
        parrallel = False,
        processing = None,
        plot_params={}
    ):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]

        processed_data = self.process_data(names, steps, quants, stack, processing)
        processed_data = utils.dict_apply(ddf_to_pdf)(processed_data)
        df_data = pd.Series(processed_data).unstack()
        if df_data.shape[1] == 1:
            df_data = df_data.iloc[:,0] # convert to series
            df_data = df_data.map(lambda x : x.to_numpy()[0])
        return df_data


