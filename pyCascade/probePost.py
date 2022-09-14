from pyCascade import utils

import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import pandas as pd
import pickle

from pyspark.sql import SparkSession

def read_probes(filename):
    df = pd.read_csv(filename, delim_whitespace=True)  # read as dataframe
    return df.stack().to_dict()  # save as tuple indexed dictionary


def read_locations(filename):
    return pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['x', 'y', 'z'])


def parallel_functions(value):
    """ 
    Function to read in data directly instead of accessing it through indexing the lazy dictionary. 
    This speeds up parrallel processing because less infromation is passed to subprocesses.
    """
    return pd.Series(utils.eval_tuple(value))

def mean_convergence(data_df):
    # n_steps = len(data_df.groupby(axis='columns', level='step').size())
    time_sum = data_df.groupby(
        axis='columns', level='name').cumsum(axis='columns')
    # data_steps = pd.Series(np.arange(1, n_steps+1))
    data_df_index = list(zip(*data_df.keys()))  # unzip list of tuples
    data_steps = [*set(data_df_index[1])] # sort and remove duplicates
    n_names = len([*set(data_df_index[0])])

    var_cum_avg = time_sum.div(
        np.tile(data_steps, n_names), axis='columns', level='step')  # cumumlative averge
    var_last_avg = var_cum_avg.groupby(axis='columns', level='name').last()
    data_diff = var_cum_avg.sub(var_last_avg, axis='columns', level='name')

    data_diff_norm = abs(data_diff.div(
        var_last_avg, axis='columns', level='name'))

    return data_diff_norm

def time_average(data_df):
    return data_df.groupby(axis='columns', level='name').mean()


class Probes(utils.Helper):
    def __init__(self, directory):
        """
        File info is stored in a tuple-indexed dictionary. Once data is access, it is read in as a (nested) tuple-indexed dictionary.
        Data is indexed as self.data[(name,step)][(stack, var)]. This format mimics the multiIndex DataFrame created in
        self.slice_int_df.
        """
        my_dict = {}  # this will be a tuple indexed 1-level dictionary.
        # create a generator to iterate over probe paths
        path_generator = glob.iglob(f'{directory}/*.pcd')
        probe_names = []
        probe_steps = []

        for path in path_generator:

            file_name = path.split('/')[-1]  # get the local file name
            probe_info = file_name.split('.')
            probe_name, probe_step, _ = probe_info[:]
            probe_step = int(probe_step)

            probe_names.append(probe_name)
            probe_steps.append(probe_step)

            # store the pcd path and pcd reader function
            my_dict[(probe_name, probe_step)] = path

        # iterate through the upper data dict
        # my_dict = utils.MyLazyDict(my_dict)  # modify the getter lazily read in data

        self.probe_names = [*set(probe_names)]  # remove duplicates
        # remove duplicates and sort
        self.probe_steps = [*set(probe_steps)]

        # assume the quants and stack is the same for all probes
        representative_dict = read_probes(my_dict[(
            self.probe_names[0], self.probe_steps[0])])
        representative_dict_keys = list(
            zip(*representative_dict.keys()))  # unzip list of tuples
        # sort and remove duplicates
        self.probe_quants = [*set(representative_dict_keys[1])]
        # sort and remove duplicates
        self.probe_stack = [*set(representative_dict_keys[0])]
        self.data = my_dict

    def get_locations(self, dir_locations):
        locations = {}
        for probe_name in self.probe_names:
            location_path = f"{dir_locations}/{probe_name}.txt"
            # preparing for lazy location reading
            locations[probe_name] = (read_locations, location_path)
        # creating lazy dict for locations
        self.locations = utils.MyLazyDict(locations)

    def slice_into_df(
        self,
        names = "self.probe_names",
        steps = "self.probe_steps",
        parallel = False
    ):

        
        # default to all probes, setps
        names, steps = [self.get_input(input) for input in [names, steps]]

        # turn outer dict into series for vectorzed opperations
        mi_series = pd.Series(self.data)
        # sort for improved speed
        mi_series.sort_index(inplace=True)
        # get desired values, temporarily converting from multiindex to dataframe for .loc speed increase
        st = utils.start_timer()
        df_from_mi_series = mi_series.unstack()
        df_sliced = df_from_mi_series.loc[names, steps]
        mi_series_sliced = df_sliced.stack()
        utils.end_timer(st, "slicing")

        st = utils.start_timer()

        # dont use parrall for debugging, else significant speed up
        if parallel:
            # initialize(36) or initialize(os.cpu_count()-1)
            spark = SparkSession.builder.getOrCreate()
            # read in data directly (not indecing self.data)
            mi_spdf = spark.createDataFrame(mi_series_sliced)
            mi_spdf = mi_spdf.apply(read_probes)
            mi_df = mi_df.toPandas()
        else:
            mi_df = mi_series_sliced.apply(parallel_functions)

        mi_df = mi_df.T

        mi_df.index.rename(['stack', 'var'], inplace=True)
        if isinstance(mi_df, pd.DataFrame):
            mi_df.columns.rename(['name', 'step'], inplace=True)

        utils.end_timer(st, "reading data")

        st = utils.start_timer()

        # memorize data that was accesed outside of self.data
        self.data.update(mi_df)  # update data dictionary

        utils.end_timer(st, "memorizing data")

        return mi_df  # return numpy array with all requested data

    def contour_plots(
        self,
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "self.probe_stack",
        parrallel = False,
        processing = None,
        plot_params={}
    ):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]

        data = self.slice_into_df(names, steps, parrallel)
        data = data.loc[(stack,quants),:]
        n_names = len(names)
        n_quants = len(quants)

        processed_data = data
        if processing is not None:
            st = utils.start_timer()
            for process_step in processing:
                processed_data = process_step(processed_data)
            utils.end_timer(st, 'processing data')

        st = utils.start_timer()

        # plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(n_names, n_quants, constrained_layout =True)

        for j, (var, var_df) in enumerate(processed_data.groupby(axis='index', level='var')):
            if 'plot_levels' in plot_params and var in plot_params['plot_levels']:
                plot_levels = plot_params['plot_levels'][var]
            else:
                plot_levels = 256

            ax_list = []
            im_list = []
            vmins = [] # for colorbar
            vmaxs = []
            for i, (name, name_df) in enumerate(var_df.groupby(axis='columns', level='name')):
                plot_df = name_df.droplevel('var', axis='index')
                plot_df = plot_df.droplevel('name', axis='columns')
                sub_ax = utils.ax_index(ax, i, j)
                
                xPlot = plot_df.columns
                if 'horizontal spacing' in plot_params:
                    xPlot *= plot_params['horizontal spacing']
                yPlot = plot_df.index
                if hasattr(self, 'locations') and 'stack span' in plot_params:
                    location = self.locations[name]
                    yAxis = location[plot_params['stack span']]
                    yPlot = yAxis[yPlot]

                if 'plot_every' in plot_params:  # usefull to plot subset of timesteps but run calcs across all timesteps
                    name_df = name_df.iloc[:,::plot_params['plot_every']]
                    xPlot =xPlot[::plot_params['plot_every']]

                im = sub_ax.contourf(xPlot, yPlot, name_df, levels=plot_levels)
                ax_list.append(sub_ax)
                im_list.append(im)

                if j > 0:
                    sub_ax.yaxis.set_visible(False)
                else:
                    sub_ax.set_ylabel(name)
                if i < n_names-1:
                    sub_ax.xaxis.set_visible(False)
                else:
                    sub_ax.set_xlabel(var)

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
        stack = "self.probe_stack",
        parrallel = False,
        plot_params = {}
        ):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]
        data = self.slice_into_df(names, steps, parrallel)
        data = data.loc[(stack,quants),:]

        fig, ax = plt.subplots(1, 1, constrained_layout =True)
        processed_data = time_average(data)
        for j, (var, var_df) in enumerate(processed_data.groupby(axis='index', level='var')):
            for i, (name, name_df) in enumerate(var_df.groupby(axis='columns', level='name')):
                plot_df = name_df.droplevel('var', axis='index')
                # if isinstance(plot_df, pd.DataFrame):
                #     plot_df = plot_df.droplevel('name', axis='columns')

                yPlot = plot_df.index
                if hasattr(self, 'locations') and 'stack span' in plot_params:
                    location = self.locations[name]
                    yAxis = location[plot_params['stack span']]
                    yPlot = yAxis[yPlot]

                ax.plot(plot_df.values, yPlot, label=f'{name}: {var}')
                if 'xlabel' in plot_params:
                    fig.supxlabel(plot_params['xlabel'])
                if 'ylabel' in plot_params:
                    fig.supylabel(plot_params['ylabel'])
                ax.legend()


