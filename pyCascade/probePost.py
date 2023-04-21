from pyCascade import utils, quantities

import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from dask import dataframe as dd
from pandarallel import pandarallel
import scipy as sp

def read_pointcloud_probes(filename):
    return dd.read_csv(filename, delim_whitespace=True)  # read as dataframe

def read_probes(filename):
    ddf = dd.read_csv(filename, delimiter = ' ', comment = "#",header = None)
    step_index = ddf.iloc[:, 0] #grab the first column for the indixes
    time_index = ddf.iloc[:, 1] #grab the second column for the times
    ddf = ddf.iloc[:, 3:] #take the data less the index rows

    _, n_cols = ddf.shape
    ddf = ddf.rename(columns=dict(zip(ddf.columns, np.arange(0, n_cols)))) #reset columns to integer 0 indexed
    ddf.columns.name = 'Stack'
    ddf.index = step_index
    return ddf, step_index, time_index

def read_locations(filename):
    return pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['x', 'y', 'z'])


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

    return utils.dict_apply(df_func)(data_dict, t_data = None)

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


def ClenshawCurtis_Quadrature(data_dict, t_data = None):
    N = 10
    interval = 2.5
    xs = [np.cos((2*(N-k)-1)*np.pi/(2*N)) for k in range(N)]
    A = np.array([[xs[i]**j for i in range(N)] for j in range(N)])
    b = [(1+(-1)**k)/(k+1) for k in range(N)]
    ws = np.linalg.solve(A,b)
    CC_weights = np.squeeze(ws[:,None])

    # ## correcting for inforrect probes locations untill R10
    # Warning('Quadrature weights shifted to match R<=10. Delete this in the future')
    # CC_weights = np.roll(CC_weights, 1)
    # #############

    Quad_weights = np.tile(CC_weights, 10) * np.repeat(CC_weights, 10) * (interval/2)**2

    def df_func(data_df):
        wighted_data = data_df*Quad_weights
        integrated_data = wighted_data.sum(axis=1)
        return integrated_data

    return utils.dict_apply(df_func)(data_dict)

# use to define lambda function with mul preset
def mul_names(data_dict, names, mul):
    for k, v in data_dict.items():
        name, _ = k
        if name in names:
            data_dict[k] = mul*v
    return data_dict

def power_spectrum(data_dict, t_data):
        fsamp = t_data[-1]-t_data[-2]
        
        def df_func(data_df):
            data = data_df.values
            N, _ = data.shape

            ######Compute power spectral density without welch#####
            # data_prime = data - data.mean(axis=0)
            # data_fft = sp.fft.fft(data_prime, axis = 0) # take fft  
            # data_fft = data_fft[:N//2+1, :] # one sided fft
            # S = 1/(N*fsamp) * np.abs(data_fft)**2 # Compute power spectral density

            # # Positive and negative frequencies appear twice 
            # # (except for zero frequency and Nyquist frequency)
            # S[1:-1] = 2*S[1:-1]

            # ## Apply a uniform filter to smooth the spectrum
            # filter_size = int(4)
            # S = sp.ndimage.filters.uniform_filter1d(S,size = filter_size, axis = 0)
            # f = np.arange(0,fsamp/2,fsamp/(N+1))
            ###################################################

            f, S = sp.signal.welch(data, fs = fsamp, axis = 0, nperseg = N//4, scaling = 'density', detrend = 'constant')#, noverlap = N//4)#, nfft = N)#, return_onesided = True)#, scaling = 'density')
            
            S_df = pd.DataFrame(S, index=f, columns=data_df.columns)
            return S_df
        return utils.dict_apply(df_func)(data_dict)


class Probes(utils.Helper):
    def __init__(self, directory, probe_type = "PROBES"):
        """
        File info is stored in a tuple-indexed dictionary. Once data is access, it is read in as a (nested) tuple-indexed dictionary.
        For POINTCLOUD_PROBES data is indexed as self.data[(name,step)][(stack, quant)]. For PROBES data is indexed as 
        self.data[(name, quant)][(stack, step)]. This format mimics the multiIndex DataFrame created in self.slice_into_df.
        """

        self.probe_type = probe_type

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
                my_dict[(probe_name, probe_tbd1)], probe_tbd2s, probe_time = read_probes(path)


            probe_names.append(probe_name)
            probe_tbd1s.append(probe_tbd1)
        
        probe_tbd2s = probe_tbd2s.compute().values
        n_indexes = len(probe_tbd2s)
        probe_tbd2s, unique_steps_indexes = np.unique(np.flip(probe_tbd2s), return_index = True)
        unique_steps_indexes = n_indexes-1-unique_steps_indexes
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

    def get_locations(self, dir_locations):
        locations = {}
        for probe_name in self.probe_names:
            location_path = f"{dir_locations}/{probe_name}.txt"
            # preparing for lazy location reading
            locations[probe_name] = (read_locations, location_path)
        # creating lazy dict for locations
        self.locations = utils.MyLazyDict(locations)

    def process_data(
        self, 
        names = "self.probe_names",
        steps = "self.probe_steps",
        quants = "self.probe_quants",
        stack = "np.s_[::]",
        processing = None):

        quants, stack, names, steps = [self.get_input(input) for input in [quants, stack, names, steps]]
        t_data = self.probe_times[steps]
        st = utils.start_timer()
        processed_data  = {}
        for name in names:
            for quant in quants:
                ddf = self.data[(name, quant)]
                processed_data[(name, quant)] = ddf[stack]#.loc[steps[0]:steps[-1]]

        processed_data = utils.dict_apply(ddf_to_pdf)(processed_data)

        def index_unique_steps(df):
            if isinstance(df, (pd.core.frame.DataFrame, pd.core.series.Series)):
                df = df.iloc[self.unique_steps_indexes].loc[steps]
            return df
    
        processed_data = utils.dict_apply(index_unique_steps)(processed_data)

        if processing is not None:
            for process_step in processing:
                processed_data = process_step(processed_data, t_data)
        
        utils.end_timer(st, 'processing data')
        return processed_data
    
    def computeQty(
            self, 
            names = "self.probe_names", 
            steps = "self.probe_steps", 
            quants = "self.probe_quants", 
            stack = "np.s_[::]", 
            processing = None):
        
        qty_dict = {}
        for name in names:
            qty = quantities.Qty()
            processed_data = self.process_data([name], steps, quants, stack, processing)
            qty.computeQty(processed_data,
                           self.probe_times[steps],
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
                        xPlot = plot_params['horizontal spacing'][xPlot]
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
                    xPlot *= plot_params['horizontal spacing']

                if 'plot_every' in plot_params:  # usefull to plot subset of timesteps but run calcs across all timesteps
                    name_df = plot_df.iloc[:,::plot_params['plot_every']]
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
    
    def frequency_plots(
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
                    xPlot *= plot_params['horizontal spacing']

                if 'plot_every' in plot_params:  # usefull to plot subset of timesteps but run calcs across all timesteps
                    name_df = plot_df.iloc[:,::plot_params['plot_every']]
                    xPlot =xPlot[::plot_params['plot_every']]

                yPlot =  np.squeeze(plot_df.values)
                if 'veritcal scaling' in plot_params:
                   yPlot*=plot_params['veritcal scaling']

                ax.loglog(xPlot, yPlot, label=f'{name}: {quant}')
                ax.loglog(xPlot, 0.001*xPlot**(-5/3))
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
        return df_data


