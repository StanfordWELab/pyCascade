from pyCascade import utils, physics
import dask
import numpy as np
import statsmodels.api as sm
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

@dask.delayed
def LengthScale(uPrime, meanU, time, show_plot=False):
    """
    Compute the length scale of the flow using the exponential fit method
    """
    func = lambda x, a: np.exp(-x/a) #define theoretical exponential decay function
    time -= time[0] # shift time to start at zero
    
    R_u = sm.tsa.stattools.acf(uPrime, nlags = len(time)-1, fft = True) # compute autocorrelation function
    R_uFit, _ = sp.optimize.curve_fit(func, time, R_u, p0=1, bounds = (0,np.inf)) # fit the exponential decay function to the autocorrelation function
    t_scale = R_uFit[0] # extract the length scale from the fit
    Lx = t_scale*meanU # compute the length scale of the flow

    if show_plot == True:
        plt.figure()
        plt.plot(time, R_u, label = 'Autocorrelation')
        plt.plot(time, func(time, *R_uFit), label = 'Exponential fit')
        plt.xlabel('Time')
        plt.ylabel('Autocorrelation')
        plt.legend()

    return Lx

class Qty(utils.Helper):
    def __init__(self):
        self.fsamp = None

        self.u = None
        self.v = None
        self.w = None
        self.p = None

        self.meanU = None
        self.meanV = None
        self.meanW = None
        self.meanP = None

        self.uPrime = None
        self.vPrime = None
        self.wPrime = None
        self.pPrime = None

        self.prms = None

        self.uu = None
        self.vv = None
        self.ww = None

        self.uv = None
        self.vw = None  
        self.uw = None

        self.Iu = None
        self.Iv = None
        self.Iw = None

        self.uu_avg = None
        self.vv_avg = None
        self.ww_avg = None
        self.pp_avg = None

        self.uv_avg = None
        self.vw_avg = None
        self.uw_avg = None

        self.Iu_avg = None
        self.Iv_avg = None
        self.Iw_avg = None

        self.y = None

    def computeQty(self, data_dict, t_data, u_str = 'comp(u,0)', v_str = 'comp(u,1)', w_str = 'comp(u,2)', p_str = 'p', calc_stats = True):
        self.fsamp = 1/(t_data[-1]-t_data[-2])

        self.u = data_dict[u_str]
        self.v = data_dict[v_str]
        self.w = data_dict[w_str]
        self.p = data_dict[p_str]

        if calc_stats:
            self.meanU = np.mean(self.u, axis = 'index')
            self.meanV = np.mean(self.v, axis = 'index')
            self.meanW = np.mean(self.w, axis = 'index')
            self.meanP = np.mean(self.p, axis = 'index')

            self.uPrime = self.u - self.meanU
            self.vPrime = self.v - self.meanV
            self.wPrime = self.w - self.meanW
            self.pPrime = self.p - self.meanP

            self.prms = np.sqrt(np.mean(self.pPrime**2, axis = 'index'))

            self.uu = self.uPrime**2
            self.vv = self.vPrime**2
            self.ww = self.wPrime**2

            self.uv = self.uPrime*self.vPrime
            self.vw = self.vPrime*self.wPrime
            self.uw = self.uPrime*self.wPrime

            self.Iu = np.sqrt(self.uu) / self.meanU
            self.Iv = np.sqrt(self.vv) / self.meanV
            self.Iw = np.sqrt(self.ww) / self.meanW

            self.uu_avg = np.mean(self.uu, axis = 'index')
            self.vv_avg = np.mean(self.vv, axis = 'index')
            self.ww_avg = np.mean(self.ww, axis = 'index')
            self.pp_avg = np.mean(self.pPrime**2, axis = 'index')

            self.uv_avg = np.mean(self.uv, axis = 'index')
            self.vw_avg = np.mean(self.vw, axis = 'index')
            self.uw_avg = np.mean(self.uw, axis = 'index')

            self.Iu_avg = np.sqrt(self.uu_avg)/self.meanU
            self.Iv_avg = np.sqrt(self.vv_avg)/self.meanU
            self.Iw_avg = np.sqrt(self.ww_avg)/self.meanU

            N, idx = self.p.shape

            Lx = []
            Ly = []
            Lz = []
            for i in range(idx):
                Lx.append(LengthScale(self.uPrime.values[:,i], self.meanU.values[i], t_data))
                Ly.append(LengthScale(self.vPrime.values[:,i], self.meanU.values[i], t_data))
                Lz.append(LengthScale(self.wPrime.values[:,i], self.meanU.values[i], t_data))

            Lx, Ly, Lz = np.array(dask.compute(Lx, Ly, Lz)) #execute the dask graph

            self.Lx = np.array(Lx)
            self.Ly = np.array(Ly)
            self.Lz = np.array(Lz)

            self.f, self.Euu = sp.signal.welch(self.uPrime, fs = self.fsamp, axis = 0, nperseg = N//4, scaling = 'density', detrend = 'constant')

            _, self.Epp = sp.signal.welch(self.uPrime, fs = self.fsamp, axis = 0, nperseg = N//4, scaling = 'density', detrend = 'constant')

    def set_y(self, y):
        self.y = y

def plot_ABL(qty_dict: dict, fit_disp = False):
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # fit log law to find uStar, z0, disp
    kappa = 0.41
    for i, (name, qty) in enumerate(qty_dict.items()):
        y = qty.y #get the height of the probes
        ax.plot(qty.meanU, y, f'{colors[i]}o', label=f'{name}')

        c0, c1 = np.polyfit(qty.meanU, np.log(y), 1) #fit a line to the log of the height
        uStar = kappa/c0 #get uStar from the slope
        z0 = np.exp(c1) #get z0 from the intercept
        disp = 0

        if fit_disp == True:
            popt, _ = curve_fit(physics.loglaw_with_disp, y, qty.meanU, p0=[uStar, z0, disp], bounds=((0,0,0),(np.inf,np.inf,np.inf)), method='dogbox') #fit the log law with displacement
            uStar, z0, disp = popt
            
        y_plot = np.linspace(0, y[-1], 100)
        ax.plot(physics.loglaw_with_disp(y_plot, uStar, z0, disp), y_plot, f'{colors[i]}-')

    ax.set_xlabel('mean velocity [m/s]')
    ax.set_ylabel('height [m]')
    ax.legend()

    return fig, ax

def plot_length_scales(qty_dict: dict):
    fig, ax = plt.subplots(1,3)
    for name, qty in qty_dict.items():
        y = qty.y
        ax[0].plot(qty.Lx, y, '-', label = name)
        ax[1].plot(qty.Ly, y, '-', label = name)
        ax[2].plot(qty.Lz, y, '-', label = name)

    ax[0].set_title('Lx')
    ax[1].set_title('Ly')
    ax[2].set_title('Lz')

    ax[0].set_ylabel('L [m]')

    ax[0].set_xlabel('y [m]')
    ax[1].set_xlabel('y [m]')
    ax[2].set_xlabel('y [m]')

    ax[2].legend()
    # place legend outside of plot
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig, ax

def plot_reynolds_stresses(qty_dict: dict):
    fig, ax = plt.subplots(2,3)
    for name, qty in qty_dict.items():
        y = qty.y
        ax[0,0].plot(qty.uu_avg, y, '-', label = name)
        ax[0,1].plot(qty.vv_avg, y, '-', label = name)
        ax[0,2].plot(qty.ww_avg, y, '-', label = name)
        ax[1,0].plot(qty.uv_avg, y, '-', label = name)
        ax[1,1].plot(qty.uw_avg, y, '-', label = name)
        ax[1,2].plot(qty.vw_avg, y, '-', label = name)

    ax[0,0].set_ylabel('y [m]')
    ax[1,0].set_ylabel('y [m]')
    ax[1,1].set_xlabel('Reynolds Stress [m^2/s^2]')

    ax[0,0].set_title('uu')
    ax[0,1].set_title('vv')
    ax[0,2].set_title('ww')
    ax[1,0].set_title('uv')
    ax[1,1].set_title('uw')
    ax[1,2].set_title('vw')

    ax[1,2].legend()

    plt.tight_layout()
    #place legend outside of plot
    box = ax[1,2].get_position()
    ax[1,2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig, ax

def plot_turbulence_intensities(qty_dict: dict):
    fig, ax = plt.subplots(1,3)

    for name, qty in qty_dict.items():
        y = qty.y
        ax[0].plot(qty.Iu_avg, y, '-', label = name)
        ax[1].plot(qty.Iv_avg, y, '-', label = name)
        ax[2].plot(qty.Iw_avg, y, '-', label = name)

    ax[0].set_ylabel('y [m]')

    ax[0].set_xlabel('Iu')
    ax[1].set_xlabel('Iv')
    ax[2].set_xlabel('Iw')
    return fig, ax

def plot_prms(qty_dict: dict):
    fig, ax = plt.subplots(1,1)
    for name, qty in qty_dict.items():
        y = qty.y
        ax.plot(qty.prms, y, '-', label = name)

    ax.set_ylabel('y [m]')
    ax.set_xlabel('Prms [Pa]')
    ax.legend()
    return fig, ax

def plot_power_spectra(qty_dict: dict, interial_offset = 10**(-1)):
    fig, ax = plt.subplots()
    for i, (name, qty) in enumerate(qty_dict.items()):
        y = qty.y
        for j, yval in enumerate(y):
            ax.loglog(qty.f, qty.Euu[:,j], '-', label = f'y = {name}, y={yval:.0f} [m]')
    ax.loglog(qty.f, interial_offset*qty.f**(-5/3), label = '-5/3')
    ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("frequency $[1/s]$")
    ax.set_ylabel("$E_u [m^3/s^2]$")
    plt.tight_layout()
    return fig, ax