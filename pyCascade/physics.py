import numpy as np
from pyCascade import utils
from matplotlib import pyplot as plt


class LES_Physics(utils.Helper):
    def __init__(self, LES_params = {}):
        # LES params
        self.LES_params = {}
        self.LES_params.update(LES_params)

    def plot_log_wind(
        self, 
        uStar = "self.LES_params['uStar']", 
        z0 = "self.LES_params['z0']", 
        disp = "self.LES_params['disp']", 
        vK_const = "self.LES_params['vK_const']",
        z_values = "self.LES_params['z_values']",
        ):

        uStar, z0, disp, vK_const, z_values = [self.get_input(input) for input in [uStar, z0, disp, vK_const, z_values]]

        z_scaled = (z_values - disp)/z0
        z_scaled[z_scaled<1] = 1

        log_wind = (uStar/vK_const)*np.log(z_scaled)
        plt.plot(log_wind, z_values)
        plt.xlabel('velocity')
        plt.ylabel('height [m]')

        for key in ['uStar', 'z0', 'disp', 'vK_const', 'z_values', 'log_wind']:
            self.LES_params[key] = eval(key) #save params

    def calc_flatplate_quantities(
        self,
        rho = "self.LES_params['rho']",
        uInf = "self.LES_params['log_wind'][-1]",
        L = "self.LES_params['z_values'][-1]",
        mu = "self.LES_params['mu']",
        yPlus = "self.LES_params['flatplate']['yPlus']"
        ):

        '''
        from https://www.cadence.com/en_US/home/tools/system-analysis/computational-fluid-dynamics/y-plus.html
        '''

        if 'flatplate' not in self.LES_params:
            self.LES_params['flatplate'] = {}

        rho, uInf, L, mu, yPlus = [self.get_input(input) for input in [rho, uInf, L, mu, yPlus]]

        Rex = rho*uInf*L/mu
        Cf = .026/Rex**(1/7)
        tauW = (Cf*rho*uInf**2)/2
        uFric = np.sqrt(tauW/rho)
        spacingW = yPlus*mu/(uFric*rho)

        print(f'U infinity is {uInf} with L = {L}')
        print(f'flat plate Reynolds number is {Rex}')
        print(f'wall spacing is {spacingW} for y+ of {yPlus}')

        for key in ['rho', 'mu']:
            self.LES_params[key] = eval(key) #save params
        
        for key in ['yPlus', 'Rex', 'Cf', 'tauW', 'uFric', 'spacingW']:
            self.LES_params['flatplate'][key] = eval(key) #save params

    def calc_norm_pressure_source(
        self,
        factor,
        uStar = "self.LES_params['uStar']",
        Lz = "self.LES_params['z_values'][-1] - self.LES_params['z_values'][0]",

        ):

        uStar, Lz = [self.get_input(input) for input in [uStar, Lz]]

        print(f"uStar is {uStar} and Lz is {Lz}")

        norm_pressure_source = factor*(uStar**2)/Lz

        print(f"the normalized pressure source is {norm_pressure_source}")

        key = 'norm_pressure_source'
        self.LES_params[key] = eval(key)

    def plot_spinup_velocity(
        self, 
        log_wind = "self.LES_params['log_wind']", 
        z_values = "self.LES_params['z_values']"
        ):

        log_wind, z_values = [self.get_input(input) for input in [log_wind, z_values]]
        
        domain_height = np.max(z_values)
        u_bulk = np.trapz(log_wind, z_values)/domain_height
        print(f'u_bulk is {u_bulk}')
        print(f'domain height is {domain_height}')

        spinup_profile = u_bulk*2*(1-z_values/domain_height)

        plt.plot(spinup_profile, z_values)
        plt.xlabel('velocity')
        plt.ylabel('height [m]')

        for key in ['log_wind', 'z_values', 'u_bulk']:
            self.LES_params[key] = eval(key) #save params

        

