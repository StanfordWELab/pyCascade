# Function that generates the profile specification files for a turbulent inflow
# ARGUMENTS:
# --n        Number of points (defaults to 100 if none given)
# --x        x location of inlet (defaults to 0 if none given)
# --z        z location of inlet (defaults to 0 if none given)
# --y0       Min height of turb inlet (defaults to 0 if none given)
# --y1       Max height of turb inlet
# --rough    Roughness length of log law
# --ustar    Friction velocity of log law
# --UatZ     Instead of friction velocity, specify U at some altitude
# --method   'ASCE' for the method prescribed in ASCE7, 'Stull' for the method from the Stull textbook. Defaults to ASCE
# --multiply Factor by which to multiply the stresses
# --filename Where to write the .dat file (other files must have standard name, so not an option)
# --plot     Flag to plot resulting profiles

import numpy as np
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, help='Number of points (defaults to 100 if none given)')
    parser.add_argument('--x', type=float, default=0, help='x location of inlet')
    parser.add_argument('--z', type=float, default=0, help='z location of inlet')
    parser.add_argument('--y0', type=float, default=0, help='Min height of turb inlet (defaults to 0 if none given)')
    parser.add_argument('--y1', type=float, required=True, help='Max height of turb inlet')
    parser.add_argument('--rough', type=float, required=True, help='Roughness length of log law')
    parser.add_argument('--ustar', type=float, help='Friction velocity of log law')
    parser.add_argument('--UatZ', type=float, nargs=2, help='Velocity at altitude, e.g. for 10 m/s at 100 m: --UatZ 10 100')
    parser.add_argument('--method', type=str, default='ASCE', help='"ASCE" for the method prescribed in ASCE7, "Stull" for the method from the Stull textbook. Defaults to ASCE')
    parser.add_argument('--multiply', type=float, default=1.0, help='Factor by which to multiply Reynolds stresses')
    parser.add_argument('--filename', type=str, default='inflowProfile.dat', help='Name of filename to write to')
    parser.add_argument('--plot', help='Include to plot profiles', action='store_true')
    args = parser.parse_args()

    if args.ustar is None and args.UatZ is None:
        raise Exception('Must specify either U at some height (--UatZ) or friction velocity (--ustar)')

    N = args.n

    # Coordinates:
    x = args.x * np.ones((N, ))
    y = np.linspace(args.y0, args.y1, N)
    z = args.z * np.ones((N, ))

    # Velocities:
    if args.ustar is None:
        ustar = 0.41 * np.divide(args.UatZ[0], np.log(args.UatZ[1] / args.rough)) # calculate ustar
    else:
        ustar = args.ustar
    U = ustar / 0.41 * np.log(np.divide((y - args.y0), args.rough)) # subtract since inlet floor starts at y0
    U[0] = 0 # force 0 at the wall to get rid of the inf
    V = np.zeros((N, ))
    W = np.zeros((N, ))

    # Turbulence intensities:
    uu = np.power(np.divide(U, np.log(y/args.rough)), 2) * args.multiply
    vw = np.zeros((N, ))
    uw = np.zeros((N, ))
    
    if args.method == 'Stull':
        vv = uu / np.sqrt(2)
        ww = vv
        uv = -vv
    else:
        vv = 0.25 * uu
        ww = 0.64 * uu 
        uv = -(ustar ** 2) * np.ones((N, )) * args.multiply
        # Check the realizability constraint:
        realizability_cond = np.sqrt(np.multiply(uu, vv))
        uv = - np.minimum(np.abs(uv), realizability_cond)

    # Write to Fluent style file:
    with open(args.filename, 'w+') as f: # overwrite
        f.write('((points point %d)' %N)
        write_array(f, x, 'x')
        write_array(f, y, 'y')
        write_array(f, z, 'z')
        write_array(f, U, 'x-velocity')
        write_array(f, V, 'y-velocity')
        write_array(f, W, 'z-velocity')
        write_array(f, uu, 'uu-reynolds-stress')
        write_array(f, vv, 'vv-reynolds-stress')
        write_array(f, ww, 'ww-reynolds-stress')
        write_array(f, vw, 'vw-reynolds-stress')
        write_array(f, uw, 'uw-reynolds-stress')
        write_array(f, uv, 'uv-reynolds-stress')
        f.write(')')

    write_file('UInlet', y, U)
    write_file('uuBarInlet', y, uu)
    write_file('vvBarInlet', y, vv)
    write_file('wwBarInlet', y, ww)
    write_file('uvBarInlet', y, uv)

    print('Files written')

    if args.plot:
        f, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        ax[0].plot(U, y, 'k-', linewidth=1)
        ax[0].set_ylabel('y (height) [m]')
        ax[0].set_xlabel('U [m/s]')
        ax[1].plot(uu, y, 'k-', linewidth=1, label='uu')
        ax[1].plot(vv, y, 'b-', linewidth=1, label='vv')
        ax[1].plot(ww, y, 'r-', linewidth=1, label='ww')
        ax[1].plot(uv, y, 'g-', linewidth=1, label='uv')
        ax[1].set_xlabel(r'$\mathrm{[m^2/s^2]}$')
        ax[1].legend()
        plt.show()


def write_array(file, arr, arr_name):
    file.write('\n(%s\n' %arr_name)
    for i in range(arr.shape[0]):
        file.write('%.8f\n' %arr[i])
    file.write(')')

def write_file(filename, index_arr, arr):
    # Write two columns (index_arr, arr) to the file specified by filename
    with open(filename, 'w+') as f:
        for i in range(index_arr.shape[0]):
            f.write('%.8f %.8f\n' %(index_arr[i], arr[i]))


if __name__ == "__main__":
    main()