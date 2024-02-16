# Function that generates the profile specification files for a turbulent inflow

# Example usage:
# python3 pyCascade/pyCascade/inflowProfile.py --n 400 --x -40 --z 0 --y0 0 --y1 30 --rough 0.000005 --UatZ 15 1.07 --Iu 0.1087 --method 'ASCE' --multiply 4 --filefmt 'rows' --filename 'inflowProfile_30m_x4.dat'
# python3 pyCascade/pyCascade/inflowProfile.py --n 400 --x -40 --z 0 --y0 0 --y1 30 --rough 0.000005 --UatZ 15 1.07 --Iu 0.1087 --method 'ASCE' --multiply 4 --filefmt 'table' --filename 'turbInflowProfile_30m_x4.dat'

import numpy as np
import argparse
import matplotlib.pyplot as plt
import pylustrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, help='Number of points (defaults to 100 if none given)')
    parser.add_argument('--x', type=float, default=0, help='x location of inlet')
    parser.add_argument('--z', type=float, default=0, help='z location of inlet')
    parser.add_argument('--y0', type=float, default=0, help='Min height of turb inlet (defaults to 0 if none given)')
    parser.add_argument('--y1', type=float, required=True, help='Max height of turb inlet')
    parser.add_argument('--rough', type=float, required=True, help='Roughness length of log law')
    parser.add_argument('--Iu', type=float, help='Turbulence intensity')
    parser.add_argument('--ustar', type=float, help='Friction velocity of log law')
    parser.add_argument('--UatZ', type=float, nargs=2, help='Velocity at altitude, e.g. for 10 m/s at 100 m: --UatZ 10 100')
    parser.add_argument('--method', type=str, default='ASCE', help='"ASCE" for the method prescribed in ASCE7, "Stull" for the method from the Stull textbook. Defaults to ASCE')
    parser.add_argument('--filename', type=str, default='inflowProfile.dat', help='Name of filename to write to')
    parser.add_argument('--filefmt', type=str, default='rows', help='For the inflowProfile.dat file, use "rows" to have one value per row, "table" to have multiple columns')
    parser.add_argument('--plot', help='Include to plot profiles', action='store_true')
    parser.add_argument('--bldgheight', type=float, default=np.nan, help='If plotting, specify the building height to normalize y-axis')
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
    uu = np.power(np.divide(U, np.log(y/args.rough)), 2)
    vw = np.zeros((N, ))
    uw = np.zeros((N, ))
    
    if args.multiplyuu:
        uu = uu_1x * args.multiply
    else:
        uu = uu_1x

    if args.method == 'Stull':
        vv = uu_1x / np.sqrt(2) * args.multiply
        ww = vv * args.multiply
        uv = -vv * args.multiply
    else:
        vv = 0.25 * uu 
        ww = 0.64 * uu
        uv = -(ustar ** 2) * np.ones((N, ))
        # Check the realizability constraint:
        realizability_cond = np.sqrt(np.multiply(uu_1x, vv))
        uv = - np.minimum(np.abs(uv), realizability_cond)

    # Write to Fluent style file:
    with open(args.filename, 'w+') as f: # overwrite
        if args.filefmt == 'rows':
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
        else:
            f.write('x\ty\tz\tx-velocity\ty-velocity\tz-velocity\tuu-reynolds-stress\tvv-reynolds-stress\tww-reynolds-stress\tvw-reynolds-stress\tuw-reynolds-stress\tuv-reynolds-stress\n')
            for i in range(x.shape[0]):
                f.write('%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n' %(x[i], y[i], z[i], U[i], V[i], W[i], uu[i], vv[i], ww[i], vw[i], uw[i], uv[i]))
        f.close()

    # Write individual files:
    # write_file('UInlet', y, U)
    # write_file('uuBarInlet', y, uu)
    # write_file('vvBarInlet', y, vv)
    # write_file('wwBarInlet', y, ww)
    # write_file('uvBarInlet', y, uv)

    print('Files written')

    if args.plot:
        pylustrator.start()
        plt.rcParams['figure.figsize'] = [5, 3]
        f, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        
        if args.bldgheight == np.nan:
            H = 1
            ax[0].set_ylabel('y [m]')
        else:
            H = args.bldgheight
            ax[0].set_ylabel('y / H')
        
        ax[0].plot(U, y / H, 'k-', linewidth=1)
        ax[0].set_xlabel('U [m/s]')
        ax[1].plot(uu, y / H, 'k-', linewidth=1, label='uu')
        ax[1].plot(vv, y / H, 'b-', linewidth=1, label='vv')
        ax[1].plot(ww, y / H, 'r-', linewidth=1, label='ww')
        ax[1].plot(uv, y / H, 'g-', linewidth=1, label='uv')
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
    f.close()


if __name__ == "__main__":
    main()