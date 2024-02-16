# Python function that takes as input tile (tile.shape = (nProbes,3)) whose rows are the x,y,z
# coordinates of the probes

import numpy as np

def writeProbes(tile, fileName):
   
    nPoints, temp = np.shape(tile)
   
    with open(fileName,'w+') as out:
        out.write(str(nPoints))
        out.write(' points\n')
        for i in range(nPoints):
            out.write('    ' + '{:06.6f}'.format(tile[i,0]) + '    ' + '{:06.6f}'.format(tile[i,1]) + '    ' + '{:06.6f}'.format(tile[i,2]) + '\n')
       
    return

def y_col(x, z, y_range, n_probes):
    y_min, y_max = y_range[:]
    tile = np.zeros((n_probes, 3))
    tile[:,0] = x
    tile[:,2] = z
    tile[:,1] = np.linspace(y_min, y_max, n_probes)
    return tile

def x_line(y, z, x_range, n_probes):
    x_min, x_max = x_range[:]
    tile = np.zeros((n_probes, 3))
    tile[:,1] = y
    tile[:,2] = z
    tile[:,0] = np.linspace(x_min, x_max, n_probes)
    return tile

if __name__=="__main__":
    # SN:
    # tile = y_col(-1060, 0, [0, 600], 100) # -5.3 for water
    # writeProbes(tile, 'profile-water_600m.txt')
    # tile = y_col(-200, 0, [0, 600], 100)
    # writeProbes(tile, 'profile-upstream_600m.txt')
    # tile = x_line(1.07, 0, [-23.196, 32.594], 400)
    # writeProbes(tile, 'streamwise_SN_height_maxdomain.txt')

    # 650Cal:
    tile = y_col(-13, 0, [0.5, 6], 400)
    writeProbes(tile, 'profile-13m-upstream_6m.txt')
    # tile = y_col(-9.2, 0, [0.4, 3], 100)
    # writeProbes(tile, 'profile-9_2m-upstream.txt')
    # tile = y_col(-5, 0, [0.6, 3], 100)
    # writeProbes(tile, 'profile-5m-upstream.txt')
    tile = y_col(-1, 0, [0.3, 6], 400)
    writeProbes(tile, 'profile-1m-upstream_6m.txt')