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

if __name__=="__main__":
    tile = y_col(-5.3, 0, [0, 5.36], 536)
    writeProbes(tile, 'x_water.txt')