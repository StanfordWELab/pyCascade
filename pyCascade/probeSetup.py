# Python function that takes as input tile (tile.shape = (nProbes,3)) whose rows are the x,y,z
# coordinates of the probes

import numpy as np

class Probes:
    def __init__(self, tile = None, name = None, fileName = None, type = None):
        self.tile = tile
        self.name = name
        self.fileName = fileName
        self.type = type

    def writeProbes(self):
    
        nPoints, temp = np.shape(self.tile)
    
        with open(self.fileName,'w+') as out:
            out.write(str(nPoints))
            out.write(' points\n')
            for i in range(nPoints):
                out.write('    ' + '{:06.6f}'.format(self.tile[i,0]) + '    ' + '{:06.6f}'.format(self.tile[i,1]) + '    ' + '{:06.6f}'.format(self.tile[i,2]) + '\n')

        return

    def y_col(self, x, z, y_range, n_probes):
        y_min, y_max = y_range[:]
        tile = np.zeros((n_probes, 3))
        tile[:,0] = x
        tile[:,2] = z
        tile[:,1] = np.linspace(y_min, y_max, n_probes)
        self.tile = tile

    def probe_fill(self, x_range, y_range, z_range):
        n_x = np.size(x_range)
        n_y = np.size(y_range)
        n_z = np.size(z_range)
        n_probes = n_x*n_y*n_z
        n_probes = np.size(x_range)*np.size(z_range)*np.size(y_range)
        tile = np.zeros((n_probes, 3))
        tile[:,0] = np.tile(x_range, n_y*n_z)
        tile[:,1] = np.tile(np.repeat(y_range, n_x), n_z)
        tile[:,2] = np.repeat(z_range, n_x*n_y)
        self.tile = tile