# Python function that takes as input tile (tile.shape = (nProbes,3)) whose rows are the x,y,z
# coordinates of the probes

import numpy as np

class Probes:
    def __init__(self, tile = None, name = None, fileName = None, type = None):
        self.tile = tile
        self.name = name
        self.fileName = fileName
        self.type = type
        self.probeCall = None

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
        self.type = "PROBE"
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

    def getProbeCall(self, minVolThick = 0, vars = ""):
    
        probeCall = f"{self.type} NAME=$(probe_path)/{self.name:22} INTERVAL $(probe_int) "

        if self.type == "PROBE":
            probeCall += f"GEOM FILE $(location_path)/{self.name + '.txt' :26}" 
            probeCall += f"VARS {vars}"
        
        elif self.type == "VOLUMETRIC_PROBE":
            mins = np.min(self.tile, axis = 0)
            maxs = np.max(self.tile, axis = 0)
            for i in range(3):
              if (maxs[i] - mins[i]) < minVolThick:
                offset = (minVolThick - (maxs[i] - mins[i]))/2
                mins[i] -= offset
                maxs[i] += offset
            probeCall += f"GEOM BOX {mins[0]:f} {maxs[0]:f}  {mins[1]:f} {maxs[1]:f}  {mins[2]:f} {maxs[2]:f} " 
            probeCall += f"VARS {vars}"
        
        elif self.type == "FLUX_PROBE":
            # mins = np.min(self.tile, axis = 0)
            # maxs = np.max(self.tile, axis = 0)
            mins = self.tile[0, :]
            maxs = self.tile[-1, :]
            means = np.mean(self.tile, axis = 0)
            normalVec = maxs - mins
            probeCall += f"XP {means[0]:f} {means[1]:f} {means[2]:f} "
            probeCall += f"NP {normalVec[0]:f} {normalVec[1]:f} {normalVec[2]:f} "
            probeCall += f"VARS mass_flux({vars}) "
            probeCall += f"sn_prod(comp(u,{np.argmax(normalVec)}),{normalVec[0]:f},{normalVec[1]:f},{normalVec[2]:f}) "

        self.probeCall = probeCall
