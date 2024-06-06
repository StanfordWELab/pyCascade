# Python function that takes as input tile (tile.shape = (nProbes,3)) whose rows are the x,y,z
# coordinates of the probes

import numpy as np
import os

class Probes:
    def __init__(self, tile = None, name = None, fileName = None, type = None):
        self.tile = tile
        self.name = name
        self.fileName = fileName
        self.type = type
        self.probeCall = None

    def writeProbes(self, append = False):
        if append:
            writeStyle = "a+"
        else:
            writeStyle = "w+"
        with open(self.fileName, writeStyle) as out:
            nPoints, _ = np.shape(self.tile)
            if os.path.exists(self.fileName) == False or writeStyle == "w+":
                out.write(str(nPoints))
                out.write(' points\n')
            else:
                print(f"Appending to file {self.fileName}")
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

    def getProbeCall(self, minVolThick = 0, vars = "", name = None):

        if name == None:
            name = self.name
    
        probeCall = f"{self.type} NAME=$(probe_path)/{name:22} INTERVAL $(probe_int) "

        if self.type == "PROBE":
            probeCall += f"GEOM FILE $(location_path)/{name + '.txt' :26}" 
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
            normalVecAbs = np.abs(normalVec)
            probeCall += f"XP {means[0]:f} {means[1]:f} {means[2]:f} "
            probeCall += f"NP {normalVec[0]:f} {normalVec[1]:f} {normalVec[2]:f} VARS "
            for var in vars:
                if "sn-" in var:
                    var = var.replace("sn-", "")
                    u_comp = f"comp(u,{np.argmax(normalVecAbs)})"
                    var = var.replace("u", u_comp)
                    if var == u_comp:
                        probeCall += f"sn_prod({var},{normalVecAbs[0]:f},{normalVecAbs[1]:f},{normalVecAbs[2]:f}) " # use abs(NP) for directional quantities
                    else:
                        probeCall += f"sn_prod({var},{normalVec[0]:f},{normalVec[1]:f},{normalVec[2]:f}) " # effectively uses NP**2 to maintain scalar positive
                else:
                    probeCall += f"mass_flux({var}) "

        self.probeCall = probeCall

## From Jack:

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
