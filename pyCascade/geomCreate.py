from solid2 import *
from solid2.extensions.bosl2 import prismoid, xrot, translate
import numpy as np
from pyCascade import probeSetup
from scipy.spatial.transform import Rotation as R
# from numpy.polynomial import chebyshev as cheb
# from chaospy.quadrature import clenshaw_curtis

class ProbedGeom:
    def __init__(self, geom, probes = []):
        self.geom = geom
        self.probes = probes
        self.checkIntegrity()

    def __add__(self, x: "ProbedGeom"):
        """
        This makes u = a + b also aggegate the associated probes
        """
        return ProbedGeom(self.geom + x.geom, self.probes + x.probes)

    # def __radd__(self, x: "ProbedGeom"):
    #     """
    #     This makes u = a + b also aggegate the associated probes
    #     """
    #     return ProbedGeom(self.geom+x.geom, self.probes + x.probes)

    def __sub__(self, x: "ProbedGeom"):
        """
        This makes u = a - b also aggegate the associated probes
        """
        return ProbedGeom(self.geom-x.geom, self.probes + x.probes)

    def __mul__(self, x: "ProbedGeom"):
        """
        This makes u = a * b also aggegate the associated probes
        """
        return ProbedGeom(self.geom*x.geom, self.probes + x.probes)

    def translate(self, v):
        """
        translates geometry and probes
        """
        self.geom = translate(v)(self.geom)
        for probe_instance in self.probes: probe_instance.tile += v 

    def rotate(self, v):
        """
        rotates geometry and probes
        """
        self.geom = rotate(v)(self.geom)
        r = R.from_rotvec([a / 180 * np.pi for a in v])
        for probe_instance in self.probes:
            probe_instance.tile = r.apply(probe_instance.tile) 

    def scale(self, v):
        """
        scales geometry and probes
        """
        self.geom = scale(v)(self.geom)
        for probe_instance in self.probes: probe_instance.tile *= v

    def checkIntegrity(self):
        for probe in self.probes:
            foo = probe.tile.shape

    def append_names(self, text):
        for probe in self.probes:
            probe.name = f'{probe.name}_{text}'
            
    def removeZeroProbes(self):
        """
        removes probes with 0 points
        """
        self.probes = [probe for probe in self.probes if probe.tile.shape[0] != 0]

    def writeProbesToSingleFile(self, directory, nameInclude = "", nameExclude = "100gecs"):
        self.removeZeroProbes()
        nameList = []
        append = False
        for probe in self.probes:
            if nameInclude in probe.name and nameExclude not in probe.name:
                probe.fileName = f"{directory}{nameInclude}.txt"
                probe.writeProbes(append = append)
                nameList.append(probe.name)
                append = True
        if len(nameList) == 0:
            return
        with open(f"{directory}nameKey_{nameInclude}.txt",'w+') as out:
            for name in nameList:
                out.write(f'{name}\n')
        return

    def writeProbesToFiles(self, directory, nameInclude = "", nameExclude = "100gecs"):
        self.removeZeroProbes()
        for probe in self.probes:
            if nameInclude in probe.name and nameExclude not in probe.name:
                probe.fileName = f"{directory}{probe.name}.txt"
                probe.writeProbes()
        return
                    



def sumProbedGeom(items: "list"):
    for i, item in enumerate(items):
        if i == 0:
            summed = item
        else:
            summed += item
    return summed

def makeProbedCube(size, nprobes, name, centered = False, spacing = "flux"):
    geom = cube(size, centered)
    probe_span = []
    probeType = None
    for i,n in enumerate(nprobes):
        points = None
        dim = size[i]
        if n == 0:
            points = np.array([])
        elif spacing == "chebychev":
            cheby_points = np.arange(1,n+1)
            cheby_points = np.cos((2*cheby_points-1)*np.pi/(2*n))  #Chebyshev Nodes
            # cheby_points, _ = cheb.chebgauss(n)
            # print(cheby_points)
            cheby_points *= dim/2
            points = cheby_points
            probeType = "PROBE"
        elif spacing == "linear":
            lin_offest = (dim / 2) * (1 - 1/n)
            points = np.linspace(-lin_offest, lin_offest ,n) #linear spacing
            probeType = "PROBE"
        elif spacing == "volumetric":
            points = np.array([-dim / 2, dim / 2])
            probeType = "VOLUMETRIC_PROBE"
        elif spacing == "flux":
            if dim == np.min(size):
                ref_normal = 1 # normal is in the direction of the smallest dimension
            else:
                ref_normal = 0
            points = np.array([-ref_normal, ref_normal])
            probeType = "FLUX_PROBE"
        else:
            raise Exception(f"spacing {spacing} not recognized")
        if centered == False:
            points += dim/2
        probe_span.append(points)
    probes = probeSetup.Probes(name = name, type = probeType)
    probes.probe_fill(*probe_span)
    return ProbedGeom(geom, [probes])


def makeRooms(x, y, z, wthick = .01, nx=1, ny=1, nz=1):
    offset = wthick
    x_empty = x - 2 * wthick
    y_empty= y - 2 * wthick
    z_empty = z - 2 * wthick
    size = (x_empty, y_empty, z_empty)

    rooms_list = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                disp = (x*i + offset, y*j + offset, z*k + offset)
                rooms_list.append(translate(disp)(cube(size, False)))
    rooms = np.sum(rooms_list)

    rooms_params = {
        'x': x,
        'y': y,
        'z': z,
        'wthick': wthick,
        'nx': nx,
        'ny': ny,
        'nz': nz
    }

    return ProbedGeom(rooms), rooms_params


def identify_openings(rooms_params):
    nx = rooms_params['nx']
    nz = rooms_params['nz']

    window_locations = []
    door_locations = []
    skylight_locations = []

    for i in range(nx):
        for k in range(nz):
            if i == 0 or i == (nx-1):
                window_locations.append([i,k,'x'])
            if i > 0:
                door_locations.append([i,k,'x'])
            if k == 0 or k == (nz-1):
                window_locations.append([i,k,'z'])
            if k > 0:
                door_locations.append([i,k,'z'])
            skylight_locations.append([i,k])

    rooms_params['window_locations'] = window_locations
    rooms_params['door_locations'] = door_locations
    rooms_params['wall_locations'] = door_locations
    rooms_params['skylight_locations'] = skylight_locations

    return rooms_params


def makeRoof(l1, l2, w1, w2, h1, h2, extraProbeOffset = 0):
    """
    prismoid roof with pointing towards y
    """
    geom = prismoid([l1,w1], [l2,w2], h=h1)
    geom = xrot(-90)(geom)
    geom = translate([l1/2,h2,w1/2])(geom)
    roof = ProbedGeom(geom)
    extraProbeTile = np.array([[l1/2, h1 + h2 + extraProbeOffset, w1/2]], dtype = float)
    if extraProbeOffset != 0:
        roof.probes = [probeSetup.Probes(tile = extraProbeTile, name = f"extraProbe_roof", type = "PROBE")]

    return roof

def makeDoors(rooms_params, w, h, nprobes_w, nprobes_h):
    x = rooms_params['x']
    y = rooms_params['y']
    z = rooms_params['z']
    wthick = rooms_params['wthick']
    door_locations =  rooms_params['door_locations']

    doors_list = []
    for door_location in door_locations:
        i, k, orientation = door_location
        if orientation == 'x':
            disp = (x*i, y/2, z*(k+.5))
            size = (wthick*2.2, h, w)
            nprobes = (1, nprobes_h, nprobes_w)
            door = makeProbedCube(size, nprobes, f"xdoor_{i}-{k}", True)
        elif orientation == 'z':
            disp = (x*(i+.5), y/2, z*k)
            size = (w, h, wthick*2.2)
            nprobes = (nprobes_w, nprobes_h, 1)
            door = makeProbedCube(size, nprobes, f"zdoor_{i}-{k}", True)
        door.translate(disp)
        doors_list.append(door)

    return sumProbedGeom(doors_list)

def makeWindows(rooms_params, w, h, nprobes_w, nprobes_h, extraProbeOffset = 0):
    x = rooms_params['x']
    y = rooms_params['y']
    z = rooms_params['z']
    wthick = rooms_params['wthick']
    window_locations =  rooms_params['window_locations']

    windows_list = []
    for window_location in window_locations:
        i, k, orientation = window_location
        if orientation == 'x':
            edge_shift = wthick * (0.5 - (i!=0))
            disp = (x*(i+(i!=0)) + edge_shift, y/2, z*(k+.5))
            size = (wthick*1.1, h, w)
            nprobes = (1, nprobes_h, nprobes_w)
            name = f"xwindow_{i}-{k}"
            extraProbeTile = np.array([[extraProbeOffset, 0, 0]])
        elif orientation == 'z':
            edge_shift = wthick * (0.5 - (k!=0))
            disp = (x*(i+.5), y/2, z*(k+(k!=0)) + edge_shift)
            size = (w, h, wthick*1.1)
            nprobes = (nprobes_w, nprobes_h, 1)
            name = f"zwindow_{i}-{k}"
            extraProbeTile = np.array([[0, 0, extraProbeOffset]])
        window = makeProbedCube(size, nprobes, name, True)
        if extraProbeOffset != 0:
            window.probes += [probeSetup.Probes(tile = extraProbeTile, name = f"extraProbe_{name}", type = "PROBE")]
        window.translate(disp)
        windows_list.append(window)

    return sumProbedGeom(windows_list)

def openWalls(rooms_params, w, h, nprobes_w, nprobes_h):
    x = rooms_params['x']
    y = rooms_params['y']
    z = rooms_params['z']
    wthick = rooms_params['wthick']
    wall_locations =  rooms_params['wall_locations']

    walls_list = []
    for wall_location in wall_locations:
        i, k, orientation = wall_location
        if orientation == 'x':
            disp = (x*i, y/2, z*(k+.5))
            size = (wthick*2.2, h, w)
            nprobes = (1, nprobes_h, nprobes_w)
            door = makeProbedCube(size, nprobes, f"xwall_{i}-{k}", True)
        elif orientation == 'z':
            disp = (x*(i+.5), y/2, z*k)
            size = (w, h, wthick*2.2)
            nprobes = (nprobes_w, nprobes_h, 1)
            door = makeProbedCube(size, nprobes, f"zwall_{i}-{k}", True)
        door.translate(disp)
        walls_list.append(door)

    return sumProbedGeom(walls_list)

def makeSkylights(rooms_params, w, h, t, nprobes_w, nprobes_h, extraProbeOffset = 0):
    x = rooms_params['x']
    y = rooms_params['y']
    z = rooms_params['z']
    wthick = rooms_params['wthick']
    skylight_locations =  rooms_params['skylight_locations']

    skylights_list = []
    for skylight_location in skylight_locations:
        edge_shift = wthick * (-0.5)
        i, k = skylight_location
        disp = (x*(i+.5), y + edge_shift, z*(k+.5))
        size = (w, wthick, h)
        nprobes = (nprobes_w, 1, nprobes_h)
        name = f"skylight_{i}-{k}"
        skylight = makeProbedCube(size, nprobes, name, True)
        skylight += ProbedGeom(cube((w, t, h), True))
        extraProbeTile = np.array([[0, extraProbeOffset, 0]])
        if extraProbeOffset != 0:
            skylight.probes += [probeSetup.Probes(tile = extraProbeTile, name = f"extraProbe_{name}", type = "PROBE")]
        skylight.translate(disp)
        skylights_list.append(skylight)

    return sumProbedGeom(skylights_list)