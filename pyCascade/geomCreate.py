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

    def __add__(self, x: "ProbedGeom"):
        """
        This makes u = a + b also aggegate the associated probes
        """
        return ProbedGeom(self.geom+x.geom, self.probes + x.probes)

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
        for probe_instance in self.probes: probe_instance["tile"]+=v 

    def rotate(self, v):
        """
        rotates geometry and probes
        """
        self.geom = rotate(v)(self.geom)
        r = R.from_rotvec([a / 180 * np.pi for a in v])
        for probe_instance in self.probes:
            probe_instance["tile"] = r.apply(probe_instance["tile"]) 

    def scale(self, v):
        """
        scales geometry and probes
        """
        self.geom = scale(v)(self.geom)
        for probe_instance in self.probes: probe_instance["tile"]*=v

    def append_names(self, text):
        for probe in self.probes:
            probe["name"] = f'{probe["name"]}_{text}'
            
    def removeZeroProbes(self):
        """
        removes probes with 0 points
        """
        self.probes = [probe for probe in self.probes if probe["tile"].shape[0] != 0]

    def write_probes(self, directory):
        self.removeZeroProbes()
        for probe in self.probes:
            tile = probe["tile"]
            fileName = f"{directory}{probe['name']}.txt"
            probeSetup.writeProbes(tile, fileName)


def sumProbedGeom(items: "list"):
    for i, item in enumerate(items):
        if i == 0:
            summed = item
        else:
            summed += item
    return summed

def makeProbedCube(size, nprobes, name, centered = False, spacing = "linear"):
    geom = cube(size, centered)
    probe_span = []
    for i,n in enumerate(nprobes):
        points = None
        if n == 0:
            points = np.array([])
        elif spacing == "chebychev":
            cheby_points = np.arange(1,n+1)
            cheby_points = np.cos((2*cheby_points-1)*np.pi/(2*n))  #Chebyshev Nodes
            # cheby_points, _ = cheb.chebgauss(n)
            # print(cheby_points)
            cheby_points *= size[i]/2
            points = cheby_points
        elif spacing == "linear":
            lin_offest = (size[i] / 2) * (1 - 1/n)
            points = np.linspace(-lin_offest, lin_offest ,n) #linear spacing
        elif spacing == "volumetric":
            points = np.array([-size[i] / 2, size[i] / 2])
        else:
            raise Exception(f"spacing {spacing} not recognized")
        if centered == False:
            points += size[i]/2
        probe_span.append(points)
    tile = probeSetup.probe_fill(*probe_span)
    probes = [{
        "tile": tile,
        "name": name
    }]
    return ProbedGeom(geom, probes)


def makeRooms(x, y, z, wthick = .01, nx=1, ny=1, nz=1):
    offset = wthick/2
    x_empty = x - wthick
    y_empty= y - wthick
    z_empty = z - wthick
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


def makeRoof(l1, l2, w1, w2, h1, h2):
    """
    prismoid roof with pointing towards y
    """
    
    geom = prismoid([l1,w1], [l2,w2], h=h1)
    geom = xrot(-90)(geom)
    geom = translate([l1/2,h2,w1/2])(geom)

    return ProbedGeom(geom)

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
            size = (wthick*2, h, w)
            nprobes = (1, nprobes_h, nprobes_w)
            door = makeProbedCube(size, nprobes, f"xdoor_{i}-{k}", True)
            door.translate(disp)
        elif orientation == 'z':
            disp = (x*(i+.5), y/2, z*k)
            size = (w, h, wthick*2)
            nprobes = (nprobes_w, nprobes_h, 1)
            door = makeProbedCube(size, nprobes, f"zdoor_{i}-{k}", True)
            door.translate(disp)
        doors_list.append(door)

    return sumProbedGeom(doors_list)

def makeWindows(rooms_params, w, h, nprobes_w, nprobes_h):
    x = rooms_params['x']
    y = rooms_params['y']
    z = rooms_params['z']
    wthick = rooms_params['wthick']
    window_locations =  rooms_params['window_locations']

    windows_list = []
    for window_location in window_locations:
        i, k, orientation = window_location
        if orientation == 'x':
            disp = (x*(i+(i!=0)), y/2, z*(k+.5))
            size = (wthick*2, h, w)
            nprobes = (1, nprobes_h, nprobes_w)
            window = makeProbedCube(size, nprobes, f"xwindow_{i}-{k}", True)
            window.translate(disp)
        elif orientation == 'z':
            disp = (x*(i+.5), y/2, z*(k+(k!=0)))
            size = (w, h, wthick*2)
            nprobes = (nprobes_w, nprobes_h, 1)
            window = makeProbedCube(size, nprobes, f"zwindow_{i}-{k}", True)
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
            size = (wthick*2, h, w)
            nprobes = (1, nprobes_h, nprobes_w)
            door = makeProbedCube(size, nprobes, f"xwall_{i}-{k}", True)
            door.translate(disp)
        elif orientation == 'z':
            disp = (x*(i+.5), y/2, z*k)
            size = (w, h, wthick*2)
            nprobes = (nprobes_w, nprobes_h, 1)
            door = makeProbedCube(size, nprobes, f"zwall_{i}-{k}", True)
            door.translate(disp)
        walls_list.append(door)

    return sumProbedGeom(walls_list)

def makeSkylights(rooms_params, w, h, nprobes_w, nprobes_h):
    x = rooms_params['x']
    y = rooms_params['y']
    z = rooms_params['z']
    wthick = z*.5
    skylight_locations =  rooms_params['skylight_locations']

    skylights_list = []
    for skylight_location in skylight_locations:
        i, k = skylight_location
        disp = (x*(i+.5), y, z*(k+.5))
        size = (w, wthick*2, h)
        nprobes = (nprobes_w, 1, nprobes_h)
        skylight = makeProbedCube(size, nprobes, f"skylight_{i}-{k}", True)
        skylight.translate(disp)
        skylights_list.append(skylight)

    return sumProbedGeom(skylights_list)