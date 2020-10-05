import numpy as np
from scipy.io import loadmat

from src.geometry.numpy.mesh import write_off


def vertices_to_off(mat_path,off_path):
    mat=loadmat(mat_path)
    x=mat['X']
    y=mat['Y']
    z=mat['Z']
    vertices=np.hstack((x,y,z))
    write_off((vertices,[]), off_path)

def read_data(mat_path):
    mat=loadmat(mat_path)
    dx=mat['U1'].T
    dy=mat['U2'].T
    dz=mat['U3'].T
    disp=np.dstack((dx,dy,dz))
    scales=mat['xi'].T #n x 5
    return disp,scales