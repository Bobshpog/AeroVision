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