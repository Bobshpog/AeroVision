import numpy as np
from scipy.io import loadmat

from src.geometry.numpy.mesh import write_off


def vertices_to_off(mat_path, off_path):
    """

    Args:
        mat_path:path to .mat file
        off_path: path to output .off location


    """
    mat = loadmat(mat_path)
    x = mat['X']
    y = mat['Y']
    z = mat['Z']
    vertices = np.hstack((x, y, z))
    write_off((vertices, []), off_path)


def read_data(mat_path):
    """

    Args:
        mat_path: path to mat

    Returns:
        (disp, scales): (nparray,nparray)
    """
    mat = loadmat(mat_path)
    dx = mat['U1'].T
    dy = mat['U2'].T
    dz = mat['U3'].T
    disp = np.dstack((dx, dy, dz))  # n x <num vertices> x 3
    scales = mat['xi'].T  # n x 5
    return disp, scales
