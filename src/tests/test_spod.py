import time
from random import randint
from unittest import TestCase
from scipy.io import loadmat
from src.geometry.spod import *
from src.geometry.animations.synth_wing_animations import *
from src.geometry.numpy.mesh import *
import cv2
from util import loss_functions
from data import matlab_reader
class Test(TestCase):

    def test_SPOD(self):
        mat = loadmat("data/synt_data_mat_files/data.mat")
        x = mat['U1']
        y = mat['U2']
        z = mat['U3']
        p = mat["X"]
        print(p.shape, x.shape)
        size = np.append(x.shape[0],3)


        pod1 = SPOD(dt=0.1)
        pod1.fit(x)
        pod2 = SPOD(dt=0.1)
        pod2.fit(y)
        pod3 = SPOD(dt=0.1)
        pod3.fit(z)

        print(pod1.spod_modes.shape)
        coords = np.zeros(size)
        print(pod1.spod_modes.shape, coords.shape)
        coords[:,0] = pod1.spod_modes[:,0,0]
        coords[:,1] = pod2.spod_modes[:,0,0]
        coords[:,2] = pod3.spod_modes[:,0,0]
        plotter = pv.Plotter()
        plotter.add_mesh(coords, scalars=np.arange(x.shape[0]),cmap="jet")
        plotter.show()

    def test_IR_SPOD(self):
        plotter = pv.Plotter()

        mat = loadmat("data/synt_data_mat_files/mode_shape_27.mat")["ModeShapes"]
        x = loadmat("data/synt_data_mat_files/x0.mat")["x0"]
        y = loadmat("data/synt_data_mat_files/y0.mat")["y0"]

        x = x.flatten()
        y = y.flatten()
        coords1 = np.zeros((27,3))
        coords1[:,0] = x
        coords1[:,1] = y
        coords1[:,2] = mat[:,0]
        coords2 = np.zeros((27,3))
        coords2[:,0] = x
        coords2[:,1] = y
        coords2[:,2] = mat[:,1]
        print(coords1[0,2])
        plotter.add_mesh(coords1+coords2,scalars=np.arange(27),cmap="jet")
        plotter.show_axes_all()
        plotter.show()

    def test_mode_shapes_vis(self):
        mode_shape_path = "data/synt_data_mat_files/modes.mat"
        scale1 = loadmat("data/synt_data_mat_files/data.mat")["xi"]
        #scale2 = loadmat("data/synt_data_mat_files/data.mat")["xi"]
        scale2 = np.zeros(scale1.shape)
        vid_path = "src/tests/temp/creation_of_modees.mp4"
        trash_path = "src/tests/temp/video_frames/"
        texture_path = "data/textures/spheres.png"
        frames = 200
        num_of_scales = 5
        ids = [6419, 6756, 7033, 7333, 7635, 7937, 8239, 8541, 8841,  # first line
               6411, 6727, 7025, 7325, 7627, 7929, 8271, 8553, 8854,  # middle
               6361, 6697, 6974, 7315, 7576, 7919, 8199, 8482, 8782]
        create_animation_few_scales(scale1, vid_path, trash_path, texture_path, mode_shape_path, frames, num_of_scales,
                                    res=[480,480])

    def test_xyz(self):
        scales = matlab_reader.read_data("data/synt_data_mat_files/data_big.mat")[2]

        print(scales.shape)
        mode_shape = matlab_reader.read_modal_shapes("data/synt_data_mat_files/modes.mat",10)
        ids = [6419, 6756, 7033, 7333, 7635, 7937, 8239, 8541, 8841,  # first line
               6411, 6727, 7025, 7325, 7627, 7929, 8271, 8553, 8854,  # middle
               6361, 6697, 6974, 7315, 7576, 7919, 8199, 8482, 8782]
        #print(mode_shape.shape)
        def L2(a, b):
            return (a**2 - b**2)**0.5

        def v_L2(a,b):
            return np.linalg.norm(a-b)

        print(np.linalg.norm(1))
        #res = loss_functions.calc_max_errors(v_L2,scales,ids,mode_shape)

        #print(res)



