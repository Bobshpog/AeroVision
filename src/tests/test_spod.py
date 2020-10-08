import time
from random import randint
from unittest import TestCase
from scipy.io import loadmat
from src.geometry.spod import *
from src.geometry.animations.synth_wing_animations import *
from src.geometry.numpy.mesh import *


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
        scale2 = loadmat("data/synt_data_mat_files/data.mat")["xi"]
        vid_path = "src/tests/temp/creation_of_modees.mp4"
        trash_path = "src/tests/temp/video_frames/"
        texture_path = "data/textures/checkers_dark_blue.png"
        frames = 10
        num_of_scales = 5

        create_vid_by_scales(scale1, scale2, vid_path, trash_path, texture_path, mode_shape_path, frames, num_of_scales,
                             show_ssim=True)
        plotter = pv.Plotter()
        plotter.set_background("white")
        tip2 = Mesh('data/wing_off_files/fem_tip.off')
        mesh = Mesh('data/wing_off_files/synth_wing_v3.off')
        #tip2.plot_faces(plotter=plotter,show=False)
        #mesh.plot_faces(plotter=plotter,texture=texture_path,camera=camera_pos["up_high"])
        #print(plotter.camera_position)
