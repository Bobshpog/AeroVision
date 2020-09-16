import glob
import os
import unittest

import cv2

from src.geometry.numpy.transforms import *
from src.geometry.numpy.wing_models import *
from src.geometry.spod import *
from src.util.timing import profile


class TestMesh(unittest.TestCase):
    class Config:
        num_of_vertices_wing = 7724
        num_of_vertices_tip = 930
        wing_path = "data/wing_off_files/finished_fem_without_tip.off"
        tip_path = "data/wing_off_files/fem_tip.off"
        camera_pos = {
            'up_middle': [(0.047, -0.053320266561896174, 0.026735639600027315),
                          (0.05, 0.3, 0.02),
                          (0, 0, 1)],

            'down_middle': [(0.04581499400545182, -0.04477050005202985, -0.028567355483893577),
                            (0.05, 0.3, 0.02),
                            (0.001212842435223535, 0.13947688005070646, -1)],

            "up_right": [(0.11460619078012961, -0.04553696541254279, 0.038810512823530784),
                         (0.05, 0.3, 0.02),
                         (0, 0.16643488101070833, 1)],

            'down_right': [(0.11460619078012961, -0.04553696541254279, -0.038810512823530784),
                           (0.05, 0.3, 0.02),
                           (0, 0.16643488101070833, -1)],
            'up_left': [(-0.019770941905445285, -0.06082136750543311, 0.038694507832388224),
                        (0.05, 0.3, 0.02),
                        (0.041, 0.0438, 1)],

            'down_left': [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
                          (0.05, 0.3, 0.02),
                          (0.041, 0.0438, -1)]
        }

    def setUp(self):
        # self.off_files = glob.glob('data/example_off_files/*.off')
        self.mesh = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        self.off_files = {'data/wing_off_files/combined_wing.off'}

    @profile
    def test_get_vertex_valence(self):
        self.mesh.get_vertex_valence()

    @profile
    def test_get_face_normals(self):
        self.mesh.get_face_normals()

    @profile
    def test_get_face_barycenters(self):
        self.mesh.get_face_barycenters()

    @profile
    def test_get_face_areas(self):
        self.mesh.get_face_areas()

    @profile
    def test_get_vertex_normals(self):
        self.mesh.get_vertex_normals()

    @profile
    def test_read_write_off(self):
        """"
        reads all off files, writes them to temp directory and calls diff

        Args
        files:  iterator over files to write
        """
        for file in self.off_files:
            data = read_off(file)
            temp_file = 'src/tests/temp/tempFile'
            write_off(data, temp_file)
            mesh_new = read_off(temp_file)
            self.assertTrue(np.array_equal(data[0], mesh_new[0]) and np.array_equal(data[1], mesh_new[1]))
            os.remove(temp_file)

    def test_visualization(self):
        """"
        for each image plots:
         - vertices
         - faces
         - wireframe
         - un-normalized normals of faces
         - normalized normals of faces
         - valence
         - vertex centroid with jet colormap


         Args
                path:  path for the .off file
        """
        for file in self.off_files:
            mesh = Mesh(file)

            # hyper parameter that decide the rate between the longest distance to the sphere to the size of the sphere
            size_of_black_sphere = 0.05

            plotter = pv.Plotter(shape=(3, 3))
            mesh.plot_vertices(index_row=0, index_col=0, show=False, plotter=plotter, title="vertices")
            mesh.plot_faces(index_row=0, index_col=1, show=False, plotter=plotter, title="faces",
                            camera=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            mesh.plot_wireframe(index_row=0, index_col=2, show=False, plotter=plotter, title="wireframe")

            faces_barycenter = mesh.get_face_barycenters()
            normals = mesh.get_face_normals()

            plotter.subplot(1, 0)
            plotter.add_text("un-normalized normals", position='upper_edge', font_size=10)
            plotter.add_arrows(faces_barycenter, normals)

            plotter.subplot(1, 1)
            plotter.add_text("normalized normals", position='upper_edge', font_size=10)
            normals = mesh.get_face_normals(norm=True)
            plotter.add_arrows(faces_barycenter, normals)

            mesh.plot_vertices(f=mesh.get_vertex_valence(),
                               index_row=1, index_col=2, show=False, plotter=plotter,
                               title='valance figure')
            plotter.subplot(2, 0)
            mean = mesh.vertices.mean(axis=0)
            distance = np.linalg.norm(mesh.vertices - mean, axis=1)
            print(mean)
            # L2 distance between the mean and the point
            max_dist = distance.max()
            mesh.plot_vertices(f=distance, index_row=2, index_col=0, show=False, plotter=plotter,
                               title="distance from mean")
            plotter.add_mesh(mesh=pv.Sphere(center=mean, radius=size_of_black_sphere * max_dist), color='black')

            mesh.connected_component(plot=True, index_row=2, index_col=1, show=False, plotter=plotter,
                                     title="CC", cmap=['red', 'green', 'blue'])
            mesh.main_cords(plot=True, show=False, plotter=plotter, index_row=2, index_col=2,
                            title="cords", font_color="white", scale=0.1)
            mesh.plot_faces(f=np.ones(mesh.vertices.shape[0]),
                            show=False, plotter=plotter, cmap=['white'], index_col=2, index_row=2)
            plotter.show(title=file)
            print(plotter.camera_position)

    def test_camera_angle(self):
        plotter = pv.Plotter(shape=(3, 3))
        self.mesh.plot_faces(show=False, plotter=plotter, title="front view")
        self.mesh.plot_faces(index_row=0, index_col=1, show=False, plotter=plotter, title="side view")
        self.mesh.plot_faces(index_row=0, index_col=2, show=False, plotter=plotter, title="below view")
        main_c = self.mesh.main_cords(plot=True, show=False, plotter=plotter, index_row=0, index_col=0)
        mesh2 = Mesh("data/wing_off_files/convex_hull.off")
        mesh2.plot_faces(index_row=1, index_col=0, show=False, plotter=plotter)
        mesh2.plot_faces(index_row=1, index_col=1, show=False, plotter=plotter, title="baseless wing")
        mesh2.plot_faces(index_row=1, index_col=2, show=False, plotter=plotter)
        self.mesh.plot_projection(index_row=2, index_col=0, show=False, plotter=plotter, normal=-main_c[2])
        self.mesh.plot_projection(index_row=2, index_col=1, show=False, plotter=plotter, normal=-main_c[1])
        self.mesh.plot_projection(index_row=2, index_col=2, show=False, plotter=plotter, normal=-main_c[0])

        # front view: pos: (122, 1063, -673) focus: (151.5, 60.5, 321.5) viewup: (0, 0,7, 0,7)
        # side view: pos:(2044,75,512) focus: (151.5, 60.5, 321.5) viewup:(-0.1,0.05,1)
        # below view: pos:(136,92,-860) focus: (151.5, 60.5, 321.5) viewup:(0,0,0)

        # for 2D angle (not the same angle as 3D..):
        # front view: pos: (340, 1411, 400) focus: (151.5, 60.5, 321.5) viewup: (0.1, 0.5, 1)
        # side view: pos:(1540, -64, 235) focus: (151.5, 60.5, 321.5) viewup:(0.1,1, 0)
        # below view: pos:(132, -11, 941) focus: (151.5, 60.5, 321.5) viewup:(0, -1, -0.1)

        plotter.subplot(0, 0)
        plotter.set_position([122, 1063, -673])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, 0.7, 0.7])
        plotter.subplot(0, 1)
        plotter.set_position([2044, 75, 512])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, 0, 0])
        plotter.subplot(0, 2)
        plotter.set_position([136, 92, -331])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, 0, 0])

        plotter.subplot(1, 0)
        plotter.set_position([122, 1063, -673])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, 0.7, 0.7])

        plotter.subplot(1, 1)
        plotter.set_position([2044, 75, 512])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, 0, 0])

        plotter.subplot(1, 2)
        plotter.set_position([136, 92, -331])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, 0, 0])

        plotter.subplot(2, 0)
        plotter.set_position([340, 1411, 400])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0.1, 0.5, 1])

        plotter.subplot(2, 1)
        plotter.set_position([1540, -64, 235])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0.1, 1, 0])

        plotter.subplot(2, 2)
        plotter.set_position([132, -11, 941])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, -1, -0.1])

        plotter.subplot(2, 0)
        plotter.show()
        print(plotter.camera_position)

    def test_Texture(self):
        plotter = pv.Plotter(shape=(1, 2))
        cam = [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
               (0.05, 0.3, 0.02),
               (0.041, 0.0438, -1)]
        cam = self.Config.camera_pos['down_middle']
        mesh2 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh1 = Mesh('data/wing_off_files/fem_tip.off')
        # cam = FemNoTip.camera_pos["up_middle"]
        mesh2.main_cords(plot=True, show=False, plotter=plotter, scale=0.1)
        mesh2.main_cords(plot=True, show=False, plotter=plotter, index_row=0, index_col=1, scale=0.1)
        mesh2.plot_faces(plotter=plotter, show=False, camera=cam, texture="data/textures/checkers2.png",
                         title="without parallel projection")
        mesh1.plot_faces(plotter=plotter, show=False, index_row=0, index_col=0)
        mesh1.plot_faces(plotter=plotter, show=False, index_row=0, index_col=1)
        mesh2.plot_faces(plotter=plotter, index_row=0, index_col=1, title="with parallel projection",
                         texture="data/textures/checkers2.png", camera=cam, show=False, depth=True)
        plotter.show()
        print(plotter.camera_position)

    def test_make_wing(self):
        unwanted_points_axis = 0.095
        labels = []
        for ver in self.mesh.vertices:
            if ver[1] >= 0.605:
                labels.append(ver)
        new_ver = []
        for v in labels:

            for i in range(30):
                x = (0.008 * np.cos(np.pi * i / 15))
                y = (0.008 * np.sin(np.pi * i / 15))
                new_ver.append(v + (0, x, y))
        write_off((np.array(new_ver), np.array([])), "src/tests/temp/fem_tip_take3.off")

    def test_projection(self):
        self.mesh.plot_projection(texture="data/textures/checkers.png")

    def test_connected(self):
        plotter = pv.Plotter()
        tip = Mesh('data/wing_off_files/fem_tip.off')
        self.mesh.connected_component(plot=True, cmap=["white", "green", "blue"],
                                      title="connected component", plotter=plotter, show=False)
        tip.connected_component(plot=True, cmap=["black", "red"], plotter=plotter)

    def test_merge_wing(self):
        mesh2 = Mesh("data/wing_off_files/fem_tip.off")
        new_ver = np.append(self.mesh.vertices, mesh2.vertices, axis=0)
        num_of_ver = 7724
        increase = np.array([num_of_ver, num_of_ver, num_of_ver])
        new_tip_f = mesh2.faces + increase
        new_faces = np.append(self.mesh.faces, new_tip_f, axis=0)
        # write_off((new_ver,new_faces), "src/tests/temp/combined_wing.off")

    def test_annimate_three(self):
        # the movement of all meshes
        g1 = []
        f1 = []
        f2 = []
        g2 = []
        f3 = []
        g3 = []
        # we need to create 6 different meshes, three of tips and three for wing. Pyvista will not recognise the
        # meshes as different otherwise.
        tip1 = Mesh('data/wing_off_files/fem_tip.off')
        tip2 = Mesh('data/wing_off_files/fem_tip.off')
        tip3 = Mesh('data/wing_off_files/fem_tip.off')
        mesh2 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh3 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        meshes = [self.mesh, tip1, mesh2, tip2, mesh3, tip3]
        # ^ define the order of each mesh
        frames = 60
        # number of frames of the gif, if no gif should be created this number should be around the 4000~ to make it
        # the same as 60~ with gif is created
        for phase in np.linspace(0, 4 * np.pi, frames + 1):
            f1.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=self.mesh.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g1.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip1.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))

            f2.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                          freq_t=1, freq_s=25, amp=0.2, t=phase))
            g2.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip2.vertices,
                                          freq_t=1, freq_s=25, amp=0.2, t=phase))

            f3.append(np.apply_along_axis(fem_wing_normal_sine, axis=1, arr=mesh3.vertices,
                                          freq_t=1, freq_s=25, amp=0.2, t=phase))

            g3.append(np.apply_along_axis(fem_tip_normal_sine, axis=1, arr=tip3.vertices,
                                          freq_t=1, freq_s=25, amp=0.2, t=phase))
            # couldnt vectorise

        fg = [f1, g1, f2, g2, f3, g3]
        cords = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0), (2, 0)]
        # cords of the subplot, both mesh are in the same subplot so both needing to be the same
        plotter = pv.Plotter(shape=(3, 1))

        self.mesh.main_cords(plot=True, index_row=0, index_col=0, scale=0.1, plotter=plotter, show=False)
        mesh2.main_cords(plot=True, index_row=1, index_col=0, scale=0.1, plotter=plotter, show=False)
        mesh3.main_cords(plot=True, index_row=2, index_col=0, scale=0.1, plotter=plotter, show=False)
        scalars = [None] * 6
        textures = ["data/textures/checkers2.png", None] * 3
        color_maps = ["jet"] * 6
        titles = ["big wave length", "", "small wave length", "", "non decaying sin", ""]
        font_colors = ["black"] * 6
        font_size = [10, 10, 10, 10, 10, 10]
        cam = [(0.005, -0.2, 0.01), (0.047, 0.3, 0), (0, 0, 1)]
        animate_few_meshes(mesh=meshes, movement=fg, f=scalars, subplot=cords,
                           texture=textures, cmap=color_maps, plotter=plotter,
                           title=titles, font_size=font_size, font_color=font_colors,
                           gif_path="src/tests/temp/three_red_wings2.gif",
                           camera=[cam, cam, cam, cam, cam, cam], depth=False
                           )
        # ^ every argument should be given as a list, the default args for this function is for a single mesh, not more
        # self.mesh.animate(movement=f, texture="data/textures/cat.jpg", gif_path="src/tests/temp/")
        # ^ would animate a single mesh in a single subplot

    @profile
    def test_annimate_six(self):
        # the movement for each of the meshes
        g1 = []
        f1 = []
        f2 = []
        g2 = []
        f3 = []
        g3 = []
        f4 = []
        g4 = []
        f5 = []
        g5 = []
        f6 = []
        g6 = []
        f7 = []
        g7 = []
        # we need to create 6 different meshes, three of tips and three for wing. Pyvista will not recognise the
        # meshes as different otherwise.
        tip1 = Mesh('data/wing_off_files/fem_tip.off')
        tip2 = Mesh('data/wing_off_files/fem_tip.off')
        tip3 = Mesh('data/wing_off_files/fem_tip.off')
        mesh1 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh2 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh3 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        tip4 = Mesh('data/wing_off_files/fem_tip.off')
        tip5 = Mesh('data/wing_off_files/fem_tip.off')
        tip6 = Mesh('data/wing_off_files/fem_tip.off')
        mesh4 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh5 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh6 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        tip7 = Mesh('data/wing_off_files/fem_tip.off')
        mesh7 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        meshes = [mesh1, tip1, mesh2, tip2, mesh3, tip3, mesh4, tip4, mesh5, tip5, mesh6, tip6, mesh7, tip7]
        # ^ define the order of each mesh
        frames = 40
        # number of frames of the gif, if no gif should be created this number should be around the 4000~ to make it
        # the same as 60~ with gif is created
        for phase in np.linspace(0, 4 * np.pi, frames + 1):
            f1.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh1.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g1.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip1.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))

            f2.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g2.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip2.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))

            f3.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh3.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g3.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip3.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))

            f4.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh4.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g4.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip4.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))

            f5.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh5.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g5.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip5.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))

            f6.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh6.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g6.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip6.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            f7.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh7.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            g7.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip7.vertices,
                                          freq_t=1, freq_s=1, amp=0.2, t=phase))
            # couldnt vectorise
        # the movement list
        fg = [f1, g1, f2, g2, f3, g3, f4, g4, f5, g5, f6, g6, f7, g7]
        cords = [(0, 0), (0, 0), (0, 1), (0, 1), (0, 2), (0, 2), (1, 0), (1, 0), (1, 1), (1, 1), (1, 2), (1, 2), (2, 1),
                 (2, 1)]
        # cords of the subplot, both mesh are in the same subplot so both needing to be the same
        plotter = pv.Plotter(shape=(3, 3))
        for i in range(3):
            plotter.subplot(2, i)
            plotter.add_mesh(mesh=pv.Sphere(center=self.Config.camera_pos["up_right"][0], radius=0.01),
                             color='black')
            plotter.add_mesh(mesh=pv.Sphere(center=self.Config.camera_pos["up_left"][0], radius=0.01),
                             color='black')
            plotter.add_mesh(mesh=pv.Sphere(center=self.Config.camera_pos["up_middle"][0], radius=0.01),
                             color='black')
            plotter.add_mesh(mesh=pv.Sphere(center=self.Config.camera_pos["down_right"][0], radius=0.01),
                             color='black')
            plotter.add_mesh(mesh=pv.Sphere(center=self.Config.camera_pos["down_left"][0], radius=0.01),
                             color='black')
            plotter.add_mesh(mesh=pv.Sphere(center=self.Config.camera_pos["down_middle"][0], radius=0.01),
                             color='black')

        mesh1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=0, texture="data/textures/checkers2.png")
        mesh1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=2, texture="data/textures/checkers2.png")
        tip1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=0)
        tip1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=2)

        scalars = [None] * 14
        textures = ["data/textures/checkers2.png", None] * 7
        color_maps = ["jet"] * 14
        titles = ["up left", "", "up middle", "", "up right", "", "down left", "", "down middle", "", "down right", "",
                  " camera view", ""]
        font_colors = ["black"] * 14
        font_size = [10] * 14
        cam = [self.Config.camera_pos["up_left"], self.Config.camera_pos["up_left"],
               self.Config.camera_pos["up_middle"],
               self.Config.camera_pos["up_middle"], self.Config.camera_pos["up_right"],
               self.Config.camera_pos["up_right"],
               self.Config.camera_pos["down_left"], self.Config.camera_pos["down_left"],
               self.Config.camera_pos["down_middle"],
               self.Config.camera_pos["down_middle"], self.Config.camera_pos["down_right"],
               self.Config.camera_pos["down_right"],
               None, None
               ]
        animate_few_meshes(mesh=meshes, movement=fg, f=scalars, subplot=cords,
                           texture=textures, cmap=color_maps, plotter=plotter,
                           title=titles, font_size=font_size, font_color=font_colors,
                           gif_path="src/tests/temp/camera_positions.gif",
                           camera=cam, depth=False
                           )

    def test_depth_screenshot(self):
        mesh2 = Mesh("data/wing_off_files/fem_tip.off")
        plotter = pv.Plotter(off_screen=True)
        frames = 40
        url = "src/tests/temp/video_frames/"
        i = 0
        im_frames = []
        for phase in np.linspace(0, 4 * np.pi, frames):
            i = i + 1
            f1 = np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=self.mesh.vertices,
                                     freq_t=1, freq_s=1, amp=0.2, t=phase)
            g1 = np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                     freq_t=1, freq_s=1, amp=0.2, t=phase)

            photo = Mesh.get_photo([self.mesh, mesh2], [f1, g1], plotter=plotter,
                                   texture=["data/textures/checkers2.png", None],
                                   cmap=[None, None], camera=self.Config.camera_pos["up_left"], resolution=[640, 480])
            depth = photo[:, :, -1]
            depth2 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
            cv2.imshow("frame", depth2)
            cv2.imwrite(url + "depth_frame" + str(i) + ".jpg", depth2)
            img = cv2.imread(url + "depth_frame" + str(i) + ".jpg")
            im_frames.append(img)
            # cv2 does not support making video from np array...
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out = cv2.VideoWriter("src/tests/temp/depth_video.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, (640, 480))
        for i in range(len(im_frames)):
            out.write(im_frames[i])
        out.release()
        for f in glob.glob(url + '*.jpg'):
            os.remove(f)
        cv2.destroyAllWindows()

    def test_spod(self):
        spod = SPOD(n_components=3)
        spod.fit(self.mesh.vertices)
        x = spod.transform(self.mesh.vertices)
        print(x.shape)
        pass


def colored_checkerboard(h=640, w=480, tile_size=5, rgb1=(0.5, 0, 0.5), rgb2=(0, 0.8, 0.8)):
    mult_h = np.ceil(h / tile_size)
    mult_w = np.ceil(w / tile_size)
    zero_one_grid = ((np.arange(mult_w)[:, None] + np.arange(mult_h)[:, None].T) % 2).astype(bool)
    B = np.kron(zero_one_grid, np.ones((tile_size, tile_size), dtype=bool))
    B = B[:h, :w, None]
    return B * np.reshape(rgb1, (1, 1, 3)) + (~B) * np.reshape(rgb2, (1, 1, 3))
    # plt.imshow(colored_checkerboard(tile_size=10))
    # plt.show()


if __name__ == '__main__':
    unittest.main()
