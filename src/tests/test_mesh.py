import glob
import os
import unittest
import trimesh
from src.geometry.numpy.transforms import *
from src.geometry.numpy.mesh import *
from src.util.timing import profile


class TestMesh(unittest.TestCase):
    def setUp(self):
        #self.off_files = glob.glob('data/example_off_files/*.off')
        self.mesh = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        self.off_files = {'data/wing_off_files/finished_fem_without_tip.off'}

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
            mesh.plot_faces(index_row=0, index_col=1, show=False, plotter=plotter, title="faces")
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
                            show=False, plotter=plotter, cmap=['black'], index_col=2, index_row=2)
            plotter.show(title=file)

    def test_camera_angle(self):
        plotter = pv.Plotter(shape=(3,3))
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
        plotter.set_viewup([0.1,1, 0])

        plotter.subplot(2, 2)
        plotter.set_position([132, -11, 941])
        plotter.set_focus([151.5, 60.5, 321.5])
        plotter.set_viewup([0, -1, -0.1])

        plotter.subplot(2, 0)
        plotter.show()
        print(plotter.camera_position)

    def test_Texture(self):
        plotter = pv.Plotter()
        mesh1 = Mesh("data/wing_off_files/fem_tip.off")
        mesh2 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh2.plot_faces(plotter=plotter, show=False, texture="data/textures/checkers2.png")
        mesh1.plot_faces(plotter=plotter, cmap=['white'], f=np.ones(mesh1.vertices.shape[0]))

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
        self.mesh.connected_component(plot=True,cmap=["white", "green", "blue"],
                                      title="connected component", plotter=plotter, show=False)
        tip.connected_component(plot=True, cmap=["black", "red"], plotter=plotter)

    def test_annimate(self):
        fg = []
        # the movement of all meshes
        g1 = []
        f1 = []
        f2 = []
        g2 = []
        g3 = []
        f3 = []
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
        for phase in np.linspace(0, 4 * np.pi, frames+1):
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
        fg.append(f1)
        fg.append(g1)
        fg.append(f2)
        fg.append(g2)
        fg.append(f3)
        fg.append(g3)
        cords = [(0, 0), (0, 0), (1, 0), (1,0), (2, 0), (2, 0)]
        # cords of the subplot, both mesh are in the same subplot so both needing to be the same
        plotter = pv.Plotter(shape=(3,1))

        self.mesh.main_cords(plot=True, index_row=0, index_col=0, scale=0.1, plotter=plotter, show=False)
        mesh2.main_cords(plot=True, index_row=1, index_col=0, scale=0.1, plotter=plotter, show=False)
        mesh3.main_cords(plot=True, index_row=2, index_col=0, scale=0.1, plotter=plotter, show=False)
        scalars = [None,None,None,None,None,None]
        textures = ["data/textures/checkers2.png", None, "data/textures/checkers2.png",
                    None, "data/textures/checkers2.png", None]
        color_maps = ["jet", "jet", "jet", "jet", "jet", "jet"]
        titles = ["big wave length", "","small wave length","","non decaying sin",""]
        font_colors = ["black", "black","black", "black","black", "black"]
        font_size = [10, 10, 10, 10, 10, 10]

        animate_few_meshes(mesh=meshes, movement=fg, f=scalars, num_of_plots=6, subplot=cords,
                           texture=textures, cmap=color_maps, plotter=plotter,
                           title=titles, font_size=font_size, font_color=font_colors,
                           gif_path="src/tests/temp/three_red_wings.gif")
        # ^ every argument should be given as a list, the default args for this function is for a single mesh, not more
        #self.mesh.animate(movement=f, texture="data/textures/cat.jpg", gif_path="src/tests/temp/")
        # ^ would animate a single mesh in a single subplot

    def test_draw_chess_board(self):
        draw_chessboard()

if __name__ == '__main__':
    unittest.main()
