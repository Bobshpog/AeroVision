import glob
import os
import unittest
import vtk
import cv2

from src.geometry.numpy.transforms import *
from src.geometry.numpy.wing_models import *
from src.geometry.spod import *
from src.util.timing import profile
# from src.geometry.numpy.animations import *
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
from datetime import datetime
import matplotlib.pyplot as plt


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
                          (0.041, 0.0438, -1)],

        }

    def setUp(self):
        # self.off_files = glob.glob('data/example_off_files/*.off')
        self.mesh = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        self.off_files = {'data/wing_off_files/synth_wing_v3.off'}

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

    def test_Texture(self):
        plotter = pv.Plotter(shape=(1, 2))
        plotter.set_background("white")
        plotter2 = pv.Plotter(shape=(1, 2))
        plotter2.set_background("white")

        tex = "data/textures/circles_normal.png"
        tex2 = "data/textures/circles3.png"
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        mesh2 = Mesh("data/wing_off_files/fem_tip.off")
        camera = [(0.047, -0.053320266561896174, 0.026735639600027315),
                  (0.05, 0.3, 0.02),
                  (0, 0, 1)]
        mesh2.plot_faces(plotter=plotter2, show=False, camera=camera)
        mesh2.plot_faces(plotter=plotter2, show=False, camera=camera, index_col=1)
        mesh.plot_faces(plotter=plotter2, texture="data/textures/checkers_blue_dot.png", show=False)
        mesh.plot_faces(plotter=plotter2, texture="data/textures/checkers_blue_dot.png", index_col=1)
        mesh2.plot_faces(plotter=plotter, camera=camera, show=False)
        mesh2.plot_faces(plotter=plotter, camera=camera, show=False, index_col=1)
        mesh.plot_faces(plotter=plotter, texture=tex, camera=camera, title="normal ratio", show=False)
        mesh.plot_faces(plotter=plotter, texture=tex2, camera=camera, title="3:1 ratio", index_col=1)

    def test_make_tip(self):  # the creation of the tip
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

    def test_six_still_pictures(self):
        mesh = Mesh('data/wing_off_files/combined_wing.off')
        mesh3 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
        mesh2 = Mesh('data/wing_off_files/fem_tip.off')
        width = 0.15
        color = "39FF14"
        cam = [camera_pos["up_left"][0], camera_pos["up_middle"][0], camera_pos["up_right"][0],
               camera_pos["rotated_down_left"][0], camera_pos["down_middle"][0], camera_pos["down_right"][0]]
        ids = [753, 9, 23, 120, 108, 38, 169, 1084, 53, 393, 1416, 68, 378, 1748, 83, 143, 3416, 3434,
               2079, 190, 348, 2537, 205, 333, 3559, 220, 318, 2648, 235, 303, 2711, 250, 3409, 3424, 3439]
        tip_ids = [906, 727, 637, 457, 368, 142, 7]
        distance = []
        subplots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        angle = [camera_pos["up_left"], camera_pos["up_middle"], camera_pos["up_right"],
                 camera_pos["rotated_down_left"], camera_pos["rotated_down_middle"], camera_pos["rotated_down_right"]]
        for i in range(6):
            distance.append(np.linalg.norm(mesh.vertices - cam[i], axis=1))
        plotter = pv.Plotter(shape=(2, 3))  # ir
        plotter.set_background("white")
        plotter2 = pv.Plotter(shape=(2, 3))  # depth
        plotter2.set_background("white")
        plotter3 = pv.Plotter(shape=(2, 3), off_screen=True)  # rgb
        plotter3.set_background("white")
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh2.vertices[tip_id], radius=0.003), color=color)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh3.vertices[v_id], radius=0.003), color=color)

        plotter.subplot(0, 1)
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh2.vertices[tip_id], radius=0.003), color=color)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh3.vertices[v_id], radius=0.003), color=color)

        plotter.subplot(0, 2)
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh2.vertices[tip_id], radius=0.003), color=color)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh3.vertices[v_id], radius=0.003), color=color)

        plotter.subplot(1, 0)
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh2.vertices[tip_id], radius=0.003), color=color)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh3.vertices[v_id], radius=0.003), color=color)

        plotter.subplot(1, 1)
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh2.vertices[tip_id], radius=0.003), color=color)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh3.vertices[v_id], radius=0.003), color=color)

        plotter.subplot(1, 2)
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh2.vertices[tip_id], radius=0.003), color=color)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=mesh3.vertices[v_id], radius=0.003), color=color)

        for i in range(6):
            plotter2.subplot(subplots[i][0], subplots[i][1])
            plotter2.set_position(angle[i][0])
            plotter2.set_focus(angle[i][1])
            plotter2.set_viewup(angle[i][2])
            plotter2.add_mesh(mesh.pv_mesh, scalars=distance[i], cmap="binary_r")

            plotter3.subplot(subplots[i][0], subplots[i][1])
            plotter3.set_position(angle[i][0])
            plotter3.set_focus(angle[i][1])
            plotter3.set_viewup(angle[i][2])
            mesh3.plot_faces(plotter=plotter3, texture="data/textures/circles_tex.jpg", show=False,
                             index_row=subplots[i][0], index_col=subplots[i][1])
            mesh2.plot_faces(plotter=plotter3, show=False,
                             index_row=subplots[i][0], index_col=subplots[i][1])
        # plotter2.show()
        plotter3.show(auto_close=False)
        screen = plotter3.screenshot()
        # screen = screen/255
        b = np.copy(screen[:, :, 0])
        screen[:, :, 0] = screen[:, :, 2]
        screen[:, :, 2] = b
        noise = np.random.poisson(0.05, size=b.shape).astype(np.uint8)
        screen[:, :, 0] += noise
        screen[:, :, 1] += noise
        screen[:, :, 2] += noise
        screen = screen / 255
        noise = np.random.normal(0.1, 0.1, size=b.shape)
        screen[:, :, 0] += noise
        screen[:, :, 1] += noise
        screen[:, :, 2] += noise
        cv2.imshow("frame", screen)
        cv2.waitKey()
        mesh2.plot_wireframe(line_width=width, plotter=plotter, camera=camera_pos["up_left"], show=False)
        mesh3.plot_wireframe(line_width=width, plotter=plotter, camera=camera_pos["up_left"], show=False)

        mesh2.plot_wireframe(line_width=width, plotter=plotter, camera=camera_pos["up_middle"], show=False, index_col=1,
                             index_row=0)
        mesh3.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["up_middle"], show=False, index_col=1,
                             index_row=0)

        mesh2.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["up_right"], show=False, index_col=2,
                             index_row=0)
        mesh3.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["up_right"], show=False, index_col=2,
                             index_row=0, )

        mesh2.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["rotated_down_left"], show=False,
                             index_col=0, index_row=1)
        mesh3.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["rotated_down_left"], show=False,
                             index_col=0, index_row=1)

        mesh2.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["rotated_down_middle"], show=False,
                             index_col=1, index_row=1)
        mesh3.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["rotated_down_middle"], show=False,
                             index_col=1, index_row=1)

        mesh2.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["rotated_down_right"], show=False,
                             index_col=2, index_row=1)
        mesh3.plot_wireframe(plotter=plotter, line_width=width, camera=camera_pos["rotated_down_right"], show=False,
                             index_col=2, index_row=1)

        plotter.subplot(0, 1)
        plotter3.close()
        plotter2.show()
        # plotter.show()

    def test_xyz(self):
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        tip = Mesh("data/wing_off_files/fem_tip.off")
        plotter = pv.Plotter()
        plotter.set_background("white")
        tip.plot_faces(show=False, plotter=plotter)
        mesh.plot_faces(show=False, plotter=plotter, texture="data/textures/circles_normal.png",
                        camera=camera_pos["up_middle"])
        vmtx = plotter.camera.GetModelViewTransformMatrix()
        print(type(vmtx))
        mtx = pv.trans_from_matrix(vmtx)

        print(mtx.shape)
        print(dir(plotter.camera))
        # plotter.camera.SetUseExplicitProjectionTransformMatrix(True)
        # plotter.camera.SetExplicitProjectionTransformMatrix(vmtx.Invert())
        # plotter.show()
        print(mtx)
        # synth_ir_video("src/tests/temp/ir_synth.mp4")

    def test_animate(self):
        # synth_wing_animation("src/tests/temp/animation_synth.mp4")
        plotter = pv.Plotter(shape=(1, 3))
        plotter.set_background("white")
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        tip = Mesh("data/wing_off_files/fem_tip.off")
        A = colored_checkerboard(tile_size=70, h=640, w=480, rgb1=(1.0, 1.0, 1.0), rgb2=(0, 0, 0))
        A = (A * 255).astype(np.uint8)
        B = colored_checkerboard(tile_size=15, h=100, w=300, rgb1=(1.0, 1.0, 1.0), rgb2=(0, 0, 0))
        B = (B * 255).astype(np.uint8)
        C = colored_checkerboard(tile_size=15, h=100, w=600, rgb1=(1.0, 1.0, 1.0), rgb2=(0, 0, 0))
        C = (C * 255).astype(np.uint8)
        tex1 = pv.numpy_to_texture(A)
        tex2 = pv.numpy_to_texture(B)
        tex3 = pv.numpy_to_texture(C)
        tip.plot_faces(show=False, plotter=plotter, camera=camera_pos["up_middle"], title="480:640")
        tip.plot_faces(show=False, plotter=plotter, camera=camera_pos["up_middle"], index_col=1, title="1:3")
        tip.plot_faces(show=False, plotter=plotter, camera=camera_pos["up_middle"], index_col=2, title="1:6")
        mesh.pv_mesh.texture_map_to_plane(inplace=True)
        plotter.subplot(0, 0)
        plotter.add_mesh(mesh.pv_mesh, texture=tex1)
        plotter.subplot(0, 1)
        plotter.add_mesh(mesh.pv_mesh, texture=tex2)
        plotter.subplot(0, 2)
        plotter.add_mesh(mesh.pv_mesh, texture=tex3)
        plotter.show()

    def test_improve_synth_wing(self):
        v, f, _ = read_off("data/wing_off_files/synth_wing_v1.off")
        a = np.array([[6481, 6460, 4717], [6460, 6440, 3341], [4717, 4722, 4703]])
        print(f.shape)
        f2 = np.append(f, a, 0)
        # write_off((v,f2),"data/wing_off_files/synth_wing_v3.off")
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")

        print(f2.shape)





def colored_checkerboard(h=640, w=480, tile_size=5, rgb1=(0.5, 0, 0.5), rgb2=(0, 0.8, 0.8)):
    mult_h = np.ceil(h / tile_size)
    mult_w = np.ceil(w / tile_size)
    zero_one_grid = ((np.arange(mult_w)[:, None] + np.arange(mult_h)[:, None].T) % 2).astype(bool)
    B = np.kron(zero_one_grid, np.ones((tile_size, tile_size), dtype=bool))
    B = B[:h, :w, None]
    return B * np.reshape(rgb1, (1, 1, 3)) + (~B) * np.reshape(rgb2, (1, 1, 3))


def homogenize(v, value=1):
    v = np.asanyarray(v)
    if hasattr(value, '__len__'):
        return np.append(v, np.asanyarray(value).reshape(v.shape[:-1] + (1,)), axis=-1)
    else:
        return np.insert(v, v.shape[-1], np.array(value, v.dtype), axis=-1)


def dehomogenize(a):
    """ makes homogeneous vectors inhomogenious by dividing by the last element in the last axis
    >>> dehomogenize([1, 2, 4, 2]).tolist()
    [0.5, 1.0, 2.0]
    >>> dehomogenize([[1, 2], [4, 4]]).tolist()
    [[0.5], [1.0]]
    """
    a = np.asfarray(a)
    return a[..., :-1] / a[..., np.newaxis, -1]


class Test2(unittest.TestCase):

    def test__wing_dist_histogram(self):
        TIP_PATH = "data/wing_off_files/synth_wing_v3.off"
        WING_PATH = "data/wing_off_files/fem_tip.off"
        tip = Mesh(TIP_PATH)
        wing = Mesh(WING_PATH)
        vertices = np.vstack([tip.vertices, wing.vertices])
        vertices = vertices[:, 1]
        plt.hist(vertices,bins=np.linspace(0,0.8,1024))
        plt.title(vertices.mean())
        plt.show()
        pass
if __name__ == '__main__':
    unittest.main()
