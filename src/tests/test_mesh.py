import glob
import unittest
import os
from src.util.timing import profile

from src.geometry.numpy.mesh import *


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.off_files = glob.glob('data/example_off_files/*.off')
        self.mesh = Mesh('data/example_off_files/phands.off')

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
            # if not os.path.exists(temp_file):
            #     os.mknod(temp_file)
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
        size_of_black_sphere = 1/3
        # hyper parameter that decide the rate between the longest distance to the sphere to the size of the sphere

        plotter = pv.Plotter(shape=(3, 3))

        self.mesh.plot_vertices(f=lambda a: 0, index_row=0, index_col=0, show=False, plotter=plotter, title="vertices")
        self.mesh.plot_faces(f=lambda a: 0, index_row=0, index_col=1, show=False, plotter=plotter, title="faces")
        self.mesh.plot_wireframe(index_row=0, index_col=2, show=False, plotter=plotter, title="wireframe")

        faces_barycenter = self.mesh.get_face_barycenters()
        normals = self.mesh.get_face_normals()

        plotter.subplot(1, 0)
        plotter.add_text("un-normalized normals", position='upper_edge', font_size=10)
        plotter.add_arrows(faces_barycenter, normals)

        plotter.subplot(1, 1)
        plotter.add_text("normalized normals", position='upper_edge', font_size=10)
        normals = self.mesh.get_face_normals(norm=True)
        plotter.add_arrows(faces_barycenter, normals)

        #  creating (x,y,z) -> id dict
        table = {}
        mean = (0, 0, 0)

        i = 0
        for cord in self.mesh.vertices:
            table[np.array_str(cord)] = i
            i += 1
            mean += cord

        mean = mean / (i + 1)

        self.mesh.plot_vertices(f=lambda a: self.mesh.get_vertex_valence(table[np.array_str(a)]),
                                index_row=1, index_col=2, show=False, plotter=plotter,
                                title='valance figure')

        plotter.subplot(2, 0)
        max_dist = np.apply_along_axis(lambda a: np.linalg.norm(a - mean), 1, self.mesh.vertices).max()
        self.mesh.plot_vertices(f=lambda a: np.linalg.norm(a - mean),
                                #  L2 distance between the mean and the point
                                index_row=2, index_col=0, show=False, plotter=plotter,
                                title="distance from mean")
        plotter.add_mesh(mesh=pv.Sphere(center=mean, radius=size_of_black_sphere * max_dist), color='black')

        plotter.show()


if __name__ == '__main__':
    unittest.main()
