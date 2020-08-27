import filecmp
import glob
import unittest

import pyvista as pv

from src.geometry.numpy.mesh import *


class TestMesh(unittest.TestCase):
    def setUp(self):
        off_files = glob.glob('./data/example_off_files/*.off')
        self.mesh = Mesh('/home/alex/PycharmProjects/AeroVision/data/opto_wing.off')

    def test_get_vertex_valence(self):
        self.mesh.get_vertex_valence()

    def test_get_face_normals(self):
        self.mesh.get_face_normals()

    def test_get_face_barycenters(self):
        self.mesh.get_face_barycenters()

    def test_get_face_areas(self):
        self.mesh.get_face_areas()

    def test_get_vertex_normals(self):
        self.mesh.get_vertex_normals()

    def test_read_write_off(self):
        """"
        reads all off files, writes them to temp directory and calls diff

        Args
        files:  iterator over files to write
        """
        for file in TestMesh.off_files:
            data = read_off(file)
            write_off(data, "temp")
            self.assertTrue(filecmp.cmp(file, "temp"))

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
        plotter = pv.Plotter(shape=(3, 3))
        mesh = self.mesh

        mesh.plot_vertices(f=lambda: 0, index_col=0, index_row=0, show=False, plotter=plotter)
        mesh.plot_faces(f=lambda: 0, index_col=0, index_row=1, show=False, plotter=plotter)
        mesh.plot_wireframe(index_col=0, index_row=2, show=False, plotter=plotter)

        plotter.subplot(1, 0)
        faces_barycenter = mesh.get_face_barycenters()
        normals = mesh.get_face_normals()

        pv.plot_arrows(faces_barycenter, normals)

        plotter.subplot(1, 1)
        normals = mesh.get_face_normals(norm=True)
        pv.plot_arrows(faces_barycenter, normals)

        #  creating (x,y,z) -> id dict
        table = {}
        mean = (0, 0, 0)
        for idx, cord in np.ndenumerate(mesh.vertices):
            table[cord] = idx
            mean += cord

        mean = mean / (idx + 1)

        val_func = lambda a: mesh.get_vertex_valence(table[a])

        dist_func = lambda a: np.linalg.norm(a - mean)  # L2 distance between the mean and the point

        mesh.plot_vertices(f=val_func, index_col=1, index_row=2, show=False, plotter=plotter)
        plotter.subplot(2, 0)
        plotter.add_mesh(mesh=pv.Sphere(center=mean), color='black')
        mesh.plot_vertices(f=dist_func, index_col=2, index_row=0, show=False, plotter=plotter)

        plotter.show()


if __name__ == '__main__':
    unittest.main()
