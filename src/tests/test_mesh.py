
import glob
import unittest
import os
import pyvista as pv
from src.util.profiler import profile

from src.geometry.numpy.mesh import *


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.off_files = glob.glob('data/example_off_files/*.off')
        self.mesh = Mesh('data/example_off_files/cat.off')

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
            self.assertTrue(np.array_equal(data[0],mesh_new[0]) and np.array_equal(data[1],mesh_new[1]))
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
        plotter = pv.Plotter(shape=(3, 3))
        mesh = self.mesh

        mesh.plot_vertices(f=lambda a: 0, index_row=0, index_col=0, show=False, plotter=plotter)
        mesh.plot_faces(f=lambda a: 0, index_row=0, index_col=1, show=False, plotter=plotter)
        mesh.plot_wireframe(index_row=0, index_col=2, show=False, plotter=plotter)

        faces_barycenter = mesh.get_face_barycenters()
        normals = mesh.get_face_normals()

        plotter.subplot(1, 0)
        plotter.add_arrows(faces_barycenter, normals)

        plotter.subplot(1, 1)
        normals = mesh.get_face_normals(norm=True)
        plotter.add_arrows(faces_barycenter, normals)

        #  creating (x,y,z) -> id dict
        table = {}
        mean = (0, 0, 0)

        i = 0
        for cord in mesh.vertices:
            table[np.array_str(cord)] = i
            i += 1
            mean += cord

        mean = mean / (i + 1)

        val_func = lambda a: mesh.get_vertex_valence(table[np.array_str(a)])

        dist_func = lambda a: np.linalg.norm(a - mean)  # L2 distance between the mean and the point

        mesh.plot_vertices(f=val_func,index_row=1, index_col=2, show=False, plotter=plotter)
        plotter.subplot(2, 0)

        mesh.plot_vertices(f=dist_func, index_row=2, index_col=0, show=False, plotter=plotter)
        plotter.add_mesh(mesh=pv.Sphere(center=mean, radius=0.2), color='black')
        plotter.show()


if __name__ == '__main__':
    unittest.main()
