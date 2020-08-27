import glob
import unittest
import filecmp
from symbol import lambdef

import pyvista as pv

from src.geometry.numpy.mesh import *


class MyTestCase(unittest.TestCase):
    off_files = glob.glob('./data/example_off_files/*.off')

    def test_read_write_off(self, files):
        """"
        reads all off files, writes them to temp directory and calls diff

         Args
            files:  iterator over files to write
        """
        passed = True
        for file in files:
            data = read_off(file)
            write_off(data, "temp")
            if not filecmp.cmp(file,"temp"):
                print("FAILED on file: {}".format(file))
                passed = False

        if passed:
            print("PASSED ON ALL FILES")

    def test_visualization(self, path):
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
        mesh = Mesh(path)

        mesh.plot_vertices(f=lambda: 0, index_col=0, index_row=0, show=False, plotter=plotter)
        mesh.plot_faces(f=lambda: 0, index_col=0, index_row=1, show=False, plotter=plotter)
        mesh.plot_wireframe(index_col=0, index_row=2, show=False, plotter=plotter)

        plotter.subplot(1,0)
        faces_barycenter = mesh.get_face_barycenters()
        normals = mesh.get_face_normals()

        pv.plot_arrows(faces_barycenter, normals)

        plotter.subplot(1, 1)
        normals = mesh.get_face_normals(norm=True)
        pv.plot_arrows(faces_barycenter, normals)

        #  creating (x,y,z) -> id dict
        table = {}
        mean = (0,0,0)
        for idx, cord in np.ndenumerate(mesh.vertices):
            table[cord] = idx
            mean += cord

        mean = mean/(idx+1)

        val_func = lambda a:mesh.get_vertex_valence(table[a])
        dist_func = lambda a:np.linalg.norm(a-mean) # L2 distance between the mean and the point

        mesh.plot_vertices(f=val_func, index_col=1, index_row=2, show=False, plotter=plotter)
        mesh.plot_vertices(f=dist_func, index_col=2, index_row=0,show=False, plotter=plotter)

        plotter.show()


    # def test_something(self):
    #     self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

