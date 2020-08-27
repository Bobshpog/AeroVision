import glob
import unittest

from src.geometry.numpy.mesh import Mesh


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


if __name__ == '__main__':
    unittest.main()
