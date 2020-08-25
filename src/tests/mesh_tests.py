import glob
import unittest


class MyTestCase(unittest.TestCase):
    off_files = glob.glob('./data/example_off_files/*.off')

    def test_read_write_off(self):
        """"
            reads all off files, writes them to temp directory and calls diff
        """
        pass

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
        """
        pass
    # def test_something(self):
    #     self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
