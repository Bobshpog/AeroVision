import numpy as np
from scipy.sparse import csr_matrix


def read_off(path):
    """
    .off file reader

    Args:
        path: path to .off file

    Returns:
        (vertices, faces)
    """
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            if lines[0] != 'OFF\n':
                print(path, 'Error: is not an .off file')
            num_vertices = lines[1].split()[0]
            num_vertices = int(num_vertices)
            vertices = []
            faces = []
            idx = 0
            indexing_vertices = True
            for line in lines[2:]:
                if line[0] == '#':
                    continue
                if indexing_vertices:
                    vertices.append(list(map(float, line.split()[:3])))
                    idx += 1
                    if idx >= num_vertices:
                        indexing_vertices = False
                else:
                    size, *split_line = line.split()
                    size = int(size)
                    split_line = [int(x) for x in split_line[:size]]
                    faces.append(split_line)
            return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)
    except IOError:
        print('Error: Failed reading file:', path)


def write_off(shape, path):
    """
    .off file writer

    Args:
        shape: (vertices, faces)
        path:  path to .off file

    Returns:
        None
    """
    vertices, faces = shape
    try:
        with open(path, 'w') as file:
            file.write('OFF\n')
            file.write(f"{len(vertices)} {len(faces)} 0\n")
            for v in vertices:
                x, y, z = tuple(v)
                file.write(f"{x} {y} {z}\n")
            for face in faces:
                face_str = ' '.join(list(map(str, face)))
                face_str = str(len(face)) + " " + face_str + "\n"
                file.write(face_str)
    except IOError:
        print('Error: Failed reading file:', path)


class Mesh:
    """
    A class representing a Mesh grid
    """

    def __init__(self, path):
        """
        Constructor

        Args:
            path: path to .off file
        """

        data = read_off(path)
        self.vertices = data[0]
        self.faces = data[1]
        adj_set = set()

        for face in self.faces:
            for a, b in zip(face[:-1], face[1:]):
                adj_set.add((min(a, b), max(a, b)))
            adj_set.add((min(face[-1], face[0]), max(face[-1], face[0])))
        adj_np_arr = np.array(list(adj_set))
        values = np.ones(len(adj_np_arr))
        x = adj_np_arr[:, 0]
        y = adj_np_arr[:, 1]
        self.adj = csr_matrix((values, (x, y)))

    # ----------------------------Basic Visualizer----------------------------#

    def plot_wireframe(self, index_row=1, index_col=1, show=True, plotter=None):
        """
       plots the wireframe of the Mesh

       Args:
           index_row: chosen subplot row
           index_col: chosen subplot column
           show: should the function call imshow()
           plotter: the pyvista plotter

        Returns:
            the pyvista plotter
        """
        pass

    def plot_vertices(self, f, index_row=1, index_col=1, show=True, plotter=None):
        """
            plots the vertices of the Mesh

            Args:
                f: map between (x,y,z) to (r,g,b)
                index_row: chosen subplot row
                index_col: chosen subplot column
                show: should the function call imshow()
                plotter: the pyvista plotter

             Returns:
                 the pyvista plotter
             """
        pass

    def plot_faces(self, f, index_row=1, index_col=1, show=True, plotter=None):
        """
             plots the faces of the Mesh

             Args:
                  f: map between (x,y,z) to (r,g,b)
                  index_row: chosen subplot row
                  index_col: chosen subplot column
                  show: should the function call imshow()
                  plotter: the pyvista plotter

             Returns:
                 the pyvista plotter
        """
        pass

    # ----------------------------Basic Properties----------------------------#
    def get_vertex_valence(self, idx=-1):
        """
        Calculates valence for a vertex or for all vertexes if idx<0

        Args:
            idx: The index of the wanted vertex if idx<0 then the entire array is wanted

        Returns:
            The valence of the vertex or of all the vertices if idx<0

        Raises:
            if idx >= len(vertices) raises IndexError
        """
        pass

    def get_face_normals(self, idx=-1, norm=False):
        """
       Calculates normal of a face or of all faces if idx<0

       Args:
            idx: The index of the wanted normal,if idx<0 then the entire array is wanted
            norm: Whether the normal should be normalized

        Returns:
            The wanted normals

        Raises:
        if idx >= len(vertices) raises IndexError
        """
        pass

    def get_face_barycenters(self, idx=-1):
        """
       Calculates barycenters of a face or of all faces if idx<0
        Args:
            idx: The index of the wanted barycenter, if idx<0 then the entire array is wanted

        Returns:
            the wanted barycenters

        Raises:
        if idx >= len(vertices) raises IndexError
        """
        pass

    def get_face_areas(self, idx=-1):
        """
               Calculates area of a face or of all faces if idx<0
                Args:
                    idx: The index of the wanted area, if idx<0 then the entire array is wanted

                Returns:
                    the wanted area

                Raises:
                if idx >= len(vertices) raises IndexError
                """
        pass

    def get_vertex_normals(self, idx=-1, norm=False):
        """
               Calculates normal of a vertex or of all vertices if idx<0
                Args:
                    idx: The index of the wanted normal,if idx<0 then the entire array is wanted
                    norm: Whether the normal should be normalized

                Returns:
                    the wanted normals

                Raises:
                if idx >= len(vertices) raises IndexError
                """
        pass
