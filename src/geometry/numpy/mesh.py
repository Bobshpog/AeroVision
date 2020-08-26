from collections import defaultdict

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
        x = adj_np_arr[:, 0]
        y = adj_np_arr[:, 1]
        values = np.ones(len(x))
        self.adj = csr_matrix((values, (x, y)), shape=(len(self.vertices), len(self.vertices)))

        self.corners=defaultdict(set)
        for idx,face in enumerate(self.faces):
            for v in face:
                self.corners[v].add(idx)


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
        if idx >= len(self.vertices):
            raise IndexError
        if idx >= 0:
            return np.sum(self.adj[:, idx]) + np.sum(self.adj[idx, :]) - self.adj[idx, idx]
        else:
            vector_valance_func = np.vectorize(lambda x: self.get_vertex_valence(x), otypes=[np.int32])
            return vector_valance_func(np.arange(0, len(self.vertices)))

    def get_face_normals(self, idx=-1, norm=False):
        """
       Calculates normal of a face or of all faces if idx<0

        Args:
            idx: The index of the wanted normal,if idx<0 then the entire array is wanted
            norm: Whether the normal should be normalized

        Returns:
            The wanted normals

        Raises:
            If idx >= len(vertices) raises IndexError

        Warning:
            Only works on triangular faces
        """
        if idx >= len(self.faces):
            raise IndexError
        if idx >= 0:
            v1, v2, v3 = self.faces[idx]
            v1, v2, v3 = self.vertices[v1], self.vertices[v2], self.vertices[v3]
            e1 = v2 - v1
            e2 = v3 - v1
            cross = np.cross(e1, e2)
            return cross / np.linalg.norm(cross) if norm else cross
        else:
            vector_face_normals_func = np.vectorize(lambda x: self.get_face_normals(x, norm=norm),
                                                    signature='()->(n)')
            return vector_face_normals_func(np.arange(0, self.faces.shape[0]))

    def get_face_barycenters(self, idx=-1):
        """
       Calculates barycenters of a face or of all faces if idx<0
        Args:
            idx: The index of the wanted barycenter, if idx<0 then the entire array is wanted

        Returns:
            the wanted barycenters

        Raises:
            if idx >= len(vertices) raises IndexError

        Warning:
            Only works on triangular faces
             """
        if idx >= len(self.faces):
            raise IndexError
        if idx >= 0:
            v = np.vectorize(lambda x: self.vertices[x], signature='()->(n)')(self.faces[idx])
            return np.mean(v, axis=0)
        else:
            vector_face_centers_func = np.vectorize(lambda x: self.get_face_barycenters(x),
                                                    signature='()->(n)')
            return vector_face_centers_func(np.arange(0, self.faces.shape[0]))

    def get_face_areas(self, idx=-1):
        """
        Calculates area of a face or of all faces if idx<0

            Args:
                 idx: The index of the wanted area, if idx<0 then the entire array is wanted

            Returns:
                 the wanted area

            Raises:
                 if idx >= len(vertices) raises IndexError
            Warning:
                 Only works on triangular faces
                """
        if idx >= len(self.faces):
            raise IndexError
        if idx >= 0:
            v1, v2, v3 = self.faces[idx]
            v1, v2, v3 = self.vertices[v1], self.vertices[v2], self.vertices[v3]
            a = np.linalg.norm(v1 - v2)
            b = np.linalg.norm(v1 - v3)
            c = np.linalg.norm(v2 - v3)
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            return area
        else:
            vector_face_area_func = np.vectorize(lambda x: self.get_face_areas(x))
            return vector_face_area_func(np.arange(0, self.faces.shape[0]))

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

                Warning:
                    only works on triangular faces
                """
        if idx >= len(self.faces):
            raise IndexError
        if idx >= 0:
            neighbours=np.array(list(self.corners[idx]))
            areas=np.vectorize(lambda x:self.get_face_areas(x))(neighbours)
            face_norms=np.vectorize(lambda x:self.get_face_normals(x,True),signature='()->(n)')(neighbours)
            #TODO: fix einsum formula
            vertex_normal=np.einsum('j,ij->ij',areas[0],face_norms)
            return vertex_normal
        else:
            vector_face_normals_func = np.vectorize(lambda x: self.get_face_normals(x, norm=norm),
                                                    signature='()->(n)')
            return vector_face_normals_func(np.arange(0, self.faces.shape[0]))

# mesh = Mesh('/home/alex/PycharmProjects/AeroVision/data/example_off_files/cat.off')
# print(mesh.get_vertex_normals(0))
# temp = mesh.get_face_areas()
# print(temp)
