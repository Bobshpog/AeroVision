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

        self.corners = defaultdict(set)
        for idx, face in enumerate(self.faces):
            for v in face:
                self.corners[v].add(idx)
        self.corners = {k: np.fromiter(v, int, len(v)) for (k, v) in self.corners.items()}

    # ----------------------------Basic Visualizer----------------------------#

    def plot_wireframe(self, index_row=0, index_col=0, show=True, plotter=None):
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

    def plot_vertices(self, f, index_row=0, index_col=0, show=True, plotter=None, cmap='jet'):
        """
            plots the vertices of the Mesh

            Args:
                f: map between (x,y,z) to scalar for the color map
                index_row: chosen subplot row
                index_col: chosen subplot column
                show: should the function call imshow()
                plotter: the pyvista plotter
                cmap: the color map to use

             Returns:
                 the pyvista plotter
             """
        pass

    def plot_faces(self, f, index_row=0, index_col=0, show=True, plotter=None, cmap='jet'):
        """
             plots the faces of the Mesh

             Args:
                  f: map between (x,y,z) to scalar for the color map
                  index_row: chosen subplot row
                  index_col: chosen subplot column
                  show: should the function call imshow()
                  plotter: the pyvista plotter
                  cmap: the color map to use

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
            return self.adj.sum(axis=1)

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
            f = self.faces
            v = self.vertices
            a = v[f[:, 0], :]
            b = v[f[:, 1], :]
            c = v[f[:, 2], :]
            fn = np.cross(b - a, c - a)
            return fn / np.linalg.norm(fn) if norm else fn

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
            v = self.vertices
            f = self.faces
            return v[f.flatten()].reshape((-1, 3, 3)).mean(axis=1)

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
            v1, v2, v3 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
            v1, v2, v3 = self.vertices[v1], self.vertices[v2], self.vertices[v3]
            a = np.linalg.norm(v1 - v2, axis=1)
            b = np.linalg.norm(v1 - v3, axis=1)
            c = np.linalg.norm(v2 - v3, axis=1)
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            return area

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
            neighbours = self.corners[idx]
            areas = np.vectorize(lambda x: self.get_face_areas(x))(neighbours)
            face_norms = np.vectorize(lambda x: self.get_face_normals(x, True), signature='()->(n)')(neighbours)
            vertex_normal = np.sum(face_norms * areas[:, np.newaxis], axis=0)
            return vertex_normal / np.linalg.norm(vertex_normal) if norm else vertex_normal
        else:
            areas_dict = self.get_face_areas()
            face_norms_dict = self.get_face_normals(norm=True)
            areas_data = [areas_dict[item] for i, row in self.corners.items() for j, item in enumerate(row) for _ in range(3)]
            rows = [3*i+k for i, row in self.corners.items() for _ in row for k in range(3)]
            cols = [j for i, row in self.corners.items() for j, item in enumerate(row) for _ in range(3)]
            areas = csr_matrix((areas_data, (rows, cols)))
            face_norms_data = [k for i, row in self.corners.items() for j, item in enumerate(row) for k in face_norms_dict[item]]
            face_norms = csr_matrix((face_norms_data, (rows, cols)))
            vertex_normals=np.array((face_norms.multiply(areas)).sum(axis=1).reshape(-1,3))
            return vertex_normals/np.linalg.norm(vertex_normals,axis=1,keepdims=True) if norm else vertex_normals

            # vector_vertex_normals_func = np.vectorize(lambda x: self.get_vertex_normals(x, norm=norm),
            #                                           signature='()->(n)')
            # return vector_vertex_normals_func(np.arange(0, self.vertices.shape[0]))
