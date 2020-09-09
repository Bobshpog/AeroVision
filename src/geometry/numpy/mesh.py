from collections import defaultdict

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA


def read_off(path):
    """
    .off file reader

    Args:
        path: path to .off file

    Returns:
        (vertices, faces, table)
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
            table = {}
            idx = 0
            indexing_vertices = True
            for line in lines[2:]:
                if line[0] == '#':
                    continue
                if indexing_vertices:
                    vertices.append(list(map(float, line.split()[:3])))
                    table[np.asarray(vertices[idx]).tobytes()] = idx
                    idx += 1
                    if idx >= num_vertices:
                        indexing_vertices = False
                else:
                    size, *split_line = line.split()
                    size = int(size)
                    split_line = [int(x) for x in split_line[:size]]
                    faces.append(split_line)
            return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32), table
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
        self.table = data[2]
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

    def plot_wireframe(self, index_row=0, index_col=0, show=True, plotter=None, title='', font_size=10,
                       font_color='black'):
        """
       plots the wireframe of the Mesh

       Args:
           index_row: chosen subplot row
           index_col: chosen subplot column
           show: should the function call imshow()
           plotter: the pyvista plotter
           title: the title of the figure
           font_size: the font size of the title
           font_color: the color of the font for the title

        Returns:
            the pyvista plotter
        """
        if plotter is None:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        pv_styled_faces = np.insert(self.faces, 0, 3, axis=1)
        pv_mesh = pv.PolyData(self.vertices, pv_styled_faces)
        # og = pv_mesh.center
        # og[-1] -= pv_mesh.length / 3.
        # projected = pv_mesh.project_points_to_plane(origin=og, normal=[1, 1, 1])
        # plotter.add_mesh(projected)
        plotter.add_mesh(pv_mesh, style='wireframe')
        if show:
            plotter.show()
        return plotter

    def plot_vertices(self, f=None, index_row=0, index_col=0, show=True, plotter=None, cmap='jet', title='',
                      font_size=10,
                      font_color='black'):
        """
            plots the vertices of the Mesh

            Args:
                f: map between (x,y,z) of vertex to scalar for the color map
                index_row: chosen subplot row
                index_col: chosen subplot column
                show: should the function call imshow()
                plotter: the pyvista plotter
                cmap: the color map to use
                title: the title of the figure
                font_size: the font size of the title
                font_color: the color of the font for the title

             Returns:
                 the pyvista plotter
        """

        if plotter is None:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        plotter.add_mesh(self.vertices, scalars=f, cmap=cmap)
        if show:
            plotter.show()
        return plotter

    def plot_faces(self, f=None, index_row=0, index_col=0, show=True, plotter=None, cmap='jet', title='',
                   font_size=10, font_color='black', texture=None):
        """
             plots the faces of the Mesh

             Args:
                  f: map between (x,y,z) of vertex to scalar for the color map
                  index_row: chosen subplot row
                  index_col: chosen subplot column
                  show: should the function call imshow()
                  plotter: the pyvista plotter
                  cmap: the color map to use
                  title: the title of the figure
                  font_size: the font size of the title
                  font_color: the color of the font for the title
                  texture: the filename for the texture of the figure

             Returns:
                 the pyvista plotter
        """
        if plotter is None:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        pv_styled_faces = np.insert(self.faces, 0, 3, axis=1)
        pv_mesh = pv.PolyData(self.vertices, pv_styled_faces)
        if texture is None:
            plotter.add_mesh(pv_mesh, scalars=f, cmap=cmap, texture=texture)
        else:
            tex = pv.read_texture(texture)
            pv_mesh.texture_map_to_plane(inplace=True)
            plotter.add_mesh(pv_mesh, texture=tex)
        if show:
            plotter.show()
        return plotter

    # ----------------------------Advanced Visualizer----------------------------#

    def connected_component(self, plot=False, index_row=0, index_col=0, show=True, plotter=None, cmap='jet', title='',
                            font_size=10, font_color='black'):
        """
             giving the connected components of the mesh

             Args:
                  plot: does the algorithm need to plot to plotter (will be more efficient with plot=false)
                  index_row: chosen subplot row
                  index_col: chosen subplot column
                  show: should the function call imshow()
                  plotter: the pyvista plotter
                  cmap: the color map to use
                  title: the title of the figure
                  font_size: the font size of the title
                  font_color: the color of the font for the title

             Returns:
                 (number of connected components, label for each vertex):(int, np.array)
        """
        cc_num, labels = connected_components(csgraph=self.adj, directed=False, return_labels=True)
        if not plot:
            return cc_num, labels
        # f=lambda a: labels[self.table[a.tobytes()]]
        self.plot_faces(f=labels, index_row=index_row, index_col=index_col,
                        show=show,
                        plotter=plotter, cmap=cmap, title=title, font_size=font_size, font_color=font_color)
        return cc_num, labels

    def main_cords(self, num_of_cords=3, scale=100, plot=False, index_row=0, index_col=0,
                   show=True, plotter=None, title='', font_size=10, font_color='black'):
        """
             returning the cords in which there are most variance

             Args:
                  num_of_cords: number of coordinates needed, maximum 3
                  scale: the scale which we would scale the cord vectors
                  plot: does the algorithm need to plot to plotter (will be more efficient with plot=false)
                  index_row: chosen subplot row
                  index_col: chosen subplot column
                  show: should the function call imshow()
                  plotter: the pyvista plotter
                  title: the title of the figure
                  font_size: the font size of the title
                  font_color: the color of the font for the title

             Returns:
                 the main cords of the mash, if plot then red is the main one, green is the second and blue is the third
        """
        pca = PCA(n_components=num_of_cords)
        pca.fit(self.vertices)
        if not plot:
            return pca.components_ * scale
        mean = self.vertices.mean(axis=0)
        if not plotter:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        plotter.add_arrows(mean, scale * pca.components_[0], color='red')
        if num_of_cords > 1:
            plotter.add_arrows(mean, scale * pca.components_[1], color='green')
        if num_of_cords > 2:
            plotter.add_arrows(mean, scale * pca.components_[2], color='blue')
        if show:
            plotter.show()
        return pca.components_ * scale

    def plot_projection(self, normal=(1, 1, 1), index_row=0, index_col=0, show=True,
                        plotter=None, title='', font_size=10, font_color='black'):
        """
       plots the projection of the Mesh

       Args:
           normal: the normal of the projected image
           index_row: chosen subplot row
           index_col: chosen subplot column
           show: should the function call imshow()
           plotter: the pyvista plotter
           title: the title of the figure
           font_size: the font size of the title
           font_color: the color of the font for the title

        Returns:
            the pyvista plotter
        """
        if plotter is None:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)

        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        pv_styled_faces = np.insert(self.faces, 0, 3, axis=1)
        pv_mesh = pv.PolyData(self.vertices, pv_styled_faces)
        plotter.set_viewup([0, 0, 0])
        og = pv_mesh.center
        print(og)

        main_c = self.main_cords(plot=False)
        pos = plotter.camera_position
        projected = pv_mesh.project_points_to_plane(origin=og, normal=normal)
        #print(plotter.camera_position)
        plotter.add_mesh(projected)
        if show:
            plotter.show()
        return plotter

    def animate(self, movement, f=None, index_col=0, index_row=0,  texture=None, cmap='jet',
                plotter=None, title='', font_size=10, font_color='black', gif_path=None):
        """
       animate the mash using f as movment metrix

       Args:
           movement: iterable with V side vector as elements
           f: map between (x,y,z) of vertex to scalar for the color map
           index_row: chosen subplot row
           index_col: chosen subplot column
           show: should the function call imshow()
           texture: the texture to use
           cmap: the colormap to use
           plotter: the pyvista plotter
           title: the title of the figure
           font_size: the font size of the title
           font_color: the color of the font for the title
           gif_path: gif path to create, None if no gif is needed


        Returns:
           the pyvista plotter
        """

        if plotter is None:
            plotter = pv.Plotter()

        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        pv_styled_faces = np.insert(self.faces, 0, 3, axis=1)
        pv_mesh = pv.PolyData(self.vertices, pv_styled_faces)
        if texture is None:
            plotter.add_mesh(pv_mesh, scalars=f, cmap=cmap, texture=texture)
        else:
            tex = pv.read_texture(texture)
            pv_mesh.texture_map_to_plane(inplace=True)
            plotter.add_mesh(pv_mesh, texture=tex)
        plotter.show(auto_close=False)
        if gif_path is not None:
            plotter.open_gif(gif_path)
        for item in movement:
            plotter.update_coordinates(item, mesh=pv_mesh)
            if gif_path is not None:
                plotter.write_frame()

        plotter.close()

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
            fn = self.get_face_normals(norm=False)
            matrix = self._get_vertex_face_adjacency()
            vertex_normal = matrix.dot(fn)
            return vertex_normal / np.linalg.norm(vertex_normal) if norm else vertex_normal

    def _get_vertex_face_adjacency(self, data=None):
        """
        Return a sparse matrix for which vertices are contained in which faces.
        A data vector can be passed which is then used instead of booleans
        """
        # Input checks:
        nv = self.vertices.shape[0]
        f = self.faces  # Convert to an ndarray or pass if already is one

        # Computation
        row = f.reshape(-1)  # Flatten indices
        col = np.tile(np.arange(len(f)).reshape((-1, 1)), (1, f.shape[1])).reshape(-1)  # Data for vertices
        shape = (nv, len(f))

        if data is None:
            data = np.ones(len(col), dtype=np.bool)

        # assemble into sparse matrix
        return coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)


def animate_few_meshes(mesh, movement, f=None, num_of_plots=1, subplot=(0, 0),  texture=None, cmap='jet',
                       plotter=None, title='', font_size=10, font_color='black', gif_path=None):
    if num_of_plots == 1:
        return mesh.animate(movement=movement,f=f,index_col=subplot[1], index_row=subplot[0], texture=texture,
                            cmap=cmap, plotter=plotter, title=title, font_color=font_color, font_size=font_size,
                            gif_path=gif_path)

    if plotter is None:
        plotter = pv.Plotter()
    pv_mesh = []
    for idx, plot in enumerate(subplot):
        plotter.subplot(plot[0],plot[1])
        plotter.add_text(title[idx], position="upper_edge", font_size=font_size[idx], color=font_color[idx])
        pv_styled_faces = np.insert(mesh[idx].faces, 0, 3, axis=1)
        pv_mesh.append(pv.PolyData(mesh[idx].vertices, pv_styled_faces))
        if texture[idx] is None:
            plotter.add_mesh(pv_mesh[idx], scalars=f[idx], cmap=cmap[idx], texture=texture[idx])
        else:
            tex = pv.read_texture(texture[idx])
            pv_mesh[idx].texture_map_to_plane(inplace=True)
            plotter.add_mesh(pv_mesh[idx], texture=tex)
    plotter.show(auto_close=False)
    if gif_path is not None:
        plotter.open_gif(gif_path)
    for frame, item in enumerate(movement[0]):
        for plt_id, plot in enumerate(subplot):
            plotter.update_coordinates(movement[plt_id][frame], mesh=pv_mesh[plt_id])
        if gif_path is not None:
            plotter.write_frame()

    plotter.close()
    pass
