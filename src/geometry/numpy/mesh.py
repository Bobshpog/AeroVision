def read_off(path):
    """
    .off file reader
    Args:
        path: path to .off file
    Returns:
        (vertices, faces):(np.array,np.array)
    """
    pass


def write_off(shape, path):
    """
    .off file writer
    Args:
        shape: (vertices,faces):(np.array,np.array)
        path:  path to .off file

    Returns:
        None
    """
    pass


class Mesh:
    """
    A class representing a Mesh grid
    """

    def __init__(self, path):
        """
        C-tor
        Args:
            path: path to .off file
        """
        data = read_off(path)
        self.vertices = data[0]
        self.faces = data[1]
        self.adj = None

    #### Basic Visualizer ####

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
                  index_row: chosen subplot row
                  index_col: chosen subplot column
                  show: should the function call imshow()
                  plotter: the pyvista plotter
             Returns:
                 the pyvista plotter
        """
        pass

    ##### Basic Properties ####
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
            the wanted normals

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
