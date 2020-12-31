from collections import defaultdict
from itertools import cycle
from time import sleep
import random
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def new_round(x,y):
    return int(x*(10**y))

def cord2index(cord,digits=4):
    #TODO Doesnt work at all with any number of digits
    return round(cord[0],digits),round(cord[1],digits),round(cord[2],digits)


def read_off_size(path):
    """
    Finds the number of vertices and faces in file
    Args:
        path: path to off file

    Returns:
        (num of vertices, num of faces)
    """
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            if lines[0] != 'OFF\n':
                print(path, 'Error: is not an .off file')
            num_vertices, num_faces = tuple(lines[1].split()[:2])
            return int(num_vertices), int(num_faces)
    except IOError:
        print('Error: Failed reading file:', path)
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
                    table[cord2index(vertices[idx])] = idx
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

    def __init__(self, path, texture=None):
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
        self.pv_mesh = None

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

        pv_styled_faces = np.insert(self.faces, 0, 3, axis=1)
        # mesh in pyvista format
        self.pv_mesh = pv.PolyData(self.vertices, pv_styled_faces)
        if texture:
            self.pv_mesh.texture_map_to_plane(inplace=True, use_bounds=1)
            self.texture = pv.read_texture(texture)
        else:
            self.texture = None

    # ----------------------------Basic Visualizer----------------------------#

    def plot_wireframe(self, line_width=None, index_row=0, index_col=0, show=True, plotter=None, title='', font_size=10,
                       title_location="upper_edge", font_color='black', camera=None):
        """
       plots the wireframe of the Mesh

       Args:
           line_width: width of the lines
           index_row: chosen subplot row
           index_col: chosen subplot column
           show: should the function call imshow()
           plotter: the pyvista plotter
           title: the title of the figure
           title_location: title location
           font_size: the font size of the title
           font_color: the color of the font for the title
           camera: the [camera position , focal point, view up] each (x,y,z) tuple

        Returns:
            the pyvista plotter
        """
        if not plotter:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position=title_location, font_size=font_size, color=font_color)
        if camera:
            plotter.set_position(camera[0])
            plotter.set_focus(camera[1])
            plotter.set_viewup(camera[2])
        plotter.add_mesh(self.pv_mesh, style='wireframe', line_width=line_width, show_scalar_bar=False, color="white")
        if show:
            plotter.show()
        return plotter

    def plot_vertices(self, f=None, index_row=0, index_col=0, show=True, plotter=None, cmap='jet', title='',
                      title_location="upper_edge", font_size=10, font_color='black', camera=None):
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
                title_location: title location
                font_size: the font size of the title
                font_color: the color of the font for the title
                camera: the [camera position , focal point, view up] each (x,y,z) tuple

             Returns:
                 the pyvista plotter
        """

        if not plotter:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position=title_location, font_size=font_size, color=font_color)
        if camera:
            plotter.set_position(camera[0])
            plotter.set_focus(camera[1])
            plotter.set_viewup(camera[2])
        plotter.add_mesh(self.vertices, scalars=f, cmap=cmap, render_points_as_spheres=True)
        if show:
            plotter.show()
        return plotter

    def _find_normal_plane_to_camera(self, camera, height, width):

        A = np.zeros((4, 4), dtype=np.float32)

        center = np.ones(4)
        center[:3] = np.array(camera[0], dtype=np.float)
        focal = np.ones(4)
        focal[:3] = np.array(camera[1], dtype=np.float)
        b1 = np.zeros((4,), dtype=np.float32)
        b1[:3] = camera[2]

        plane_normal = focal - center
        # plane_normal = plane_normal / np.linalg.norm(plane_normal)
        b2 = np.zeros(4)
        b2[:3] = np.cross(b1[:3], plane_normal[:3])
        A[:3, 1] = b1[:3]
        A[:3, 2] = plane_normal[:3]
        A[:3, 3] = center[:3]
        A[:3, 0] = b2[:3]
        A[3, 3] = 1

        # A = np.array([
        #     [9.99963954e-01, -8.49057398e-03, 0, -4.74510255e-02],
        #     [1.61827673e-04, 1.90589989e-02, 9.99818348e-01, -2.57221580e-02],
        #     [-8.48903165e-03, -9.99782309e-01, 1.90596859e-02, -5.34192476e-02]
        #     , [0, 0, 0, 1]])
        A = np.eye(4)
        A_inv = np.linalg.inv(A)

        # center2=focal
        origin = center - height / 2 * b1 + width / 2 * b2  # bottom left
        point_u = center - height / 2 * b1 - width / 2 * b2  # bottom right
        point_v = center + height / 2 * b1 + width / 2 * b2  # top left
        b = self.pv_mesh.GetBounds()
        # origin[:3] = [b[0], b[2], b[4]]  # BOTTOM LEFT CORNER
        # point_u[:3] = [b[1], b[2], b[4]]  # BOTTOM RIGHT CORNER
        # point_v[:3] = [b[0], b[3], b[4]]  # TOP LEFT CORNER
        origin = A_inv @ origin
        point_u = A_inv @ point_u
        point_v = A_inv @ point_v
        origin = origin / origin[-1]
        point_u = point_u / point_u[-1]
        point_v = point_v / point_v[-1]
        return origin[:3], point_u[:3], point_v[:3]

    def plot_faces(self, f=None, index_row=0, index_col=0, show=True, plotter=None, cmap='jet', title=None,
                   title_location="upper_edge", font_size=10, font_color='black', texture=None, camera=None):
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
                  title_location: title location
                  font_size: the font size of the title
                  font_color: the color of the font for the title
                  texture: the filename for the texture of the figure or np array
                  camera: the [camera position , focal point, view up] each (x,y,z) tuple

             Returns:
                 the pyvista plotter
        """
        if not plotter:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)
        if title:
            plotter.add_text(title, position=title_location, font_size=font_size, color=font_color)
        if camera:
            plotter.set_position(camera[0])
            plotter.set_focus(camera[1])
            plotter.set_viewup(camera[2])
        if self.texture:
            plotter.add_mesh(self.pv_mesh, texture=self.texture)
        elif texture is None:
            plotter.add_mesh(self.pv_mesh, scalars=f, cmap=cmap, texture=texture, show_scalar_bar=False)
        else:
            if isinstance(texture, np.ndarray):
                tex = pv.numpy_to_texture(texture)
            else:
                tex = pv.read_texture(texture)
            self.pv_mesh.texture_map_to_plane(inplace=True)
            plotter.add_mesh(self.pv_mesh, texture=tex)
        if show:
            plotter.show()
        return plotter

    # ----------------------------Advanced Visualizer----------------------------#

    def connected_component(self, plot=False, index_row=0, index_col=0, show=True, plotter=None, cmap='jet', title='',
                            font_size=10, font_color='black', camera=None):
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
                  camera: the [camera position , focal point, view up] each (x,y,z) tuple

             Returns:
                 (number of connected components, label for each vertex):(int, np.array)
        """
        cc_num, labels = connected_components(csgraph=self.adj, directed=False, return_labels=True)
        if not plot:
            return cc_num, labels
        self.plot_faces(f=labels, index_row=index_row, index_col=index_col,
                        show=show, plotter=plotter, cmap=cmap, title=title,
                        font_size=font_size, font_color=font_color, camera=camera)
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

    def plot_projection(self, normal=(1, 1, 1), index_row=0, index_col=0, show=True, texture=None, cmap="jet", f=None,
                        plotter=None, title='', font_size=10, font_color='black'):
        """
       plots the projection of the Mesh

       Args:
           normal: the normal of the projected image
           index_row: chosen subplot row
           index_col: chosen subplot column
           show: should the function call imshow()
           texture: the texture to use
           cmap: the color map to use
           f: the color for each vertex or face.
           plotter: the pyvista plotter
           title: the title of the figure
           font_size: the font size of the title
           font_color: the color of the font for the title

        Returns:
            the pyvista plotter
        """
        if not plotter:
            plotter = pv.Plotter()
        plotter.subplot(index_column=index_col, index_row=index_row)

        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        tex = None
        if texture:

            if isinstance(texture, np.ndarray):
                tex = pv.numpy_to_texture(texture)
            else:
                tex = pv.read_texture(texture)
            self.pv_mesh.texture_map_to_plane(inplace=True)
            # plotter.add_mesh(pv_mesh, texture=tex)

        og = self.pv_mesh.center
        projected = self.pv_mesh.project_points_to_plane(origin=og, normal=normal)
        projected.texture_map_to_plane()
        plotter.add_mesh(projected, texture=tex)
        if show:
            plotter.show()
        return plotter

    def animate(self, movement, f=None, index_col=0, index_row=0, texture=None, cmap='jet',
                plotter=None, title='', font_size=10, font_color='black', gif_path=None, camera=None):
        """
       animate the mash using movement as movement metrix press "q" after adjusting the frame to start the animation

       Args:
           movement: tuple of  V side vector as elements, (hull_movement,tip_movement)
           f: map between (x,y,z) of vertex to scalar for the color map
           index_row: chosen subplot row
           index_col: chosen subplot column
           texture: the texture to use
           cmap: the colormap to use
           plotter: the pyvista plotter
           title: the title of the figure
           font_size: the font size of the title
           font_color: the color of the font for the title
           gif_path: gif path to create, None if no gif is needed
           camera: the [camera position , focal point, view up] each (x,y,z) tuple


        Returns:
           None
        """

        if not plotter:
            plotter = pv.Plotter()

        plotter.subplot(index_column=index_col, index_row=index_row)
        plotter.add_text(title, position="upper_edge", font_size=font_size, color=font_color)
        if camera:
            plotter.set_position(camera[0])
            plotter.set_focus(camera[1])
            plotter.set_viewup(camera[2])
        if not texture:
            plotter.add_mesh(self.pv_mesh, scalars=f, cmap=cmap, texture=texture)
        else:
            if isinstance(texture, np.ndarray):
                tex = pv.numpy_to_texture(texture)
            else:
                tex = pv.read_texture(texture)
            self.pv_mesh.texture_map_to_plane(inplace=True)
            plotter.add_mesh(self.pv_mesh, texture=tex)
        plotter.show(auto_close=False)
        if gif_path:
            plotter.open_gif(gif_path)
        for item in movement:
            plotter.update_coordinates(item, mesh=self.pv_mesh)
            if gif_path:
                plotter.write_frame()

        plotter.close()

    @staticmethod
    def get_photo(mesh, movement, resolution, cmap, plotter, camera, title=None, title_location="upper_edge",
                  background_photos=None,cam_noise_lambda=None, background_scale=1, title_color="black"):
        """
        Take a photo of the mesh in a certain position
        all args in case for more then one mesh should be in list

       Args:
           mesh: the mesh to use
           movement: V side vector
           texture: the texture to use
           cmap: the colormap to use, used only if texture is not supplied
           plotter: the pyvista plotter, clear the mesh "get_photo" in the plotter
           camera: the [camera position , focal point, view up] each (x,y,z) tuple
           resolution: the image resolution [w,h]
           background_photos: list of background photos to use in random


        Returns:
           An image shot from camera of the mesh
        """
        return Mesh.get_many_photos(mesh, movement, resolution, cmap,
                                    plotter, [camera], title, title_location, background_photos=background_photos,cam_noise_lambda=cam_noise_lambda,
                                    background_scale=background_scale, title_color=title_color)[0]

    @staticmethod
    def get_many_photos(mesh, movement, resolution, cmap, plotter, camera, title=None, title_location="upper_edge",
                        background_photos=None, background_scale=1, title_color="black", cam_noise_lambda=None):
        """
        Take a photo of the mesh in a cerain position
        all args in case for more then one mesh should be in list

       Args:
           mesh: the mesh to use
           movement: V side vector
           texture: the texture to use
           cmap: the colormap to use, used only if texture is not supplied
           plotter: the pyvista plotter, clear the mesh "get_photo" in the plotter
           camera: list of [camera position , focal point, view up] each (x,y,z) tuple
           resolution: the image resolution [w,h]
           background_photos: list of background photos to use in random


        Returns:
           An image shot from camera of the mesh
        """
        to_return = np.zeros(shape=(len(camera), resolution[1], resolution[0], 4))
        num_of_mesh = len(mesh)
        if background_photos:
            plotter.add_background_image(random.choice(background_photos), scale=background_scale)
        if cam_noise_lambda:
            cam_noise = np.zeros((len(camera), 3, 3))
            cam_noise[:,0] += np.random.normal(0, cam_noise_lambda[0], (len(camera), 3))
            cam_noise[:,1] += np.random.normal(0, cam_noise_lambda[1], (len(camera), 3))
            cam_noise[:,2] += np.random.normal(0, cam_noise_lambda[2], (len(camera), 3))
            camera = np.array(camera) + cam_noise

        if num_of_mesh == 1:
            mesh = [mesh]
        for i in range(num_of_mesh):
            if not mesh[i].texture:
                plotter.add_mesh(mesh[i].pv_mesh, cmap=cmap,
                                 name='get_photo_' + str(i))
            else:
                plotter.add_mesh(mesh[i].pv_mesh, texture=mesh[i].texture, name='get_photo_mesh_' + str(i))
            plotter.update_coordinates(movement[i], mesh=mesh[i].pv_mesh)
        if title:
            plotter.add_text(title, position=title_location, font_size=10, color=title_color, name="title", shadow=True)
        plotter.set_background(color="white")
        plotter.show(auto_close=False, window_size=resolution)
        for idx, cam in enumerate(camera):
            plotter.set_position(cam[0])
            plotter.set_focus(cam[1])
            plotter.set_viewup(cam[2])
            depth = plotter.get_image_depth(fill_value=None)
            depth = np.abs(depth)
            screen = plotter.screenshot(window_size=resolution)
            screen = screen / 255
            to_return[idx] = np.append(screen, depth.reshape(resolution[1], resolution[0], 1), axis=-1)
        if background_photos:
            plotter.remove_background_image()
        return np.asarray(to_return, np.float32)

    @staticmethod
    def get_many_noisy_photos(mesh, movement, resolution, cmap, plotter, camera, title=None, title_location="upper_edge",
                              background_photos=None, background_scale=1, title_color="black", cam_noise_lambda=None,
                              texture_params=(255/2, 155, (1000, 1000, 3))):
        """
        Take a photo of the mesh in a cerain position WITH GAUSSIAN TEXTURE
        all args in case for more then one mesh should be in list

       Args:
           mesh: the mesh to use
           movement: V side vector
           texture: the texture to use
           cmap: the colormap to use, used only if texture is not supplied
           plotter: the pyvista plotter, clear the mesh "get_photo" in the plotter
           camera: list of [camera position , focal point, view up] each (x,y,z) tuple
           resolution: the image resolution [w,h]
           background_photos: list of background photos to use in random
           texture_params: (mean, variance, shape) of the noisy texture


        Returns:
           An image shot from camera of the mesh
        """
        to_return = np.zeros(shape=(len(camera), resolution[1], resolution[0], 4))
        num_of_mesh = len(mesh)
        if background_photos:
            plotter.add_background_image(random.choice(background_photos), scale=background_scale)
        if cam_noise_lambda:
            cam_noise = np.zeros((len(camera), 3, 3))
            cam_noise[:,0] += np.random.normal(0, cam_noise_lambda[0], (len(camera), 3))
            cam_noise[:,1] += np.random.normal(0, cam_noise_lambda[1], (len(camera), 3))
            cam_noise[:,2] += np.random.normal(0, cam_noise_lambda[2], (len(camera), 3))
            camera = np.array(camera) + cam_noise

        if num_of_mesh == 1:
            mesh = [mesh]
        for i in range(num_of_mesh):
            tex = np.random.normal(texture_params[0], texture_params[1], texture_params[2]).astype(np.uint8)
            tex[np.where(tex > 255)] = 255
            tex[np.where(tex < 0)] = 0
            tex = pv.numpy_to_texture(tex)
            mesh[i].pv_mesh.texture_map_to_plane(inplace=True)
            plotter.add_mesh(mesh[i].pv_mesh, texture=tex, name='get_photo_mesh_' + str(i))
            plotter.update_coordinates(movement[i], mesh=mesh[i].pv_mesh)
        if title:
            plotter.add_text(title, position=title_location, font_size=10, color=title_color, name="title", shadow=True)
        plotter.set_background(color="white")
        plotter.show(auto_close=False, window_size=resolution)
        for idx, cam in enumerate(camera):
            plotter.set_position(cam[0])
            plotter.set_focus(cam[1])
            plotter.set_viewup(cam[2])
            depth = plotter.get_image_depth(fill_value=None)
            depth = np.abs(depth)
            screen = plotter.screenshot(window_size=resolution)
            screen = screen / 255
            to_return[idx] = np.append(screen, depth.reshape(resolution[1], resolution[0], 1), axis=-1)
        if background_photos:
            plotter.remove_background_image()
        return np.asarray(to_return, np.float32)

    # ----------------------------Basic Properties----------------------------#
    def __len__(self):
        return 1

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

        if not data:
            data = np.ones(len(col), dtype=np.bool)

        # assemble into sparse matrix
        return coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)


def animate_few_meshes(mesh, movement, f=None, subplot=(0, 0), texture=None, cmap='jet',
                       plotter=None, title='', font_size=10, font_color='black', gif_path=None, camera=None,
                       depth=False):
    """
   animate few mashes using f as movment metrix. press "q" after adjusting the frame to start the animation

   Args:
       mesh: list of the meshes to plot
       movement:  list of iterable with Vn side vector as elements for the num of vercies of the n-th mesh
       f: list of function that map between id of vertex to scalar for the color map
       subplot: list of subplots to use, each is a tuple in the form of: (row,col)
       texture: list of the textures to use
       cmap: list of the colormap to use
       plotter: the pyvista plotter
       title: the title of the figure
       font_size: the font size of the title
       font_color: the color of the font for the title
       gif_path: gif path to create, None if no gif is needed
       camera: list of the [camera position , focal point, view up] each (x,y,z) tuple


    Returns:
       None
    """
    num_of_plots = len(mesh)
    if num_of_plots == 1:
        return mesh.animate(movement=movement, f=f, index_col=subplot[1], index_row=subplot[0], texture=texture,
                            cmap=cmap, plotter=plotter, title=title, font_color=font_color, font_size=font_size,
                            gif_path=gif_path, camera=camera)

    if not plotter:
        plotter = pv.Plotter()
    pv_mesh = []
    # adding mushes with textures

    for idx in range(num_of_plots):
        plotter.subplot(subplot[idx][0], subplot[idx][1])
        if depth:
            plotter.enable_depth_peeling(0)
        plotter.add_text(title[idx], position="upper_edge", font_size=font_size[idx], color=font_color[idx])
        if camera[idx]:
            plotter.set_position(camera[idx][0])
            plotter.set_focus(camera[idx][1])
            plotter.set_viewup(camera[idx][2])
        pv_mesh.append(mesh[idx].pv_mesh)
        if not texture[idx]:
            plotter.add_mesh(pv_mesh[idx], scalars=f[idx], cmap=cmap[idx], texture=texture[idx])
        else:
            if isinstance(texture[idx], np.ndarray):
                tex = pv.numpy_to_texture(texture[idx])
            else:
                tex = pv.read_texture(texture[idx])
            pv_mesh[idx].texture_map_to_plane(inplace=True)
            plotter.add_mesh(pv_mesh[idx], texture=tex)
    # starting the animation
    plotter.show(auto_close=False)
    if gif_path:
        plotter.open_gif(gif_path)
    for frame, item in enumerate(movement[0]):
        for plt_id in range(num_of_plots):
            plotter.update_coordinates(movement[plt_id][frame], mesh=pv_mesh[plt_id])
        if gif_path:
            plotter.write_frame()

    plotter.close()


def draw_chessboard(n=8, pixel_width=500):
    """
    Draw an n x n chessboard using PIL.
    """

    def sq_start(i):
        """
        Return the square corners, suitable for use in PIL drawings
        """
        return i * pixel_width / n

    def square(i, j):
        """
        Return the square corners, suitable for use in PIL drawing
        """
        return map(sq_start, [i, j, i + 1, j + 1])

    image = Image.new("L", (pixel_width, pixel_width))
    draw_square = ImageDraw.Draw(image).rectangle
    squares = (square(i, j)
               for i_start, j in zip(cycle((0, 1)), range(n))
               for i in range(i_start, n, 2))

    for sq in squares:
        draw_square(sq, fill='white')
        image.save("chessboard.png")
