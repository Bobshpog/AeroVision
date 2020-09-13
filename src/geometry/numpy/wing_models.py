from src.geometry.numpy.mesh import *


class FemWing:
    num_of_vertices_wing = 7724
    num_of_vertices_tip = 930
    wing_path = "data/wing_off_files/finished_fem_without_tip.off"
    tip_path = "data/wing_off_files/fem_tip.off"
    camera_pos = {
        'up_middle': [(0.047, -0.053320266561896174, 0.026735639600027315),
                      (0.05, 0.3, 0.02),
                      (0, 0, 1)],

        'down_middle': [(0.04581499400545182, -0.04477050005202985, -0.028567355483893577),
                        (0.05, 0.3, 0.02),
                        (0.001212842435223535, 0.13947688005070646, -1)],

        "up_right": [(0.11460619078012961, -0.04553696541254279, 0.038810512823530784),
                     (0.05, 0.3, 0.02),
                     (0, 0.16643488101070833, 1)],

        'down_right': [(0.11460619078012961, -0.04553696541254279, -0.038810512823530784),
                       (0.05, 0.3, 0.02),
                       (0, 0.16643488101070833, -1)],

        'up_left': [(-0.019770941905445285, -0.06082136750543311, 0.038694507832388224),
                    (0.05, 0.3, 0.02),
                    (0.041, 0.0438, 1)],

        'down_left': [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
                      (0.05, 0.3, 0.02),
                      (0.041, 0.0438, -1)]
        }

    @staticmethod
    def translate_displacement_from_data(coordinates, displacement):
        """
       getting two N*3 np arrays of the cooridnates and dissplacment of the fem wing in the lab and return a
       N*3 movement vector of our own wing

        Args:
            coordinates: np array of the coordinates, size N*3
            displacement: np array of the displacement for each row size N*3

        Returns:
            list [wing_displacement, tip_displacement] where both wing and tip displacement are np array to be
            used for movement vector

        """
        # TODO alex
        pass

    @staticmethod
    def get_ir_coords(coordinates, displacement, ir_id):
        """
        Retrieves a np.array of the coordinates related to the movement
        Args:
            coordinates: np array of the coordinates, size N*3
            displacement: np array of the displacement for each row size N*3
            ir_id: nparray of the id we search for

        Returns:
            returns an nparray of the coordinates of the vertices associates with the ir coordinates

        """
        pass

    @staticmethod
    def get_wing_photo(movement=None, texture=None, camera=None, f=None, cmap="jet", resolution=None,
                       plotter=None):
        """
        Take a photo of the wing with the tip in a cerain position

       Args:
           movement: V side vector
           f: map between (x,y,z) of vertex to scalar for the color map, used only if texture is not supplied
           texture: the texture to use
           cmap: the colormap to use, used only if texture is not supplied
           plotter: the pyvista plotter, clear the mesh "get_photo" in the plotter
           camera: the [camera position , focal point, view up] each (x,y,z) tuple, used only if plotter is supplied
           resolution: the image resolution [w,h]

        Returns:
           An image shot from camera of the wing and tip
        """

        pass

    @staticmethod
    def from_cvs_to_picture(displacement, coordinates, ir_id):
        """
        Retrieves a np.array of the coordinates related to the movement
        Args:
            coordinates: np array of the coordinates, size N*3
            displacement: np array of the displacement for each row size N*3
            ir_id: nparray of the ir id we search for

        Returns:
            returns a tuple (picture,ir) TODO make sure on format

        """
        pass
