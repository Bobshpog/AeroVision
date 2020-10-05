from dataclasses import dataclass

from src.geometry.numpy.mesh import *


@dataclass
class FiniteElementWingModel:
    coordinates: np.ndarray
    ir_idx_list: list
    texture: str
    wing_path: str
    tip_path: str
    cameras: list
    wing_vertices_num: int
    tip_vertices_num: int
    plotter: pv.BasePlotter
    resolution: list
    cmap: str

    def __post_init__(self):
        self.wing = Mesh(self.wing_path)
        self.tip = Mesh(self.tip_path)

    def _get_new_position(self, displacement):
        """
        Recovers the new coordinates of the wing updated with displacement

        Args:
            displacement: np array of the displacement for each row size N*3

        Returns:
            (new wing position, new tip position)


        """
        TIP_RADIUS = 0.008
        NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
        wing_table = self.wing.table
        tip_table = self.tip.table
        new_tip_position = np.zeros((self.tip_vertices_num, 3),dtype='float')
        new_wing_position = np.zeros((self.wing_vertices_num, 3),dtype='float')
        tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE,endpoint=False)
        x = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
        y = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
        count,tip_count,wing_count=0,0,0
        for idx, cord in enumerate(self.coordinates):
            if cord[1]>=0.605:
                for i in range(30):
                    new_tip_position[tip_table[cord2index(cord + (0, x[i], y[i]))]] = 0*cord + displacement[idx]
                tip_count+=1
            elif cord2index(cord) in wing_table:
                new_wing_position[wing_table[cord2index(cord)]] = 0*cord + displacement[idx]
                wing_count+=1
            else:
                count+=1
        # TODO: Ido, Runs fine without Assertions
        # assert(wing_count==self.wing_vertices_num-self.tip_vertices_num/30)
        # assert(tip_count==self.tip_vertices_num/30)
        return self.wing.vertices+new_wing_position, self.tip.vertices+new_tip_position

    def _get_ir_cords(self, displacement):
        """
        Retrieves a np.array of the coordinates related to the movement
        Args:
            displacement: np array of the displacement for each row size N*3

        Returns:
            returns an np.array of the coordinates of the vertices associates with the ir coordinates

        """
        return self.coordinates[self.ir_idx_list] + displacement[self.ir_idx_list]

    def _get_wing_photo(self, movement):
        """
        Take a photo of the wing with the tip in a cerain position

       Args:
           movement: [wing movement, tip movement]

        Returns:
           An image shot from camera of the wing and tip
        """
        cameras = self.cameras
        photos = Mesh.get_photo((self.wing, self.tip),
                                         movement=movement, resolution=self.resolution, camera=cameras,
                                         cmap=self.cmap,
                                         texture=[self.texture, None], plotter=self.plotter)
        return photos

    def __call__(self, displacement):
        """
        Creates a photo and ir coordinates associated with a displacement
        Args:
            displacement: np array of the displacement in each axis

        Returns:
            photo, ir

        """
        movement = self._get_new_position(displacement)
        ir = self._get_ir_cords(displacement)
        photo = self._get_wing_photo(movement=movement)
        return photo, ir



@dataclass
class SyntheticWingModel:
    coordinates: np.ndarray
    ir_idx_list: list
    texture: str
    wing_path: str
    tip_path: str
    cameras: list
    wing_vertices_num: int
    tip_vertices_num: int
    plotter: pv.BasePlotter
    resolution: list
    cmap: str

    def __post_init__(self):
        self.wing = Mesh(self.wing_path)
        self.tip = Mesh(self.tip_path)

    def _get_new_position(self, displacement):
        """
        Recovers the new coordinates of the wing updated with displacement

        Args:
            displacement: np array of the displacement for each row size N*3

        Returns:
            (new wing position, new tip position)


        """
        TIP_RADIUS = 0.008
        NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
        wing_table = self.wing.table
        tip_table = self.tip.table
        new_tip_position = np.zeros((self.tip_vertices_num, 3),dtype='float')
        new_wing_position = self.coordinates + displacement
        tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE,endpoint=False)
        x = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
        y = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
        count,tip_count,wing_count=0,0,0
        for idx, cord in enumerate(self.coordinates):
            if cord[1]>=0.605:
                for i in range(30):
                    new_tip_position[tip_table[cord2index(cord + (0, x[i], y[i]))]] =  displacement[idx]
                tip_count+=1
            elif cord2index(cord) in wing_table:
                wing_count+=1
            else:
                count+=1
        # TODO: Ido, Runs fine without Assertions
        # assert(wing_count==self.wing_vertices_num-self.tip_vertices_num/30)
        # assert(tip_count==self.tip_vertices_num/30)
        return new_wing_position, self.tip.vertices+new_tip_position

    def _get_ir_cords(self, displacement):
        """
        Retrieves a np.array of the coordinates related to the movement
        Args:
            displacement: np array of the displacement for each row size N*3

        Returns:
            returns an np.array of the coordinates of the vertices associates with the ir coordinates

        """
        return self.coordinates[self.ir_idx_list] + displacement[self.ir_idx_list]

    def _get_wing_photo(self, movement):
        """
        Take a photo of the wing with the tip in a cerain position

       Args:
           movement: [wing movement, tip movement]

        Returns:
           An image shot from camera of the wing and tip
        """
        cameras = self.cameras
        photos = Mesh.get_many_photos((self.wing, self.tip),
                                         movement=movement, resolution=self.resolution, camera=cameras,
                                         cmap=self.cmap,
                                         texture=[self.texture, None], plotter=self.plotter)
        return photos

    def __call__(self, displacement):
        """
        Creates a photo and ir coordinates associated with a displacement
        Args:
            displacement: np array of the displacement in each axis

        Returns:
            photo, ir

        """
        movement = self._get_new_position(displacement)
        ir = self._get_ir_cords(displacement)
        photo = self._get_wing_photo(movement=movement)
        return photo, ir
