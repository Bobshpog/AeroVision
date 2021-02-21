from dataclasses import dataclass, field
from typing import Union, List, Tuple

import cv2
from tqdm import trange

import src.util.image_transforms as tforms
from src.data import matlab_reader
from src.geometry.numpy.mesh import *
from src.geometry.numpy.transforms import mesh_compatibility_creation, tip_arr_creation
from src.util.SSIM_PIL import compare_ssim
from src.util.error_helper_functions import calc_errors
from src.util.loss_functions import L_infinity


#
# @dataclass
# class FiniteElementWingModel:
#     coordinates: np.ndarray
#     ir_idx_list: list
#     texture_wing: str
#     texture_tip: Union[str, None]
#     wing_path: str
#     old_wing_path: str
#     tip_path: str
#     cameras: list
#     wing_vertices_num: int
#     tip_vertices_num: int
#     plotter: pv.BasePlotter
#     resolution: list
#     cmap: str
#
#
#     def __post_init__(self):
#         self.wing = Mesh(self.wing_path, self.texture_wing)
#         self.tip = Mesh(self.tip_path, texture=self.texture_tip)
#         self.old_wing = Mesh(self.old_wing_path)
#         self.compatibility_arr = mesh_compatibility_creation(self.wing.vertices)
#         self.tip_arr = tip_arr_creation(self.old_wing.vertices)
#
#
#     def _get_new_position(self, displacement):
#         """
#         Recovers the new coordinates of the wing updated with displacement
#
#         Args:
#             displacement: np array of the displacement for each row size N*3
#
#         Returns:
#             (new wing position, new tip position)
#
#
#         """
#         TIP_RADIUS = 0.008
#         NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
#         wing_table = self.wing.table
#         tip_table = self.tip.table
#         new_tip_position = np.zeros((self.tip_vertices_num, 3), dtype='float')
#         new_wing_position = np.zeros((self.wing_vertices_num, 3), dtype='float')
#         tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
#         x = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
#         y = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
#         count, tip_count, wing_count = 0, 0, 0
#         for idx, cord in enumerate(self.coordinates):
#             if cord[1] >= 0.605:
#                 for i in range(30):
#                     new_tip_position[tip_table[cord2index(cord + (0, x[i], y[i]))]] = displacement[idx]
#                 tip_count += 1
#             elif cord2index(cord) in wing_table:
#                 new_wing_position[wing_table[cord2index(cord)]] = displacement[idx]
#                 wing_count += 1
#             else:
#                 count += 1
#         # TODO: Ido, Runs fine without Assertions
#         # assert(wing_count==self.wing_vertices_num-self.tip_vertices_num/30)
#         # assert(tip_count==self.tip_vertices_num/30)
#         return self.wing.vertices + new_wing_position, self.tip.vertices + new_tip_position
#
#     def _get_ir_cords(self, displacement):
#         """
#         Retrieves a np.array of the coordinates related to the movement
#         Args:
#             displacement: np array of the displacement for each row size N*3
#
#         Returns:
#             returns an np.array of the coordinates of the vertices associates with the ir coordinates
#
#         """
#         return self.coordinates[self.ir_idx_list] + displacement[self.ir_idx_list]
#
#     def _get_wing_photo(self, movement):
#         """
#         Take a photo of the wing with the tip in a cerain position
#
#        Args:
#            movement: [wing movement, tip movement]
#
#         Returns:
#            An image shot from camera of the wing and tip
#         """
#         cameras = self.cameras
#         photos = Mesh.get_photo((self.wing, self.tip),
#                                 movement=movement, resolution=self.resolution, camera=cameras,
#                                 cmap=self.cmap, plotter=self.plotter, background_photos=self.background_photos,
#                                 cam_noise_lambda=self.cam_noise_lambda)
#
#         return photos
#
#     def __call__(self, displacement):
#         """
#         Creates a photo and ir coordinates associated with a displacement
#         Args:
#             displacement: np array of the displacement in each axis
#
#         Returns:
#             photo, ir
#
#         """
#         movement = self._get_new_position(displacement)
#         ir = self._get_ir_cords(displacement)
#         photo = self._get_wing_photo(movement=movement)
#         return photo, ir


@dataclass
class SyntheticWingModel:
    coordinates: np.ndarray
    ir_idx_list: list
    texture_wing: str
    texture_tip: Union[str, None]
    wing_path: str
    tip_path: str
    old_wing_path: str
    cameras: list
    wing_vertices_num: int
    tip_vertices_num: int
    plotter: list   # list of plotters to choose in random
    resolution: list
    cmap: str
    background_photos: List[str] = field(default_factory=list)
    cam_noise_lambda: Tuple[float] = None
    random_texture: bool = False

    def __post_init__(self):
        self.wing = Mesh(self.wing_path, texture=self.texture_wing)
        self.tip = Mesh(self.tip_path, texture=self.texture_tip)
        self.old_wing = Mesh(self.old_wing_path)
        self.compatibility_arr = mesh_compatibility_creation(self.wing.vertices)
        self.tip_arr = tip_arr_creation(self.old_wing.vertices)
        self.background_photos = self.background_photos[:]

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
        new_tip_position = np.zeros((self.tip_vertices_num, 3), dtype='float')
        new_wing_position = (self.coordinates + displacement)[self.compatibility_arr]
        tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
        x = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
        y = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
        # count, tip_count, wing_count = 0, 0, 0
        for idx in self.tip_arr:
            cord = self.old_wing.vertices[idx]
            for i in range(30):
                new_tip_position[tip_table[cord2index(cord + (0, x[i], y[i]))]] = displacement[idx]

        # for idx, cord in enumerate(self.old_wing.vertices):
        #    if cord[1] >= 0.605:
        #        for i in range(30):
        #            new_tip_position[tip_table[cord2index(cord + (0, x[i], y[i]))]] = displacement[idx]
        #        tip_count += 1
        #    elif cord2index(cord) in wing_table:
        #        wing_count += 1
        #    else:
        #        count += 1
        return new_wing_position, self.tip.vertices + new_tip_position

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
        if self.random_texture:
            photos = Mesh.get_many_noisy_photos((self.wing, self.tip),
                                                movement=movement, resolution=self.resolution, camera=cameras,
                                                cmap=self.cmap, plotter=random.choice(self.plotter),
                                                background_photos=None, cam_noise_lambda=self.cam_noise_lambda)
        else:
            photos = Mesh.get_many_photos((self.wing, self.tip),
                                          movement=movement, resolution=self.resolution, camera=cameras,
                                          cmap=self.cmap, plotter=random.choice(self.plotter), background_photos=None,
                                          cam_noise_lambda=self.cam_noise_lambda)
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

    @staticmethod
    def photo_from_scales(mode_shape, scales, texture, resolution, camera, plotter=None, compatibility_arr=None,
                          tip_index_arr=None, cv=False, title=None, numpy_scales=False):
        """
        Creates a list of photos from mode shape and scales
        Args:
            mode_shape: mode shape as recived from matlabreader functions
            scales: list of string! num of scales >= num of scales in mode shape
            texture: path of texture, NO NUMPY TEXTURE SUPPORT
            resolution: resolution
            camera: list of camera, should be the same length of the list of scales!
            plotter: the plotter to use, pass it in case making a lot of usages, will create white background
            compatibility_arr: compatibility array between meshes, made from mesh_compatibility_creation, pass when
                                creating many photos.
            tip_index_arr: old mesh tip array, created via tip_arr_creation with old mesh verticies, pass when
                           creating many photos.
            cv: weather we would switch the red and blue to make it cv2 compatible
            title: title of the photo
        Returns:
            [photo] in length of len(scales)

        """
        mesh = Mesh("data/wing_off_files/synth_wing_v5.off", texture)
        tip = Mesh("data/wing_off_files/fem_tip.off")
        old_mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        if plotter is None:
            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("white")
        if compatibility_arr is None:
            compatibility_arr = mesh_compatibility_creation(mesh.vertices)
        if tip_index_arr is None:
            tip_index_arr = tip_arr_creation(old_mesh.vertices)
        TIP_RADIUS = 0.008
        NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
        tip_vertices_num = 930
        tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
        y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
        z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
        to_return = []
        curr_scale = np.zeros(mode_shape.shape[2])
        h1 = np.zeros((tip_vertices_num, 3), dtype='float')
        for i in range(len(scales)):
            if numpy_scales:
                temp = scales[i]
            else:
                temp = np.fromstring(scales[i], dtype=np.float32, sep=' ')
            curr_scale[:temp.shape[0]] = temp
            difference = (curr_scale * mode_shape).sum(axis=2).T
            g1 = mesh.vertices + difference[compatibility_arr]
            for id in tip_index_arr:
                for k in range(NUM_OF_VERTICES_ON_CIRCUMFERENCE):
                    cord = old_mesh.vertices[id]
                    vector = np.array((cord[0] + difference[id, 0], cord[1] + y_t[k] + difference[id, 1],
                                       cord[2] + z_t[k] + difference[id, 2]))
                    h1[tip.table[cord2index(cord + (0, y_t[k], z_t[k]))]] = vector
            photo = Mesh.get_photo([mesh, tip], [g1, h1], plotter=plotter, title=title,
                                   cmap=None, camera=camera[i], resolution=resolution)
            if cv:
                photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            to_return.append(photo)

        return to_return

    @staticmethod
    def two_scales_to_compare(mode_shape, real_scales, reconstruct_scales, texture, resolution, camera, loss_function,
                              ir,
                              plotter=None, compatibility_arr=None, tip_index_arr=None, cv=False, numpy_scales=False, ):
        """
        Creates a list of photos from mode shape and scales
        Args:
            mode_shape: mode shape as recived from matlabreader functions
            real_scales: list of string of the real_scales! num of scales >= num of scales in mode shape
            reconstruct_scales: list of string of the reconstructed scales! same number as ^
            texture: path of texture, NO NUMPY TEXTURE SUPPORT
            resolution: resolution
            camera: list of camera, should be the same length of the list of scales!
            loss_function: loss function to use
            ir: ir indicies
            plotter: the plotter to use, pass it in case making a lot of usages, will create white background
            compatibility_arr: compatibility array between meshes, made from mesh_compatibility_creation, pass when
                                creating many photos.
            tip_index_arr: old mesh tip array, created via tip_arr_creation with old mesh verticies, pass when
                           creating many photos.
            cv: weather we would switch the red and blue to make it cv2 compatible
        Returns:
            [photo] in length of len(scales)

        """
        first_photo = SyntheticWingModel.photo_from_scales(mode_shape, real_scales, texture, resolution, camera,
                                                           plotter, compatibility_arr, tip_index_arr, cv,
                                                           numpy_scales=numpy_scales)
        second_photo = SyntheticWingModel.photo_from_scales(mode_shape, reconstruct_scales, texture, resolution, camera,
                                                            plotter, compatibility_arr, tip_index_arr, cv,
                                                            numpy_scales=numpy_scales)
        to_return = []
        color1 = (0, 0, 0)
        color2 = (0, 0, 0)
        color3 = (0, 0, 255)
        padding = 150

        if not numpy_scales:
            for i in range(len(real_scales)):
                real_scales[i] = np.fromstring(real_scales[i], dtype=np.float32, sep=' ')
                reconstruct_scales[i] = np.fromstring(reconstruct_scales[i], dtype=np.float32, sep=' ')
            real_scales = np.array(real_scales)
            reconstruct_scales = np.array(reconstruct_scales)
        # real_scales = torch.tensor(real_scales, device="cpu")
        # reconstruct_scales = torch.tensor(reconstruct_scales, device="cpu")
        l_inf = L_infinity(mode_shape, 1, real_scales, reconstruct_scales)
        # print(torch.max(l_inf))
        # l_inf = l_inf.numpy()
        print(l_inf.flatten())
        err = calc_errors(loss_function, mode_shape, 1, ir, real_scales, reconstruct_scales)
        scale_err = err[3].numpy()
        for i in range(len(first_photo)):
            pil_img1 = Image.fromarray(np.uint8(first_photo[i] * 255))
            pil_img2 = Image.fromarray(np.uint8(second_photo[i] * 255))
            ssim = compare_ssim(pil_img1, pil_img2)

            img_a = cv2.hconcat([first_photo[i], second_photo[i]])
            img_d = np.ones(shape=(img_a.shape[0] + padding, img_a.shape[1], img_a.shape[2]))
            img_d[padding:] = img_a
            img_d[padding + 80:, resolution[0]] = 0
            img_d[padding + 80, :] = 0

            cv2.putText(img_d, "general errors:", (200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, color3,
                        lineType=2)
            cv2.putText(img_d, "scale errors:", (int(1.4 * resolution[0]), 20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, color3,
                        lineType=2)
            cv2.putText(img_d, "ssim:" + f'{ssim: .2f}', (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.75, color1,
                        lineType=2)
            cv2.putText(img_d, "3d reconstruction:" + f'{err[0][i]: .3e}', (0, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color1,
                        lineType=2)
            cv2.putText(img_d, "ir reconstruction:" + f'{err[1][i]: .3e}', (0, 110), cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color1,
                        lineType=2)
            cv2.putText(img_d, "avg reconstruction:" + f'{err[2][i]: .3e}', (0, 140), cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color1,
                        lineType=2)
            cv2.putText(img_d, "L inifinity:" + f'{l_inf[i]: .3e}', (0, 170), cv2.FONT_HERSHEY_TRIPLEX, 0.75, color1,
                        lineType=2)
            cv2.putText(img_d, "scale 0:" + f'{scale_err[0][i]: .3e}', (resolution[0] + 1, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color2,
                        lineType=2)
            cv2.putText(img_d, "scale 1:" + f'{scale_err[1][i]: .3e}', (resolution[0] + 1, 80),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color2,
                        lineType=2)
            cv2.putText(img_d, "scale 2:" + f'{scale_err[2][i]: .3e}', (resolution[0] + 1, 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color2,
                        lineType=2)
            cv2.putText(img_d, "scale 3:" + f'{scale_err[3][i]: .3e}', (resolution[0] + 1, 140),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color2,
                        lineType=2)
            cv2.putText(img_d, "scale 4:" + f'{scale_err[4][i]: .3e}', (resolution[0] + 1, 170),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                        color2,
                        lineType=2)
            cv2.putText(img_d, "scale 5:" + f'{scale_err[5][i]: .3e}', (2 * resolution[0] - 260, 50),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.75, color2,
                        lineType=2)
            cv2.putText(img_d, "scale 6:" + f'{scale_err[6][i]: .3e}', (2 * resolution[0] - 260, 80),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.75, color2,
                        lineType=2)
            cv2.putText(img_d, "scale 7:" + f'{scale_err[7][i]: .3e}', (2 * resolution[0] - 260, 110),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.75, color2,
                        lineType=2)
            cv2.putText(img_d, "scale 8:" + f'{scale_err[8][i]: .3e}', (2 * resolution[0] - 260, 140),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.75, color2,
                        lineType=2)
            cv2.putText(img_d, "scale 9:" + f'{scale_err[9][i]: .3e}', (2 * resolution[0] - 260, 170),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.75, color2,
                        lineType=2)
            cv2.putText(img_d, "reconstructed scales:", (800, 210), cv2.FONT_HERSHEY_TRIPLEX, 1, color3,
                        lineType=2)
            cv2.putText(img_d, "real scales:", (200, 210), cv2.FONT_HERSHEY_TRIPLEX, 1, color3,
                        lineType=2)
            to_return.append(img_d)

        return to_return

    @staticmethod
    def noisy_image_creation(cam, pois, s_p, gaussian_var, gaussian_mean=0, cam_noise=None):
        mesh = Mesh("data/wing_off_files/synth_wing_v5.off", "data/textures/checkers_dark_blue.png")
        tip = Mesh('data/wing_off_files/fem_tip.off')
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('white')
        text = "gauss:" + f'{gaussian_var: .2f}' + " poisson:" + f'{pois: .2f}' + " S&P:" + f'{s_p: .2f}'
        res = [640, 640]
        normal_factor = res[0] / 640
        photo = Mesh.get_many_photos([mesh, tip], [mesh.vertices, tip.vertices], res, "jet", plotter,
                                     [cam], cam_noise_lambda=cam_noise)
        tform = tforms.TranformSingleNoisyBW(np.zeros(photo.shape).astype(np.float32), pois, gaussian_mean,
                                             gaussian_var, s_p)
        photo = tform(photo)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(30 * normal_factor), int(30 * normal_factor))
        font_scale = normal_factor
        color = (0, 0, 0)
        thickness = 2
        return cv2.putText(photo[0], text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def radical_list_creation(wing_path, p):
        #   changed after testing, proved that its the most moving vers so made it efficient
        if not 0 <= p <= 1:
            raise ValueError("Probability is between 0 and 1")
        mesh = Mesh(wing_path)
        num_to_return = int(mesh.vertices.shape[0] * p)
        return mesh.vertices[:, 1].argsort()[-num_to_return:][::-1]

    @staticmethod
    def radical_list_creation_VERY_SLOW(wing_path, p=1, num_of_scales=10):
        #   it returns (radical list, the total L2 distance for each vertex)
        mesh = Mesh(wing_path)
        mode_shape = matlab_reader.read_modal_shapes("data/mode_shapes/synth_mode_shapes_9103_10.mat", 10)
        scales = matlab_reader.read_data("data/synt_data_mat_files/data2.mat")[2]
        comp_arr = mesh_compatibility_creation(mesh.vertices)
        mean_scales = np.mean(scales, axis=0)
        static_diff = (mean_scales * mode_shape).sum(axis=2)
        if num_of_scales < 10:
            static_diff[num_of_scales:] = 0
        tot_distance = np.zeros(mode_shape.shape[1])
        for phase in trange(scales.shape[0]):
            difference = (scales[phase, :] * mode_shape).sum(axis=2)
            distance = np.linalg.norm(difference - static_diff, axis=0)
            tot_distance += distance
        tot_distance = tot_distance[comp_arr]
        num_to_return = int(mesh.vertices.shape[0] * p)
        return tot_distance.argsort()[-num_to_return:][::-1], tot_distance

    @staticmethod
    def create_movement_vector(mode_shape, scale, data_scale, wing, tip, old_mesh_vertices,
                               compatibility_arr, tip_arr, y_t, z_t):
        right_diff = (scale * mode_shape).sum(axis=2).T / data_scale
        right_movement = wing.vertices + right_diff[compatibility_arr]
        NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
        good_tip_movement = np.zeros(tip.vertices.shape, dtype='float')
        for id in tip_arr:
            for k in range(NUM_OF_VERTICES_ON_CIRCUMFERENCE):
                cord = old_mesh_vertices[id]
                vector = np.array((cord[0] + right_diff[id, 0], cord[1] + y_t[k] + right_diff[id, 1],
                                   cord[2] + z_t[k] + right_diff[id, 2]))
                good_tip_movement[tip.table[cord2index(cord + (0, y_t[k], z_t[k]))]] = vector

        return right_movement, good_tip_movement
