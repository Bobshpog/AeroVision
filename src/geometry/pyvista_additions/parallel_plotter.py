import pyvista as pv
from multiprocessing import Process, Manager
from copy import deepcopy
from abc import ABC
from src.geometry.pyvista_additions.improvd_pyvista_renderer import ImprovedPlotter
from src.geometry.numpy.mesh import *
from src.data import matlab_reader
from src.geometry.numpy.transforms import mesh_compatibility_creation, tip_arr_creation
from src.geometry.numpy.wing_models import SyntheticWingModel
import cv2
from src.util.loss_functions import L_infinity
from src.util.error_helper_functions import calc_errors
from torch import norm
# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#

class ParallelPlotterBase(Process, ABC):
    # from cfg import VIS_CMAP, VIS_STRATEGY, VIS_SHOW_EDGES, VIS_SMOOTH_SHADING, \
    #     VIS_N_MESH_SETS, VIS_SHOW_GRID, VIS_SHOW_NORMALS

    def __init__(self):
        super().__init__()
        self.sd = Manager().dict()
        self.sd['epoch'] = -1
        self.sd['poison'] = False

        self.last_plotted_epoch = -1
        # self.f = faces if self.VIS_STRATEGY == 'mesh' else None
        self.train_d, self.val_d, self.data_cache, self.plt_title = None, None, None, None

        # self.kwargs = {'smooth_shade_on': self.VIS_SMOOTH_SHADING, 'show_edges': self.VIS_SHOW_EDGES,
        #                'strategy': self.VIS_STRATEGY, 'cmap': self.VIS_CMAP,
        #                'grid_on': self.VIS_SHOW_GRID}

    def run(self):
        # Init on consumer side:
        pv.set_plot_theme("document")
        while 1:
            try:
                if self.last_plotted_epoch != -1 and self.sd['poison']:  # Plotted at least one time + Poison
                    print(f'Pipe poison detected. Displaying one last time, and then exiting plotting supervisor')
                    self.try_update_data(final=True)
                    self.plot_data()
                    break
            except (BrokenPipeError, EOFError):  # Missing parent
                print(f'Producer missing. Exiting plotting supervisor')
                break

            self.try_update_data()
            self.plot_data()

    # Meant to be called by the consumer
    def try_update_data(self, final=False):
        current_epoch = self.sd['epoch']
        if current_epoch != self.last_plotted_epoch:
            self.last_plotted_epoch = current_epoch
            self.train_d, self.val_d = deepcopy(self.sd['data'])
            #   train_d and val_d are list of tupples (X, Y, reconstructed Y)
        if final:
            self.plt_title = f'Final visualization before closing for Epoch {self.last_plotted_epoch}'
        else:
            self.plt_title = f'Visualization for Epoch {self.last_plotted_epoch}'

            # Update version with one single read
        # Slight problem of atomicity here - with respect to current_epoch. Data may have changed in the meantime -
        # but it is still new data, so no real problem. May be resolved with lock = manager.Lock() before every update->
        # lock.acquire() / lock.release(), but this will be problematic for the main process - which we want to optimize

    # Meant to be called by the consumer
    def plot_data(self):
        raise NotImplementedError

    # self.sd[data] = (train data, val data) such that:
    # both data are lists of data_point and each data point is (X, Y, reconstructed Y)

    # Meant to be called by the producer
    def push(self, new_epoch, new_data):
        # new_data = (train_dict,vald_dict)
        old_epoch = self.sd['epoch']
        assert new_epoch != old_epoch

        # Update shared data (possibly before process starts)
        self.sd['data'] = new_data
        self.sd['epoch'] = new_epoch

        if old_epoch == -1:  # First push
            self.start()

    def cache(self, data):
        self.data_cache = data

    def uncache(self):
        cache = self.data_cache
        self.data_cache = None
        return cache
        # Meant to be called by the producer

    def finalize(self):
        self.sd['poison'] = True
        print('Workload completed - Please exit plotter to complete execution')
        self.join()


# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#
class RunTimeWingPlotter(ParallelPlotterBase):
    def __init__(self, mean_photo, texture, cam_location, mode_shapes, wing_path, tip_path, ir_index, output_scaling,
                 old_mesh_path='data/wing_off_files/synth_wing_v3.off', background_image=None):
        super().__init__()

        self.old_mesh_path = old_mesh_path
        if isinstance(texture, str):
            texture = [texture]
        self.texture = texture
        self.mean_photo = mean_photo
        self.cam = cam_location
        self.background_image = background_image
        self.mesh_path = wing_path
        self.tip_path = tip_path
        self.mode_shape = mode_shapes
        self.compatibility_arr = mesh_compatibility_creation(Mesh(self.mesh_path).vertices)
        self.output_scaling = output_scaling
        self.ir = ir_index
        self.tip_arr = tip_arr_creation(Mesh(old_mesh_path).vertices)
        NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
        TIP_RADIUS = 0.008
        tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
        self.y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
        self.z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)

    # def prepare_plotter_dict(self, params, network_output):
    #
    #     dict = {'texture': params["texture"],
    #             'cam': params['camera_position'],
    #             'background_picture': params['background_picture'],
    #             'mode_shape_path': params['mode_shape_path'],
    #             ''
    #
    #             }
    #     return dict

    def plot_data(self):
        self.plot_cv()
        self.plot_pyvista()

    def plot_pyvista(self):
        row_w = [2] + [5 for _ in range(len(self.val_d) + len(self.train_d))]
        plotter = ImprovedPlotter(shape=(len(self.val_d) + len(self.train_d) + 1, 5), row_weights=row_w,
                                  col_weights=[0.7, 0.8, 0.8, 1, 1], border=True, border_width=5, border_color="black")
        tex = random.choice(self.texture)
        plotter.link_views()
        plotter.set_background("white")
        self.set_background_image(plotter)
        self.add_text_to_plotter(plotter, 2, 2)
        old_mesh = Mesh(self.old_mesh_path)
        for row, data_point in zip(range(len(self.train_d)), self.train_d):
            good_mesh = Mesh(self.mesh_path, tex)
            good_tip = Mesh(self.tip_path)
            bad_mesh = Mesh(self.mesh_path, tex)
            bad_tip = Mesh(self.tip_path)
            self.plot_row(data_point, row + 1, plotter, good_mesh, good_tip, bad_mesh, bad_tip, old_mesh)

        for row, data_point in zip(range(len(self.val_d)), self.val_d):
            good_mesh = Mesh(self.mesh_path, tex)
            good_tip = Mesh(self.tip_path)
            bad_mesh = Mesh(self.mesh_path, tex)
            bad_tip = Mesh(self.tip_path)
            self.plot_row(data_point, row + len(self.train_d) + 1, plotter, good_mesh, good_tip, bad_mesh,
                          bad_tip, old_mesh)
        plotter.enable_anti_aliasing()
        plotter.show(full_screen=True)

    def plot_row(self, data_point, row, plotter, good_mesh, good_tip, bad_mesh, bad_tip, old_mesh):
        good_mesh.plot_faces(index_row=row, plotter=plotter, index_col=3, show=False, camera=self.cam)
        good_tip.plot_faces(show=False, index_row=row, plotter=plotter, index_col=3)
        bad_mesh.plot_faces(index_row=row, index_col=4, show=False, camera=self.cam, plotter=plotter)
        bad_tip.plot_faces(show=False, index_row=row, index_col=4, plotter=plotter)

        plotter.subplot(row, 2)
        gray_photo = np.zeros(shape=(data_point[0][0].shape[0], data_point[0][0].shape[1], 3))
        gray_photo[:, :, 0] = data_point[0][0]
        gray_photo[:, :, 1] = data_point[0][0]
        gray_photo[:, :, 2] = data_point[0][0]
        plotter.add_background_photo(gray_photo*255)
        plotter.subplot(row, 1)

        gray_photo_with_mean = np.zeros(shape=(data_point[0][0].shape[0], data_point[0][0].shape[1], 3))
        photo_with_mean = add_mean_photo_to_photo(self.mean_photo, data_point[0][0])
        gray_photo_with_mean[:, :, 0] = photo_with_mean
        gray_photo_with_mean[:, :, 1] = photo_with_mean
        gray_photo_with_mean[:, :, 2] = photo_with_mean
        plotter.add_background_photo(photo_with_mean * 255)

        right_movement, good_tip_movement = SyntheticWingModel.create_movement_vector(
            self.mode_shape, data_point[1], self.output_scaling, good_mesh, good_tip, old_mesh.vertices,
            self.compatibility_arr, self.tip_arr, self.y_t, self.z_t
        )
        wrong_movement, bad_tip_movement = SyntheticWingModel.create_movement_vector(
            self.mode_shape, data_point[2], self.output_scaling, bad_mesh, bad_tip, old_mesh.vertices,
            self.compatibility_arr, self.tip_arr, self.y_t, self.z_t
        )

        plotter.update_coordinates(right_movement, good_mesh.pv_mesh)
        plotter.update_coordinates(good_tip_movement, good_tip.pv_mesh)
        plotter.update_coordinates(wrong_movement, bad_mesh.pv_mesh)
        plotter.update_coordinates(bad_tip_movement, bad_tip.pv_mesh)

    def set_background_image(self, plotter, mesh_col=(4, 3)):
        if not self.background_image:
            return None
        for row in range(len(self.val_d) + len(self.train_d)):
            for col in mesh_col:
                plotter.subplot(row, col)
                plotter.add_background_image(random.choice(self.background_image), as_global=False)

    def add_text_to_plotter(self, plotter, num_training, num_valid):    # assumes same position of subplots
        color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_TRIPLEX
        size = 1.4
        plotter.subplot(0, 0)
        txt = cv2.putText(
            np.ones(shape=(100, 450, 3)) * 255, "epoch " + str(self.last_plotted_epoch), (120, 50), font, size, color,
            thickness=2
        )
        plotter.add_background_photo(txt)
        txt = cv2.putText(
            np.ones(shape=(100, 450, 3)) * 255, "Human input", (50, 50), font, size, color, thickness=2
        )
        plotter.subplot(0, 1)
        plotter.add_background_photo(txt)
        txt = cv2.putText(
            np.ones(shape=(100, 550, 3)) * 255,  "Real input", (120, 50), font, size, color, thickness=2
        )
        plotter.subplot(0, 2)
        plotter.add_background_photo(txt)
        txt = cv2.putText(
            np.ones(shape=(100, 550, 3)) * 255, "Reconstructed", (75, 50), font, size, color, thickness=2
        )
        plotter.subplot(0, 3)
        plotter.add_background_photo(txt)
        txt = cv2.putText(
            np.ones(shape=(100, 550, 3)) * 255,  "True scales", (105, 50), font, size, color, thickness=2
        )
        plotter.subplot(0, 4)
        plotter.add_background_photo(txt)
        for i in range(num_training):
            txt = cv2.putText(
                np.ones(shape=(300, 450, 3)) * 255,  "Training", (45, 150), font, size + 1, color, thickness=2
            )
            plotter.subplot(i+1, 0)
            plotter.add_background_photo(txt)
        for i in range(num_valid):
            txt = cv2.putText(
                np.ones(shape=(300, 450, 3)) * 255, "Validation", (40, 150), font, size + 0.5, color, thickness=2
            )
            plotter.subplot(i+1 + num_training, 0)
            plotter.add_background_photo(txt)

    def plot_cv(self):
        headline_color = (0, 0, 255)
        resolution = [640, 480]
        text_w = 250
        txt2_w = text_w + 1280
        headlines = np.ones(shape=(100, txt2_w, 3))
        cv2.putText(headlines, "general errors:", (100 + text_w, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, headline_color,
                    lineType=2, thickness=2)
        cv2.putText(headlines, "scale errors:", (resolution[0] + text_w + 120, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5,
                    headline_color,
                    lineType=2, thickness=2)
        cv2.putText(headlines, f"epoch {str(self.last_plotted_epoch)}:", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, headline_color,

                    lineType=2, thickness=2)
        headlines[71:75, :, :] = 0
        headlines[:, text_w - 4:text_w, :] = 0
        for im in self.create_txt_out_of_scale(np.array([self.train_d[0][2], self.train_d[1][2]]) / self.output_scaling,
                                          np.array([self.train_d[0][1], self.train_d[1][1]]) / self.output_scaling, "Training"):
            headlines = cv2.vconcat([headlines, im])
        for im in self.create_txt_out_of_scale(np.array([self.val_d[0][2], self.val_d[1][2]]) / self.output_scaling,
                                               np.array([self.val_d[0][1], self.val_d[1][1]]) / self.output_scaling, "valid"):
            headlines = cv2.vconcat([headlines, im])
        cv2.imshow("headlines", headlines)

    def create_txt_out_of_scale(self, scale_a, scale_b, from_where): # scales are list of scales
        general_err_color = (0, 0, 0)
        scale_err_color = (0, 0, 0)
        text_color = (0, 0, 255)
        padding = 70
        resolution = [640, 480]
        l_inf = L_infinity(self.mode_shape, 1, scale_a, scale_b)
        err = calc_errors(norm, self.mode_shape, 1, self.ir, scale_a, scale_b)
        scale_err = err[3].numpy()
        text_w = 250
        img_text_height = 20
        to_return = []
        for i in range(scale_a.shape[0]):
            if i == 1:
                time = "Second"
            else:
                time = "First"
            text = np.ones(shape=(100 + padding, text_w, 3))
            cv2.putText(text, time, (15, img_text_height + 20), cv2.FONT_HERSHEY_TRIPLEX, 1.5, thickness=2,
                        color=text_color,
                        lineType=2)
            cv2.putText(text, from_where, (15, img_text_height + 75 + 20), cv2.FONT_HERSHEY_TRIPLEX, 1.5, thickness=2,
                        color=text_color,
                        lineType=2)
            text[:, text_w - 4:text_w, :] = 0
            img_d = np.ones(shape=(100 + padding, 1280, 3))

            cv2.putText(img_d, "3d reconstruction:" + f'{err[0][i]: .3e}', (0, img_text_height),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, general_err_color, lineType=2)

            cv2.putText(img_d, "ir reconstruction:" + f'{err[1][i]: .3e}', (0, img_text_height + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, general_err_color, lineType=2)

            cv2.putText(img_d, "L inifinity:" + f'{l_inf[i]: .3e}', (0, img_text_height + 30 * 3),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, general_err_color, lineType=2)

            cv2.putText(img_d, "scale 0:" + f'{scale_err[0][i]: .3e}', (resolution[0] + 1, img_text_height),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 1:" + f'{scale_err[1][i]: .3e}', (resolution[0] + 1, img_text_height + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 2:" + f'{scale_err[2][i]: .3e}', (resolution[0] + 1, img_text_height + 30 * 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 3:" + f'{scale_err[3][i]: .3e}', (resolution[0] + 1, img_text_height + 30 * 3),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 4:" + f'{scale_err[4][i]: .3e}', (resolution[0] + 1, img_text_height + 30 * 4),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 5:" + f'{scale_err[5][i]: .3e}', (2 * resolution[0] - 260, img_text_height),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 6:" + f'{scale_err[6][i]: .3e}', (2 * resolution[0] - 260, img_text_height + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 7:" + f'{scale_err[7][i]: .3e}',
                        (2 * resolution[0] - 260, img_text_height + 30 * 2),cv2.FONT_HERSHEY_TRIPLEX,  0.75,
                        scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 8:" + f'{scale_err[8][i]: .3e}', (2 * resolution[0] - 260, img_text_height + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)

            cv2.putText(img_d, "scale 9:" + f'{scale_err[9][i]: .3e}', (2 * resolution[0] - 260, img_text_height + 120),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, scale_err_color, lineType=2)
            img_f = cv2.hconcat([text, img_d])
            img_f[96 + padding:100 + padding, :, :] = 0
            to_return.append(img_f)
        return to_return


def add_mean_photo_to_photo(mean_photo, X):
    return cv2.cvtColor(mean_photo, cv2.COLOR_RGB2GRAY) + X
