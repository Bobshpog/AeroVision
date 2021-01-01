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
    def __init__(self, mean_photo, texture, cam_location, mode_shapes, wing_path, tip_path,
                 old_mesh_path='data/wing_off_files/synth_wing_v3.off', background_image=None, scale_by=10000):
        super().__init__()

        self.old_mesh_path = old_mesh_path
        self.texture = texture
        self.mean_photo = mean_photo
        self.cam = cam_location
        self.background_image = background_image
        self.mesh_path = wing_path
        self.tip_path = tip_path
        self.mode_shape = mode_shapes
        self.compatibility_arr = mesh_compatibility_creation(Mesh(self.mesh_path).vertices)
        self.data_scale = scale_by

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

        plotter = ImprovedPlotter(shape=(len(self.val_d) + len(self.train_d), 4))
        plotter.set_background("white")
        self.set_background_image(plotter)

        old_mesh = Mesh(self.old_mesh_path)
        for row, data_point in zip(range(len(self.train_d)), self.train_d):
            good_mesh = Mesh(self.mesh_path, self.texture)
            good_tip = Mesh(self.tip_path)
            bad_mesh = Mesh(self.mesh_path, self.texture)
            bad_tip = Mesh(self.tip_path)
            self.plot_row(data_point, row, plotter, "training", good_mesh, good_tip, bad_mesh, bad_tip, old_mesh)

        for row, data_point in zip(range(len(self.val_d)), self.val_d):
            good_mesh = Mesh(self.mesh_path, self.texture)
            good_tip = Mesh(self.tip_path)
            bad_mesh = Mesh(self.mesh_path, self.texture)
            bad_tip = Mesh(self.tip_path)
            self.plot_row(data_point, row + len(self.train_d), plotter, "validation", good_mesh, good_tip, bad_mesh,
                          bad_tip, old_mesh)

        plotter.show()

    def plot_row(self, data_point, row, plotter, from_where, good_mesh, good_tip, bad_mesh, bad_tip, old_mesh): # from_where = "training" \ "validation"
        good_mesh.plot_faces(index_row=row, title="Mesh reconstructed from true scales", plotter=plotter, index_col=2,
                             show=False, camera=self.cam, font_size=7)
        good_tip.plot_faces(show=False, index_row=row, plotter=plotter, index_col=2)
        bad_mesh.plot_faces(index_row=row, index_col=3, title="Mesh reconstructed from net generated scales",
                            show=False, camera=self.cam, plotter=plotter, font_size=7)
        bad_tip.plot_faces(show=False, index_row=row, index_col=3, plotter=plotter)

        plotter.subplot(row, 0)
        gray_photo = np.zeros(shape=(data_point[0][0].shape[0], data_point[0][0].shape[1], 3))
        gray_photo[:, :, 0] = data_point[0][0]
        gray_photo[:, :, 1] = data_point[0][0]
        gray_photo[:, :, 2] = data_point[0][0]
        final_photo_without_mean = cv2.putText(
            gray_photo*255, "Input image in "+from_where, (0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), lineType=2
        )
        plotter.add_background_photo(final_photo_without_mean)

        plotter.subplot(row, 1)
        gray_photo_with_mean = np.zeros(shape=(data_point[0][0].shape[0], data_point[0][0].shape[1], 3))
        photo_with_mean = add_mean_photo_to_photo(self.mean_photo, data_point[0][0])
        gray_photo_with_mean[:, :, 0] = photo_with_mean
        gray_photo_with_mean[:, :, 1] = photo_with_mean
        gray_photo_with_mean[:, :, 2] = photo_with_mean
        final_photo_with_mean = cv2.putText(
            gray_photo_with_mean * 255, "Image plus avg photo", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
            lineType=2
        )
        plotter.add_background_photo(final_photo_with_mean)

        right_movement, good_tip_movement = SyntheticWingModel.create_movement_vector(
            self.mode_shape, data_point[1], self.data_scale, good_mesh, good_tip, old_mesh.vertices,
            self.compatibility_arr, self.tip_arr, self.y_t, self.z_t
        )
        wrong_movement, bad_tip_movement = SyntheticWingModel.create_movement_vector(
            self.mode_shape, data_point[2], self.data_scale, bad_mesh, bad_tip, old_mesh.vertices,
            self.compatibility_arr, self.tip_arr, self.y_t, self.z_t
        )

        plotter.update_coordinates(right_movement, good_mesh.pv_mesh)
        plotter.update_coordinates(good_tip_movement, good_tip.pv_mesh)
        plotter.update_coordinates(wrong_movement, bad_mesh.pv_mesh)
        plotter.update_coordinates(bad_tip_movement, bad_tip.pv_mesh)

    def set_background_image(self, plotter, mesh_col=(2, 3)):
        if not self.background_image:
            return None
        for row in range(len(self.val_d) + len(self.train_d)):
            for col in mesh_col:
                plotter.subplot(row, col)
                plotter.add_background_image(random.choice(self.background_image), as_global=False)



def add_mean_photo_to_photo(mean_photo, X):
    return cv2.cvtColor(mean_photo, cv2.COLOR_RGB2GRAY) + X
