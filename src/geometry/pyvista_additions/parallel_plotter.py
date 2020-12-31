import pyvista as pv
from multiprocessing import Process, Manager
from copy import deepcopy
from abc import ABC
from src.geometry.pyvista_additions.improvd_pyvista_renderer import ImprovedPlotter
from src.geometry.numpy.mesh import *
from src.data import matlab_reader
from src.geometry.numpy.transforms import mesh_compatibility_creation, tip_arr_creation


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
    def __init__(self, mean_photo, texture, cam_location, mode_shape, wing_path, tip_path,
                 old_mesh_path='data/wing_off_files/synth_wing_v3.off', background_image=None):
        super().__init__()
        self.old_mesh = Mesh(old_mesh_path)
        self.texture = texture
        self.mean_photo = mean_photo
        self.cam = cam_location
        self.background_image = background_image
        self.mesh_path = wing_path
        self.tip_path = tip_path
        self.mode_shape = mode_shape
        self.compatibility_arr = mesh_compatibility_creation(Mesh(self.mesh_path).vertices)

        self.tip_arr = tip_arr_creation(self.old_mesh.vertices)
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

        for row, data_point in zip(range(len(self.train_d)), self.train_d):
            good_mesh = Mesh(self.mesh_path, self.texture)
            good_tip = Mesh(self.tip_path)
            bad_mesh = Mesh(self.mesh_path, self.texture)
            bad_tip = Mesh(self.tip_path)
            self.plot_row(data_point, row, plotter, "training", good_mesh, good_tip, bad_mesh, bad_tip)

        for row, data_point in zip(range(len(self.val_d)), self.val_d):
            good_mesh = Mesh(self.mesh_path, self.texture)
            good_tip = Mesh(self.tip_path)
            bad_mesh = Mesh(self.mesh_path, self.texture)
            bad_tip = Mesh(self.tip_path)
            self.plot_row(data_point, row + len(self.train_d), plotter, "validation", good_mesh, good_tip, bad_mesh, bad_tip)

        plotter.show()

    def plot_row(self, data_point, row, plotter, from_where, good_mesh, good_tip, bad_mesh, bad_tip): # from_where = "training" \ "validation"
        good_mesh.plot_faces(index_row=row, title="Mesh reconstructed from true scales", plotter=plotter, index_col=2,
                                        show=False, camera=self.cam)
        good_tip.plot_faces(show=False, index_row=row, plotter=plotter, index_col=2)
        bad_mesh.plot_faces(index_row=row, index_col=3, title="Mesh reconstructed from net generated scales",
                                       show=False, camera=self.cam, plotter=plotter)
        bad_tip.plot_faces(show=False, index_row=row, index_col=3, plotter=plotter)

        plotter.subplot(row, 0)
        plotter.add_text("Input image in "+from_where, position="upper_edge", font_size=10, color="black")
        plotter.subplot(row, 1)
        plotter.add_text("Image plus avg photo", position="upper_edge", font_size=10, color="black")

        plotter.subplot(row, 0)
        plotter.remove_background_image()
        plotter.add_background_photo(data_point[0])

        plotter.subplot(0, 1)
        plotter.remove_background_image()
        plotter.add_background_photo(add_mean_photo_to_photo(self.mean_photo, data_point[0]))

        right_diff = (data_point[1] * self.mode_shape).sum(axis=2).T
        right_movement = good_mesh.vertices + right_diff[self.compatibility_arr]
        NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
        good_tip_movement = np.zeros(good_tip.vertices.shape, dtype='float')
        for id in self.tip_arr:
            for k in range(NUM_OF_VERTICES_ON_CIRCUMFERENCE):
                cord = self.old_mesh.vertices[id]
                vector = np.array((cord[0] + right_diff[id, 0], cord[1] + self.y_t[k] + right_diff[id, 1],
                                   cord[2] + self.z_t[k] + right_diff[id, 2]))
                good_tip_movement[good_tip.table[cord2index(cord + (0, self.y_t[k], self.z_t[k]))]] = vector

        wrong_diff = (data_point[2] * self.mode_shape).sum(axis=2).T
        wrong_movement = bad_mesh.vertices + wrong_diff[self.compatibility_arr]

        bad_tip_movement = np.zeros(bad_tip.vertices.shape, dtype='float')
        for id in self.tip_arr:
            for k in range(NUM_OF_VERTICES_ON_CIRCUMFERENCE):
                cord = self.old_mesh.vertices[id]
                vector = np.array((cord[0] + wrong_diff[id, 0], cord[1] + self.y_t[k] + wrong_diff[id, 1],
                                   cord[2] + self.z_t[k] + wrong_diff[id, 2]))
                bad_tip_movement[bad_tip.table[cord2index(cord + (0, self.y_t[k], self.z_t[k]))]] = vector

        plotter.update_coordinates(right_movement, good_mesh.pv_mesh)
        plotter.update_coordinates(good_tip_movement, good_tip)
        plotter.update_coordinates(wrong_movement, bad_mesh.pv_mesh)
        plotter.update_coordinates(bad_tip_movement, good_tip)

    def set_background_image(self, plotter, mesh_col=(2, 3)):
        if not self.background_image:
            return None
        for row in range(len(self.val_d) + len(self.train_d)):
            for col in mesh_col:
                plotter.subplot(row, col)
                plotter.add_background_image(random.choice(self.background_image), as_global=False)


def add_mean_photo_to_photo(mean_photo, X):  # TODO most likely need to play with dimantions
    return mean_photo + X
