import csv
import functools
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

from geometry.numpy.wing_models import FiniteElementWingModel


@functools.lru_cache(2)
def get_csv_len(csv_path):
    with open(csv_path, 'r') as csv_file:
        return len(csv_file.readlines()) - 1


def get_cords_from_csv(points_path):
    point_num = get_csv_len(points_path)
    with open(points_path, 'r') as point_file:
        point_file.readline()
        points_csv = csv.reader(point_file)
        cords_arr = np.zeros((point_num, 3), 'float')
        for idx, row in enumerate(points_csv):
            cords_arr[idx] = [float(x) for x in row[2:5]]  # indexes of relevant nodes
        return cords_arr


def process_csv_pair(points_path, scales_path):
    scales_csv_parameters_num = get_csv_len(scales_path)
    point_num = get_csv_len(points_path)
    with open(points_path, 'r') as point_file, open(scales_path, 'r') as scales_file:
        point_file.readline()
        scales_file.readline()
        points_csv = csv.reader(point_file)
        scales_csv = csv.reader(scales_file)
        disp_np_arr = np.zeros((point_num, 3), 'float')
        scales_np_arr = np.zeros(scales_csv_parameters_num, 'float')
        for idx, row in enumerate(points_csv):
            disp_np_arr[idx] = [float(x) for x in row[5:8]]  # indexes of relevant nodes
        scales_np_arr[:] = [row[1] for row in scales_csv]
        return disp_np_arr, scales_np_arr


class Config:
    mesh_wing_path = 'data/wing_off_files/finished_fem_without_tip.off'
    mesh_tip_path = 'data/wing_off_files/fem_tip.off'
    im_width = 640
    im_height = 480
    resolution = [im_width, im_height]
    num_scales = 11  # number of modal scales
    num_vertices_input = 9067
    compression = 'gzip'
    num_of_vertices_wing = 7724
    num_of_vertices_tip = 930
    ir_list = list(range(28))
    # cameras in pyvista format
    cameras = [[(0.047, -0.053320266561896174, 0.026735639600027315),
                (0.05, 0.3, 0.02),
                (0, 0, 1)],

               [(0.04581499400545182, -0.04477050005202985, -0.028567355483893577),
                (0.05, 0.3, 0.02),
                (0.001212842435223535, 0.13947688005070646, -1)],

               [(0.11460619078012961, -0.04553696541254279, 0.038810512823530784),
                (0.05, 0.3, 0.02),
                (0, 0.16643488101070833, 1)],

               [(0.11460619078012961, -0.04553696541254279, -0.038810512823530784),
                (0.05, 0.3, 0.02),
                (0, 0.16643488101070833, -1)],

               [(-0.019770941905445285, -0.06082136750543311, 0.038694507832388224),
                (0.05, 0.3, 0.02),
                (0.041, 0.0438, 1)],

               [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
                (0.05, 0.3, 0.02),
                (0.041, 0.0438, -1)]]
    texture = 'data/textures/checkers2.png'
    cmap = 'jet'


class DatabaseBuilder:

    def __init__(self, data_folder,raw_data_subfolder="synthetic_data_raw_samples", database_subfolder='databases'):
        """

        Args:
            data_folder: relative/absolute path to data_folder
            raw_data_subfolder: path to raw data sub folder relative to data_folder
            database_subfolder: path to database sub folder relative to data_folder
        """
        self.root_data_folder = Path(data_folder)
        self.raw_data_folder = Path(self.root_data_folder / raw_data_subfolder)
        self.db_folder = Path(self.root_data_folder / database_subfolder)
        if not self.db_folder.exists():
            os.mkdir(self.db_folder)

    def _csv_pair_generator(self):
        """
           A generator for csv data pairs
        """
        file_list = sorted(next(os.walk(self.raw_data_folder))[2])
        num_files = len(file_list) // 2
        for i in range(num_files):
            scale_file = file_list[i]
            point_file = file_list[num_files + i]
            yield self.raw_data_folder / point_file, self.raw_data_folder / scale_file

    def __call__(self, db_name=None):
        """
       Creates a database from data in self.raw_data_folder
        Args:
            db_name: name of new database, defaults to generic scheme

        Returns:
            path to the newly created database
        """
        if db_name is None:
            texture_name = Path(Config.texture).stem
            mesh_wing_name = Path(Config.mesh_wing_path).stem
            mesh_tip_name = Path(Config.mesh_tip_path).stem
            db_name = self.db_folder / f"{str(datetime.now())}__{mesh_wing_name}__"\
                                           f"{mesh_tip_name}__{texture_name}.hdf5"
        size = len(list(self._csv_pair_generator()))
        with h5py.File(db_name, 'w') as hf:
            hf.attrs['cameras'] = Config.cameras
            hf.attrs['mesh_wing_path'] = Config.mesh_wing_path
            hf.attrs['mesh_tip_path'] = Config.mesh_tip_path
            hf.attrs['resolution'] = Config.resolution
            hf.attrs['texture'] = Config.texture
            hf.attrs['num_scales'] = Config.num_scales
            dset_images = hf.create_dataset('images',
                                            shape=(size, len(Config.cameras), Config.im_height, Config.im_width, 4),
                                            compression=Config.compression,
                                            dtype=np.float)
            dset_scales = hf.create_dataset('scales', shape=(size, Config.num_scales), compression=Config.compression,
                                            dtype=np.float)
            dset_video_names = hf.create_dataset('video_names', shape=(size,), compression=Config.compression,
                                                 dtype=h5py.string_dtype(encoding='ascii'))
            dset_ir = hf.create_dataset('ir', shape=(size, len(Config.ir_list), 3), compression=Config.compression,
                                        dtype=np.float)
            dset_displacement = hf.create_dataset('wing_movements', shape=(size, Config.num_vertices_input, 3),
                                                  dtype=np.float)
            dset_cords = hf.create_dataset('cords', (1, Config.num_vertices_input, 3), dtype=np.float)
            plotter = pv.Plotter(off_screen=True)
            cords = get_cords_from_csv(next(self._csv_pair_generator())[0])
            dset_cords[0] = cords
            wing_model = FiniteElementWingModel(cords, Config.ir_list, Config.texture, Config.mesh_wing_path,
                                                Config.mesh_tip_path, Config.cameras, Config.num_of_vertices_wing,
                                                Config.num_of_vertices_tip, plotter, Config.resolution, Config.cmap)
            for idx, (points_path, scales_path) in enumerate(self._csv_pair_generator()):
                print(idx, points_path, scales_path)
                dset_displacement[idx], dset_scales[idx] = process_csv_pair(points_path, scales_path)
                dset_images[idx], dset_ir[idx] = wing_model(dset_displacement[idx])
                dset_video_names[idx] = f"{points_path.name}||{scales_path.name}"
            return db_name
