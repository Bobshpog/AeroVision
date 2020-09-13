import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

from geometry.numpy.mesh import Mesh


def process_csv(file):
    pass #TODO implement


class Config:
    mesh_path = 'data/wing_off_files/combined_wing.off'
    im_width = 640
    im_height = 480
    resolution = [im_width, im_height]
    num_scales = 6
    num_vertices_input = -1  # TODO
    compression = 'gzip'
    ir_count = 28
    cameras = []  # TODO, insert camera angles
    texture = 'data/textures/checkers2.png'


class DatabaseBuilder:

    def __init__(self, data_folder):
        self.root_data_folder = Path(data_folder)
        self.synthetic_data_folder = Path(self.root_data_folder / "synthetic data")
        self.db_folder = Path(data_folder / 'databases')
        if not self.db_folder.exists():
            os.mkdir(self.db_folder)

    def __call__(self, output_path=None):
        if output_path is None:
            output_path = self.db_folder / f"{str(datetime.now())}.hdf5"  # TODO fix path with relevant metadata
        video_dirs = next(os.walk(self.synthetic_data_folder))[1]
        size = len(video_dirs)
        with h5py.File(output_path, 'w') as hf:
            dset_images = hf.create_dataset('images',
                                            shape=(size, Config.num_angles, Config.im_width, Config.im_height, 4),
                                            compression=Config.compression,
                                            dtype=np.float)  # TODO make sure image is in float format
            dset_scales = hf.create_dataset('scales', shape=(size, Config.num_scales), compression=Config.compression,
                                            dtype=np.float)
            dset_video_names = hf.create_dataset('video_names', shape=(size,), compression=Config.compression,
                                                 dtype=h5py.string_dtype(encoding='ascii'))
            dset_ir = hf.create_dataset('ir', shape=(size, Config.ir_count, 3), compression=Config.compression,
                                        dtype=np.float)
            dset_movements = hf.create_dataset('movements', shape=(size, Config.num_vertices_input, 3), dtype=np.float)
            mesh = Mesh(Config.mesh_path)
            plotter = pv.Plotter()
            for folder_name in video_dirs:
                folder_files = next(os.walk(folder_name))[2]
                for idx, csvfile in enumerate(
                        folder_files):  # TODO make sure every file includes both measurements and scales
                    # extract a single movement from csv
                    movement, scales = process_csv(csvfile)
                    dset_movements[idx] = movement
                    movement_reconstructed = mesh.get_movement_reconstructed(movement)
                    dset_scales[idx] = scales
                    dset_ir[idx] = mesh.get_ir_coords(movement_reconstructed)
                    dset_video_names[idx] = folder_name
                    for j, camera in enumerate(Config.cameras):
                        dset_images[idx, j] = mesh.get_photo(movement_reconstructed, resolution=Config.resolution,
                                                             texture=Config.texture, camera=camera, plotter=plotter)
