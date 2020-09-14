import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

from geometry.numpy.mesh import Mesh
from geometry.numpy.wing_models import FemWing

def process_csv(file):
    pass  # TODO implement


class Config:
    mesh_hull_path = 'data/wing_off_files/finished_fem_without_tip.off'
    mesh_tip_path = 'data/wing_off_files/fem_tip.off'
    im_width = 640
    im_height = 480
    resolution = [im_width, im_height]
    num_scales = 11  # number of modal scales
    num_vertices_input = 9649
    compression = 'gzip'
    ir_count = 28
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


class DatabaseBuilder:

    def __init__(self, data_folder):
        self.root_data_folder = Path(data_folder)
        self.synthetic_data_folder = Path(self.root_data_folder / "synthetic_data")
        self.db_folder = Path(data_folder / 'databases')
        if not self.db_folder.exists():
            os.mkdir(self.db_folder)

    def __call__(self, output_path=None):
        if output_path is None:
            texture_name=Config.texture.split('/')[-1]
            mesh_name=Config.mesh_path.split('/')[-1]
            output_path = self.db_folder / f"{str(datetime.now())}||{Config.cameras}||{mesh_name}||{texture_name}.hdf5"  # TODO fix path with relevant metadata
        video_dirs = next(os.walk(self.synthetic_data_folder))[1]
        size = len(video_dirs)
        with h5py.File(output_path, 'w') as hf:
            dset_images = hf.create_dataset('images',
                                            shape=(size, len(Config.cameras), Config.im_width, Config.im_height, 4),
                                            compression=Config.compression,
                                            dtype=np.float)  # TODO make sure image is in float format
            dset_scales = hf.create_dataset('scales', shape=(size, Config.num_scales), compression=Config.compression,
                                            dtype=np.float)
            dset_video_names = hf.create_dataset('video_names', shape=(size,), compression=Config.compression,
                                                 dtype=h5py.string_dtype(encoding='ascii'))
            dset_ir = hf.create_dataset('ir', shape=(size, Config.ir_count, 3), compression=Config.compression,
                                        dtype=np.float)
            dset_movements = hf.create_dataset('movements', shape=(size, Config.num_vertices_input, 3), dtype=np.float)
            mesh_wing_hull=Mesh(Config.mesh_hull_path)
            mesh_wing_tip=Mesh(Config.mesh_tip_path)
            plotter = pv.Plotter()
            for folder_name in video_dirs:
                folder_files = next(os.walk(folder_name))[2]
                for idx, csvfile in enumerate(
                        folder_files):  # TODO make sure every file includes both measurements and scales
                    # extract a single movement from csv
                    movement, scales = process_csv(csvfile)
                    dset_movements[idx] = movement
                    dset_scales[idx] = scales

                    dset_ir[idx] = FemWing.get_ir_coords()
                    dset_video_names[idx] = folder_name
                    for j, camera in enumerate(Config.cameras):
                        dset_images[idx, j] = mesh.get_photo(movement_reconstructed, resolution=Config.resolution,
                                                             texture=Config.texture, camera=camera, plotter=plotter)
