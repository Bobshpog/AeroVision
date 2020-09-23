import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import h5py_cache as h5c
import numpy as np
from tqdm import tqdm

from data.data_generators.data_gen import DataGenerator


#
#
# class Config:
#     mesh_wing_path = 'data/wing_off_files/finished_fem_without_tip.off'
#     mesh_tip_path = 'data/wing_off_files/fem_tip.off'

#     num_scales = 11  # number of modal scales
#     num_vertices_input = 9067
#     compression = 'gzip'
#     num_vertices_wing = 7724
#     num_vertices_tip = 930

#     im_width = 640
#     im_height = 480
#     resolution = [im_width, im_height]
#     ir_list = list(range(28))
#     # cameras in pyvista format
#     cameras = [[(0.047, -0.053320266561896174, 0.026735639600027315),
#                 (0.05, 0.3, 0.02),
#                 (0, 0, 1)],
#
#                [(0.04581499400545182, -0.04477050005202985, -0.028567355483893577),
#                 (0.05, 0.3, 0.02),
#                 (0.001212842435223535, 0.13947688005070646, -1)],
#
#                [(0.11460619078012961, -0.04553696541254279, 0.038810512823530784),
#                 (0.05, 0.3, 0.02),
#                 (0, 0.16643488101070833, 1)],
#
#                [(0.11460619078012961, -0.04553696541254279, -0.038810512823530784),
#                 (0.05, 0.3, 0.02),
#                 (0, 0.16643488101070833, -1)],
#
#                [(-0.019770941905445285, -0.06082136750543311, 0.038694507832388224),
#                 (0.05, 0.3, 0.02),
#                 (0.041, 0.0438, 1)],
#
#                [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
#                 (0.05, 0.3, 0.02),
#                 (0.041, 0.0438, -1)]]
#     texture = 'data/textures/checkers2.png'
#     cmap = 'jet'
BATCH_SIZE = 50


@dataclass
class DatabaseBuilder:
    data_generator: DataGenerator
    db_folder: Union[str, Path] = field(repr=False)

    compression: str = 'gzip'

    def __post_init__(self):
        if isinstance(self.db_folder, str):
            self.db_folder = Path(self.db_folder)

        if not self.db_folder.exists():
            os.mkdir(self.db_folder)

    def __call__(self, db_name=None):
        """
       Creates a database from data in self.raw_data_folder
        Args:
            db_name: name of new database, defaults to generic scheme

        Returns:
            path to the newly created database
        """
        if db_name is None:
            db_name = self.db_folder / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}__{self.data_generator}.hdf5"
        num_datapoints = len(self.data_generator)
        num_scales, image_shape, num_ir = self.data_generator.get_data_sizes()

        with h5c.File(str(db_name.absolute()), 'w', chunk_cache_mem_size=500 * 1024 ** 2) as hf:

            self.data_generator.save_metadata(hf, 'generator metadata')
            data_grp = hf.create_group('data')
            dset_images = data_grp.create_dataset('images',
                                                  shape=(num_datapoints, *image_shape),
                                                  compression=self.compression,
                                                  dtype=np.float)
            dset_scales = data_grp.create_dataset('scales', shape=(num_datapoints, num_scales),
                                                  compression=self.compression,
                                                  dtype=np.float)
            dset_video_names = data_grp.create_dataset('video_names', shape=(num_datapoints,),
                                                       compression=self.compression,
                                                       dtype=h5py.string_dtype(encoding='ascii'))
            dset_ir = data_grp.create_dataset('ir', shape=(num_datapoints, num_ir, 3),
                                              compression=self.compression,
                                              dtype=np.float)

            progress_bar = tqdm(enumerate(self.data_generator), desc='Building Database', total=num_datapoints,
                                file=sys.stdout)
            names, images, ir, scales = [], [], [], []
            cache_idx = 0
            for idx, datapoint in progress_bar:
                # datapoint =(video_name, image, ir, scales)
                # print(idx, points_path, scales_path)
                names.append(datapoint[0]), images.append(datapoint[1]), ir.append(datapoint[2]), scales.append(
                    datapoint[3])
                if idx % BATCH_SIZE == 0 :
                    dset_video_names[cache_idx:idx+1] = names
                    dset_images[cache_idx:idx+1] = images
                    dset_ir[cache_idx:idx+1] = ir
                    dset_scales[cache_idx:idx+1] = scales
                    cache_idx = idx+1
                    names, images, ir, scales = [], [], [], []
                    pass

            dset_video_names[cache_idx:] = names
            dset_images[cache_idx:] = images
            dset_ir[cache_idx:] = ir
            dset_scales[cache_idx:] = scales
            return db_name
