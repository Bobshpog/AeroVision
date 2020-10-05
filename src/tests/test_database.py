import time
from random import randint
from unittest import TestCase

import cv2
import h5py
import numpy as np
from tqdm import trange

import src.data.database as db
from src.data.data_generators.synthetic_csv_gen import SyntheticCSVGenerator
from src.data.data_generators.synthetic_mat_gen import SyntheticMatGenerator
from src.data.data_generators.synthetic_sin_decay_gen import SyntheticSineDecayingGen
from src.util.timing import profile


class Test(TestCase):
    def test_process_csv_pair(self):
        points, scales = db.process_csv_pair('data/synthetic_data_raw_samples/load1dice.csv',
                                             'data/synthetic_data_raw_samples/Load1Modal.csv')
        pass


class TestDatabaseBuilder(TestCase):
    class Config:
        mesh_wing_path = 'data/wing_off_files/finished_fem_without_tip.off'
        mesh_tip_path = 'data/wing_off_files/fem_tip.off'

        num_scales = 11  # number of modal scales
        num_vertices_input = 9067
        compression = 'gzip'
        num_vertices_wing = 7724
        num_vertices_tip = 930

        im_width = 640
        im_height = 480
        resolution = [im_width, im_height]
        ir_list = ids = [6419, 6756, 7033, 7333, 7635, 7937, 8239, 8541, 8841,  # first line
                         6411, 6727, 7025, 7325, 7627, 7929, 8271, 8553, 8854,  # middle
                         6361, 6697, 6974, 7315, 7576, 7919, 8199, 8482, 8782]
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

    @profile
    def test___call__(self):
        Config = self.Config
        data_generator_sin = SyntheticSineDecayingGen('data/synthetic_data_raw_samples', Config.mesh_wing_path,
                                                      Config.mesh_tip_path, 5, 2, Config.ir_list, Config.resolution,
                                                      Config.cameras, Config.texture, Config.cmap
                                                      )
        data_generator = SyntheticMatGenerator('data/data_samples/Daniella_data.mat', Config.mesh_wing_path,
                                               Config.mesh_tip_path, Config.ir_list, Config.resolution, Config.cameras,
                                               Config.texture, Config.cmap)
        database = db.DatabaseBuilder(data_generator, 'data/databases', batch_size=64)
        data_file_path = database()
        with h5py.File(data_file_path, 'r') as f:
            print(list(f['data']['video_names']))
            pass

    @profile
    def test_read_hdf5(self):
        TRAINING_DB_PATH = "data/databases/20200923-215518__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
        VALIDATION_DB_PATH = "data/databases/20200924-184304__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
        with h5py.File(VALIDATION_DB_PATH, 'r') as hf:
            dset = hf['data']['images']
            time.perf_counter()
            images = dset[:, 0, :, :, 0]
            index = randint(images.shape[0])
            cv2.imshow("photo", images[index])
            pass

    @profile
    def test_read_py(self):
        for i in trange(1000, 1060):
            x = np.load(f'data/images/{i}.npy')
