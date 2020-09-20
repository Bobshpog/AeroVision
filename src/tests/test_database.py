from unittest import TestCase

import h5py

import src.data.database as db
from data.data_generators.synthetic_csv_gen import SyntheticCSVGenerator
from data.data_generators.synthetic_sin_decay_gen import SyntheticSineDecayingGen
from util.timing import profile


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

    @profile
    def test___call__(self):
        Config = self.Config
        data_generator_sin = SyntheticSineDecayingGen('data/synthetic_data_raw_samples', Config.mesh_wing_path,
                                                      Config.mesh_tip_path, 50, 20, Config.ir_list, Config.resolution,
                                                      Config.cameras, Config.texture, Config.cmap
                                                      )
        data_generator = SyntheticCSVGenerator('data/synthetic_data_raw_samples', Config.mesh_wing_path,
                                               Config.mesh_tip_path, Config.ir_list, Config.resolution, Config.cameras,
                                               Config.texture, Config.cmap)
        database = db.DatabaseBuilder(data_generator_sin, 'data/databases')
        data_file_path = database()
        with h5py.File(data_file_path, 'r') as f:
            print(list(f['data']['video_names']))
            pass
