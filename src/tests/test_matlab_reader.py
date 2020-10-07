from unittest import TestCase

import numpy as np

from src.data.matlab_reader import vertices_to_off, read_data, read_modal_shapes


class Test(TestCase):
    def test_vertices_to_off(self):
        vertices_to_off('data/data_samples/Daniella_data.mat', 'data/wing_off_files/new_synth_wing.off')

    def test_read_data(self):
        ret = read_data('data/data_samples/Daniella_data.mat')
        pass


    def test_read_mode_shapes(self):
        d=read_modal_shapes("data/mode_shapes/synth_mode_shapes_9103_10.mat", 10)
        res=d*np.ones(d.shape[1:])
        pass
