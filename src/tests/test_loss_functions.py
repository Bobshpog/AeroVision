from unittest import TestCase

import numpy as np

from src.data.matlab_reader import read_modal_shapes
from src.util.loss_functions import vertex_mean_rms


class Test(TestCase):
    def test_vertex_mean_rms(self):
        mode_shapes = read_modal_shapes('data/mode_shapes/synth_mode_shapes_9103_10.mat', 5)[:,:1,:]

        vertex_mean_rms(mode_shapes, 4, np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]))
