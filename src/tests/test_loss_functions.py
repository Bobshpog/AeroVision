from unittest import TestCase

import numpy as np
from util import loss_functions
from src.geometry.numpy.wing_models import *
from src.data.matlab_reader import read_modal_shapes
from src.util.loss_functions import vertex_mean_rms


class Test(TestCase):
    def test_vertex_mean_rms(self):
        mode_shapes = read_modal_shapes('data/mode_shapes/synth_mode_shapes_9103_10.mat', 5)[:,:1,:]

        vertex_mean_rms(mode_shapes, 4, np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]))


    def test_different_3d_reconst(self):
        scale_factor = int(1e14)
        scales = matlab_reader.read_data("data/synt_data_mat_files/data2.mat")[2] * scale_factor
        zero_scale = np.zeros((scales.shape[0], 10))
        rad = SyntheticWingModel.radical_list_creation("data/wing_off_files/synth_wing_v5.off", 0.2)
        mode_shape = matlab_reader.read_modal_shapes("data/mode_shapes/synth_mode_shapes_9103_10.mat", 10)[:, rad]
        loss = loss_functions.reconstruction_loss_3d_new(torch.norm, mode_shape, scale_factor, scales, zero_scale)
        loss2 = loss_functions.reconstruction_loss_3d(torch.norm, mode_shape, scale_factor, scales, zero_scale)
        loss3 = np.zeros(loss.shape)
        for phase in trange(scales.shape[0]):
            loss3[phase] = loss[phase]/loss2[phase] # cant go wrong
        print("mean of loss: " + str(loss.mean()))
        print("mean of loss2: " + str(loss2.mean()))
        print("mean of loss3: " + str(loss3.mean()))
        print("var of loss: " + str(loss.std()))
        print("ver of loss2: " + str(loss2.std()))
        print("ver of loss3: " + str(loss3.var()))

