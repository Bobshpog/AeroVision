from unittest import TestCase

from src.data.matlab_reader import vertices_to_off, read_data


class Test(TestCase):
    def test_vertices_to_off(self):
        vertices_to_off('data/data_samples/Daniella_data.mat', 'data/wing_off_files/new_synth_wing.off')


    def test_read_data(self):
        ret=read_data('data/data_samples/Daniella_data.mat')
        pass
