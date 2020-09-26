from unittest import TestCase

from src.data.matlab_reader import vertices_to_off


class Test(TestCase):
    def test_vertices_to_off(self):
        vertices_to_off('data/data_samples/Daniella_data.mat', 'data/wing_off_files/new_synth_wing.off')
