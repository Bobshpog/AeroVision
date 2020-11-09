from unittest import TestCase

from src.data.database_analyzer import DatabaseAnalyzer


class TestDatabaseAnalyzer(TestCase):
    def test_calc_histogram(self):
        hdf_path="data/databases/20201016-232432__SyntheticMatGenerator(mesh_wing='synth_wing_v3.off', mesh_tip='fem_tip.off', resolution=[640, 480], texture_path='checkers_dark_blue.png'.hdf5"
        idx=2
        analyzer=DatabaseAnalyzer(hdf_path,100)
        analyzer.show_val_split(idx)
        bin_dict=analyzer.create_bin_dict(idx)
        pass