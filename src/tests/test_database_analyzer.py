import pickle
from unittest import TestCase

from src.data.database_analyzer import DatabaseAnalyzer


class TestDatabaseAnalyzer(TestCase):
    def test_calc_histogram(self):
        hdf_path="data/databases/20201107-202226__SyntheticMatGenerator(mesh_wing='synth_wing_v5.off', mesh_tip='fem_tip.off', resolution=[640, 480], texture_path='checkers_dark_blue.png'.hdf5"
        idx=2
        analyzer=DatabaseAnalyzer(hdf_path,1024)
        l=list(analyzer.find_val_split(0.15))


        analyzer.show_val_split_histogram(l)
        with open('src/tests/temp/val_split.pkl','wb') as f:
            pickle.dump(l,f)
        bin_dict=analyzer.create_bin_dict(idx)
        pass