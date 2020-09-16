from unittest import TestCase

import h5py

import src.data.database as db
from util.timing import profile


class Test(TestCase):
    def test_process_csv_pair(self):
        points, scales = db.process_csv_pair('data/synthetic_data_raw_samples/load1dice.csv',
                                             'data/synthetic_data_raw_samples/Load1Modal.csv')
        pass


class TestDatabaseBuilder(TestCase):
    def test__csv_pair_generator(self):
        print(list(db.DatabaseBuilder('data')._csv_pair_generator()))
        pass
    @profile
    def test___call__(self):
        database=db.DatabaseBuilder('data')
        data_file_path=database()
        with h5py.File(data_file_path,'r') as f:
            print(list(f['video_names']))
            pass

