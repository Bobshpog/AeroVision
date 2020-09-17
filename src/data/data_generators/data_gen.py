from abc import ABC, abstractmethod

import h5py
import numpy as np


class DataGenerator(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def save_metadata(self, hdf5: h5py.File, group_name:str) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> (str,np.array, np.array):
        """
        Iterator that generates data
        Returns:
        (Video name, Coordinates np.array, output np.array)
        """
        pass
