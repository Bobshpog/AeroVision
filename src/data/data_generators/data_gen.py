from abc import ABC, abstractmethod
from functools import lru_cache

import h5py
import numpy as np


class DataGenerator(ABC):
    """
    Abstract Data Generator base class
    """
    @lru_cache(1)
    def __len__(self) -> int:
        """
        Returns:
            Number of values in iterator
        """
        return len(list(self))

    @abstractmethod
    def save_metadata(self, hdf5: h5py.File, group_name: str) -> None:
        """
        Saves metadata about the database
        Args:
            hdf5: h5py object
            group_name:  name of the group
        """
        pass

    @abstractmethod
    def __iter__(self) -> (str, np.array, np.array):
        """
        Iterator that generates data
        Returns:
        (Video name, Coordinates np.array, output np.array)
        """
        pass
