from abc import ABC, abstractmethod
from functools import lru_cache

import h5py
import numpy as np
from memoization import cached


class DataGenerator(ABC):
    """
    Abstract Data Generator base class
    """

    @cached(max_size=1)
    def __len__(self) -> int:
        """
        Returns:
            Number of values in iterator
        """
        length = sum(1 for _ in self)
        return length

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
    def __iter__(self) -> (str, np.array, np.array, np.array):
        """
        Iterator that generates data
        Returns:
        (Video name, Coordinates np.array,ir np.array, output np.array)
        """
        pass

    @abstractmethod
    def get_data_sizes(self) -> (int, int):
        """

        Returns:
            (num_vertices_input, num_scales,image_shape, num_ir)
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass