from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import random
import h5py
import matplotlib.pyplot as plt
import numpy as np
from math import gcd

@dataclass
class DatabaseAnalyzer:
    hdf5_path: Union[str, Path]
    num_bins: int

    def __post_init__(self):
        with h5py.File(self.hdf5_path, 'r') as hf:
            self.scales = hf['data']['scales'][()]

    def _find_bin_edges(self, scale_id):
        """
        finds the bin edges of the chosen scale
        Args:
            scale_id:

        Returns:
            numbins sized ndarray of bin edges
        """
        scales = self.scales[:, scale_id]
        return np.linspace(scales.min(), scales.max(), self.num_bins + 1)

    def create_bin_dict(self, scale_id):
        """
        creates a dictionary where key is the id of the bin and value is all the indices of the values it contains
        Args:
            scale_id:

        Returns:
        A dictionary where key is the id of the bin and value is all the indices of the values it contains
        """
        bin_dict = defaultdict(list)
        digitized = np.digitize(self.scales[:, scale_id], self._find_bin_edges(scale_id))
        for idx, bin_idx in enumerate(digitized):
            bin_dict[bin_idx].append(idx)
        return dict(bin_dict)

    def find_val_split(self, q: float, start: Optional[int] = None, step_size: Optional[int] = None, allowed_err=0.1) -> tuple:
        """
        Finds all entries in database to be used for the validation split s.t. its size is ~q *num_bins
        Args:
            q: a number in range [0,1] that represents the % of entries in the validation set
            start: the start position where we begin the iteration
            step_size: a number that represents the step size of the iteration
            allowed_err: worst case = (q+allowed_err)*|scales in bins| defualt 0.1

        Returns:
            A tuple of indices of entries in the validation set
        """

        bins = self.create_bin_dict(0)
        bins_len = len(bins)
        total_scales = self.scales.shape[0]
        total_selected = 0
        to_return = tuple()

        if step_size is None:
            step_size = bins_len
            while gcd(step_size,bins_len) > 1:
                step_size = step_size + 1
        if start is None:
            start = random.randrange(0, bins_len)
        for i in range(bins_len):
            if total_selected > q * total_scales:
                return tuple(to_return)
            temp = total_selected + len(bins[(start + i * step_size) % bins_len])  # avoiding unnecessary computation
            if temp < (q+allowed_err) * total_scales:
                total_selected = temp
                to_return += tuple(bins[(start + i * step_size) % bins_len])

        return tuple(to_return)


def show_histogram(self, scale_id):
    bin_edges = self._find_bin_edges(scale_id)
    plt.hist(self.scales[:, scale_id], bin_edges)
    plt.title(f'Scale{scale_id} Distribution')
    plt.xlabel(f'scale{scale_id}')
    plt.ylabel('No of problems datapoints')
    plt.ticklabel_format(style="sci", scilimits=(0, 0), axis='x')
    plt.show()
    # ax = {}
    # fig, ((ax[0], ax[1], ax[2]), (ax[3], ax[4], _)) = plt.subplots(2, 3)
    # for i in range(5):
    #     ax[i].set_title(f'scale{i}')
    #     ax[i].hist(self.scales[:, i])
    # fig.show()
