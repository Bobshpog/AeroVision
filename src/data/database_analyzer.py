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
        d = dict(bin_dict)
        return d

    def find_val_split(self, q: float, position=None, step_size=None,
                       allowed_err=0.0001) -> tuple:
        """
        Finds all entries in database to be used for the validation split s.t. its size is ~q *num_bins
        Args:
            q: a number in range [0,1] that represents the % of entries in the validation set
            position: the start position where we begin the iteration
            step_size: a number that represents the step size of the iteration
            allowed_err: worst case = (q+allowed_err)*|scales in bins| defualt 0.0001
        Returns:
            A tuple of indices of entries in the validation set
        """

        bins = self.create_bin_dict(0)
        bins_len = self.num_bins
        total_scales = self.scales.shape[0]
        total_selected = 0
        to_return = tuple()
        if step_size is None:
            step_size=bins_len
            while gcd(step_size, bins_len) > 1:
                step_size = random.randint(1, bins_len)
        if position is None:
            position = random.randrange(0, bins_len)
        for i in range(bins_len):
            if total_selected > q * total_scales:
                return tuple(to_return)
            position= (position + step_size) % bins_len
            if bins.get(position) is not None:
                possible_selection = total_selected + len(bins[position])
                if possible_selection < (q + allowed_err) * total_scales:
                    total_selected = possible_selection
                    to_return += tuple(bins[position % bins_len])
        raise ValueError("couldn't satisfy condition")

    def show_val_split_histogram(self):
        arr = list(self.find_val_split(q=0.15))
        rest = list(set(range(self.scales.shape[0])) - set(arr))
        print("common items: ", list(set(arr).intersection(rest)))
        print("length = " + str(len(arr)))
        p_real = 100* len(arr) / self.scales.shape[0]
        print(f"%:  {p_real:.2f}")
        bin_edges = self._find_bin_edges(0)
        s = self.scales[:, 0]
        plt.hist(s[arr], bin_edges, label="validation")
        plt.hist(s[rest], bin_edges, label="training")
        plt.legend()
        plt.title(f'histogram {p_real:.2f} of items')
        plt.xlabel(f'scale{0}')
        plt.ylabel('No of problems datapoints')
        plt.ticklabel_format(style="sci", scilimits=(0, 0), axis='x')
        plt.show()

    def show_histogram(self, scale_id):
        bin_edges = self._find_bin_edges(scale_id)
        plt.hist(self.scales[:, scale_id], bin_edges)
        plt.title(f'Scale{scale_id} Distribution')
        plt.xlabel(f'scale{scale_id}')
        plt.ylabel('No of problems datapoints')
        plt.ticklabel_format(style="sci", scilimits=(0, 0), axis='x')
        plt.show()

#def show_histogram(self, scale_id):
#    bin_edges = self._find_bin_edges(scale_id)
#    plt.hist(self.scales[:, scale_id], bin_edges)
#    plt.title(f'Scale{scale_id} Distribution')
#    plt.xlabel(f'scale{scale_id}')
#    plt.ylabel('No of problems datapoints')
#    plt.ticklabel_format(style="sci", scilimits=(0, 0), axis='x')
#    plt.show()

    # ax = {}
    # fig, ((ax[0], ax[1], ax[2]), (ax[3], ax[4], _)) = plt.subplots(2, 3)
    # for i in range(5):
    #     ax[i].set_title(f'scale{i}')
    #     ax[i].hist(self.scales[:, i])
    # fig.show()
