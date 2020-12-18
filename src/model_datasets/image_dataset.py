import multiprocessing as mp

import h5py
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, hdf5_path, transform=None, output_scaling=None, out_transform=None, cache_size=0, min_index=0,
                 max_index=None,
                 index_list=None, dtype=np.float):
        """
        Initialization
        Args:
            hdf5_path: path to hdf5 database
            transform: transformation to be perforemed on the input
            out_transform: transformation to be performed on the output
            cache_size: size of cache
            min_index: minimum index, this is referencing the position of the index in the list of the possible indices
            max_index: maximum index, this is referencing the position of the index in the list of the possible indices
            index_list: list of possible indices to use, uses all if this is None
        """
        self.hdf5_path = hdf5_path
        self.hf = None
        self.transform = transform
        self.output_scaling = output_scaling
        self.out_transform = out_transform
        self.cache_size = cache_size
        self.cache_dict = mp.Manager().dict()
        self.min_index = min_index
        self.index_list = index_list
        self.dtype = dtype
        with h5py.File(self.hdf5_path, 'r') as hf:
            if max_index is None:
                if index_list:
                    max_index = len(index_list)
                else:
                    max_index = hf['data']['images'].len()
            self.database_len = min(hf['data']['images'].len(), max_index - min_index)

    def __getitem__(self, item):
        item += self.min_index
        if self.index_list:
            item = self.index_list[item]
        if item in self.cache_dict:
            image, scales = self.cache_dict[item]
        else:
            if self.hf is None:
                self.hf = h5py.File(self.hdf5_path, 'r')
            dataset = self.hf['data']
            transform = self.transform
            image = dataset['images'][item]
            scales = dataset['scales'][item]
            if transform:
                image = transform(image)
            if self.output_scaling:
                scales *= self.output_scaling

            if len(self.cache_dict) < self.cache_size:
                self.cache_dict[item] = image, scales
        scales = scales.astype(self.dtype)
        if self.out_transform:
            noisy_scales = self.out_transform(scales)
            return image.astype(self.dtype), scales,noisy_scales
        else:
            return image.astype(self.dtype), scales

    def __len__(self):
        return self.database_len

    # def __del__(self):
    #     if self.hf:
    #         self.hf.close()
