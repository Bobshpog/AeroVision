import h5py
from torch.utils.data import Dataset
import multiprocessing as mp


class ImageDataset(Dataset):
    def __init__(self, hdf5_path, transform=None, out_transform=None, cache_size=0, min_index=0, max_index=None):
        """
        Initialization
        Args:
            hdf5_path: path to hdf5 database
            transform: list of transforms to perform on the images
        """
        self.hdf5_path = hdf5_path
        self.hf = None
        self.transform = transform
        self.out_transform = out_transform
        self.cache_size = cache_size
        self.cache_dict = mp.Manager().dict()
        self.min_index = min_index
        with h5py.File(self.hdf5_path, 'r') as hf:
            if max_index is None:
                max_index = hf['data']['images'].len()
            self.database_len = min(hf['data']['images'].len(), max_index - min_index)

    def __getitem__(self, item):
        item += self.min_index
        if item in self.cache_dict:
            return self.cache_dict[item]

        if self.hf is None:
            self.hf = h5py.File(self.hdf5_path, 'r')
        dataset = self.hf['data']
        transform = self.transform
        out_transform = self.out_transform
        image = dataset['images'][item]
        scales = dataset['scales'][item]
        if transform:
            image = transform(image)
        if out_transform:
            scales = out_transform(scales)

        if len(self.cache_dict) < self.cache_size:
            self.cache_dict[item] = image, scales
        return image, scales

    def __len__(self):
        return self.database_len

    # def __del__(self):
    #     if self.hf:
    #         self.hf.close()
