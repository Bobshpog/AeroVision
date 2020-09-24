import h5py
from torch.utils.data import Dataset


class SinFunctionDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        """
        Initialization
        Args:
            hdf5_path: path to hdf5 database
            transform: list of transforms to perform on the images
        """
        self.hdf5_path = hdf5_path
        self.hf = None
        self.transform = transform
        with h5py.File(self.hdf5_path, 'r') as hf:
            self.database_len = hf['data']['images'].len()

    def __getitem__(self, item):
        if self.hf is None:
            self.hf = h5py.File(self.hdf5_path, 'r')
        dataset = self.hf['data']
        transform = self.transform
        image = dataset['images'][item]
        scales = dataset['scales'][item]
        if transform:
            image = transform(image)
        return image, scales

    def __len__(self):
        return self.database_len

    # def __del__(self):
    #     if self.hf:
    #         self.hf.close()
