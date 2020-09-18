import h5py
from torch.utils.data import Dataset


class SinFunctionDataset(Dataset):
    def __init__(self, hdf5_path, transforms=None):
        """
        Initialization
        Args:
            hdf5_path: path to hdf5 database
            transforms: list of transforms to perform on the images
        """
        super.__init__()
        self.hf = h5py.File(hdf5_path, 'r')
        self.transforms = transforms

    def __getitem__(self, item):
        hf = self.hf
        transforms = self.transforms
        image = hf['data']['image'][item]
        scales = hf['data']['scales'][item]
        if isinstance(transforms, list):
            for transform in transforms:
                image = transform(image)
        return image, scales

    def __len__(self):
        return self.hf['data']['images'].len()

    def __del__(self):
        self.hf.close()
