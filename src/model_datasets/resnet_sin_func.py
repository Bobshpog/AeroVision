import h5py
from torch.utils.data import Dataset


class SinFunctionDataset(Dataset):
    def __init__(self, hdf5_path ):
        super.__init__()
        self.hf=h5py.File(hdf5_path, 'r')

    def __getitem__(self, item):
        hf=self.hf
        image=hf['data']['image'][item]
        scales = hf['data']['scales'][item]
        return image, scales
    def __len__(self):
        return self.hf['data']['images'].len()

    def __del__(self):
        self.hf.close()