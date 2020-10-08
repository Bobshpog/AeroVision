from unittest import TestCase
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model_datasets.image_dataset import ImageDataset
from src.models.resnet_synth import CustomInputResnet
import src.util.image_transforms as my_transforms
import h5py
from src.geometry.animations.synth_wing_animations import *
class TestCustomInputResnet(TestCase):

    def test_run_model(self):
        CHECKPOINT_PATH = ""
        mode_shape_path = "data/synt_data_mat_files/modes.mat"
        vid_path = "src/tests/temp/creation_of_modees.mp4"
        trash_path = "src/tests/temp/video_frames/"
        texture_path = "data/textures/checkers_dark_blue.png"
        frames = 1000
        num_of_scales = 5
        model = CustomInputResnet.load_from_checkpoint(CHECKPOINT_PATH)
        model.eval()

        with h5py.File("data/databases/20201007-192101__SyntheticMatGenerator(mesh_wing='synth_wing_v3.off', mesh_tip='fem_tip.off', resolution=[640, 480], texture_path='checkers_dark_blue.png'.hdf5", 'r') as hf:
            mean_image = hf['generator metadata']['mean images'][()]
        out_transform = transforms.Compose([partial(my_transforms.mul_by_10_power, 4)])
        transform = my_transforms.top_middle_rgb(mean_image)
        val_dset = ImageDataset("data/databases/20201007-192101__SyntheticMatGenerator(mesh_wing='synth_wing_v3.off', mesh_tip='fem_tip.off', resolution=[640, 480], texture_path='checkers_dark_blue.png'.hdf5",
                                transform=transform, out_transform=out_transform, cache_size=1000,
                                min_index=895)
        val_loader = DataLoader(val_dset, 1, shuffle=False, num_workers=0)
        scale1 = np.zeros((5,2000))
        scale2 = np.zeros((5, 2000))
        name_of_picture = "depth_frameZ"
        for i,x,y in enumerate(val_loader):
            scale1[:,i] = y.numpy()
            scale2[:,i] = model(x).numpy()
            X = x + mean_image
            #todo reconstruct X
            cv2.imwrite(trash_path + name_of_picture + str(i) + ".png", np.asarray(X * 255, np.uint8))

        create_vid_by_scales(scale1, scale2, vid_path, trash_path, texture_path, mode_shape_path, frames, num_of_scales,
                             name_of_picture, show_ssim=True)


