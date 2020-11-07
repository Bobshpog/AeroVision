from functools import partial
from unittest import TestCase

import h5py
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import src.util.image_transforms as my_transforms
from src.geometry.animations.synth_wing_animations import *
from src.model_datasets.image_dataset import ImageDataset
from src.models.resnet_synth import CustomInputResnet, run_resnet_synth


class TestCustomInputResnet(TestCase):

    def test_run_model(self):
        CHECKPOINT_PATH = ""
        mode_shape_path = "data/synt_data_mat_files/modes.mat"
        dset = "data/databases/20201016-232432__SyntheticMatGenerator(mesh_wing='synth_wing_v3.off', mesh_tip='fem_tip.off', resolution=[640, 480], texture_path='checkers_dark_blue.png'.hdf5"
        vid_path = "src/tests/temp/creation_of_modees.mp4"
        trash_path = "src/tests/temp/video_frames/"
        texture_path = "data/textures/checkers_dark_blue.png"
        frames = 1000
        num_of_scales = 5
        model = CustomInputResnet.load_from_checkpoint(CHECKPOINT_PATH)
        model.eval()

        with h5py.File(dset, 'r') as hf:
            mean_image = hf['generator metadata']['mean images'][()]
        out_transform = transforms.Compose([partial(my_transforms.mul_by_10_power, 4)])
        transform = my_transforms.top_middle_rgb(mean_image)
        val_dset = ImageDataset(dset,
                                transform=transform, out_transform=out_transform, cache_size=1000,
                                min_index=895)
        val_loader = DataLoader(val_dset, 1, shuffle=False, num_workers=0)
        scale1 = np.zeros((5, 2000))
        scale2 = np.zeros((5, 2000))
        name_of_picture = "depth_frameZ"
        for i, x, y in enumerate(val_loader):
            scale1[:, i] = y.numpy()
            scale2[:, i] = model(x).numpy()
            X = x + mean_image
            # todo reconstruct X
            cv2.imwrite(trash_path + name_of_picture + str(i) + ".png", np.asarray(X * 255, np.uint8))

        create_vid_by_scales(scale1, scale2, vid_path, trash_path, texture_path, mode_shape_path, frames, num_of_scales,
                             name_of_picture, show_ssim=True, res=[100, 400])

    def test_run_resnet_synth(self):
        BATCH_SIZE = None  # 16 for Resnet50, 64 for resnet 18
        NUM_EPOCHS = 1000
        VAL_CACHE_SIZE = 1000
        TRAIN_CACHE_SIZE = 5500  # around 6500 total images (640,480,3) total space
        NUM_INPUT_LAYERS = 1
        NUM_OUTPUTS = 5
        RESNET_TYPE = '18'  # '18', '50', '34'
        LOSS_FUNC = F.smooth_l1_loss
        EXPERIMENT_NAME = None
        TRAINING_DB_PATH = ""
        VALIDATION_DB_PATH = TRAINING_DB_PATH
        VAL_SPLIT = None
        TRANSFORM = my_transforms.top_middle_bw
        OUTPUT_SCALE = 1e4
        LEARNING_RATE = 1e-2
        WEIGTH_DECAY = 0
        COSINE_ANNEALING_STEPS = 10
        run_resnet_synth(NUM_INPUT_LAYERS, NUM_OUTPUTS, "test", TRAINING_DB_PATH, VALIDATION_DB_PATH, 895, TRANSFORM)
