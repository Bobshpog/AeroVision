import pickle
from functools import partial
from unittest import TestCase

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import src.util.image_transforms as my_transforms
from src.geometry.pyvista_additions.parallel_plotter import RunTimeWingPlotter
from src.models.abstract_resnet import run_resnet_synth
from src.util.loss_functions import L_infinity, reconstruction_loss_3d, y_hat_get_scale_i, y_get_scale_i, \
    l1_norm_indexed, l1_norm


class TestCustomInputResnet(TestCase):
    #
    # def test_run_model(self):
    #     CHECKPOINT_PATH = ""
    #     mode_shape_path = "data/synt_data_mat_files/modes.mat"
    #     dset = "data/databases/20201016-232432__SyntheticMatGenerator(mesh_wing='synth_wing_v3.off', mesh_tip='fem_tip.off', resolution=[640, 480], texture_path='checkers_dark_blue.png'.hdf5"
    #     vid_path = "src/tests/temp/creation_of_modees.mp4"
    #     trash_path = "src/tests/temp/video_frames/"
    #     texture_path = "data/textures/checkers_dark_blue.png"
    #     frames = 1000
    #     num_of_scales = 5
    #     model = CustomInputResnet.load_from_checkpoint(CHECKPOINT_PATH)
    #     model.eval()
    #
    #     with h5py.File(dset, 'r') as hf:
    #         mean_image = hf['generator metadata']['mean images'][()]
    #     out_transform = transforms.Compose([partial(my_transforms.mul_by_10_power, 4)])
    #     transform = my_transforms.top_middle_rgb(mean_image)
    #     val_dset = ImageDataset(dset,
    #                             transform=transform, out_transform=out_transform, cache_size=1000,
    #                             min_index=895)
    #     val_loader = DataLoader(val_dset, 1, shuffle=False, num_workers=0)
    #     scale1 = np.zeros((5, 2000))
    #     scale2 = np.zeros((5, 2000))
    #     name_of_picture = "depth_frameZ"
    #     for i, x, y in enumerate(val_loader):
    #         scale1[:, i] = y.numpy()
    #         scale2[:, i] = model(x).numpy()
    #         X = x + mean_image
    #         # todo reconstruct X
    #         cv2.imwrite(trash_path + name_of_picture + str(i) + ".png", np.asarray(X * 255, np.uint8))

    # create_vid_by_scales(scale1, scale2, vid_path, trash_path, texture_path, mode_shape_path, frames, num_of_scales,
    #                     name_of_picture, show_ssim=True,res=[100,400])

    def test_resnet_noisy(self):
        NETWORK_CLASS = None  # CustomInputResnet or MultiResnet
        BATCH_SIZE = None  # 16 for Resnet50, 64 for resnet 18
        NUM_EPOCHS = 1000
        NUM_INPUT_LAYERS = 1
        NUM_OUTPUTS = 10
        RESNET_TYPE = ''  # 'res18', 'res50', 'res34', 'mobile2'
        LOSS_FUNC = F.smooth_l1_loss
        EXPERIMENT_NAME = None
        TRAINING_DB_PATH = ""
        VALIDATION_DB_PATH = TRAINING_DB_PATH
        EXTRA_PARAMS = None # need to supply  num_pictures, use_depth=False,latent_layer_size_per=128 for MultiResnet
        with open('data/validation_splits/2/val_split.pkl', 'rb') as f:
            VAL_SPLIT = pickle.load(f)
        # Total possible cahce is around 6500 total images (640,480,3) total space
        VAL_CACHE_SIZE = len(VAL_SPLIT)
        TRAIN_CACHE_SIZE = 6500 * 3 - VAL_CACHE_SIZE
        MONITOR = 'val_l1_smooth'

        TRANSFORM = my_transforms.TranformSingleNoisyBW
        OUTPUT_SCALE = 1e4
        LEARNING_RATE = 1e-2
        WEIGTH_DECAY = 0
        COSINE_ANNEALING_STEPS = 10
        MAX_CAMERAS = 9
        NORMAL_CAMS = 6
        MM_IN_METER = 1e-3

        POISSON_RATE = None
        GAUSS_MEAN = 0
        GAUSS_VAR = None
        SP_RATE = None

        with h5py.File(TRAINING_DB_PATH, 'r') as hf:
            metadata = hf['generator metadata']
            attrs = metadata.attrs
            mean_images = metadata['mean images'][()]
            mode_shapes = metadata['modal shapes'][()]
            ir = metadata.attrs['ir'][()]
            _scales = hf['data']['scales'][()]
            scales_mean = _scales.mean(axis=0)
            scales_std = _scales.std(axis=0)
            mean_image = mean_images[0]
            wing_path = 'data/wing_off_files/' + attrs['mesh_wing_path']
            tip_path = 'data/wing_off_files/' + attrs['mesh_tip_path']
            resolution = attrs['resolution']
            camera = attrs['cameras']
            texture = 'data/textures/' + attrs['texture']
        parallel_plotter = RunTimeWingPlotter(mean_photo=mean_image[:, :, :NUM_INPUT_LAYERS],
                                              texture=texture, cam_location=camera,
                                              mode_shapes=mode_shapes, wing_path=wing_path, tip_path=tip_path,
                                              ir_index=ir, output_scaling=OUTPUT_SCALE, rgb=(NUM_INPUT_LAYERS != 1))

        transform = TRANSFORM(mean_images, POISSON_RATE, GAUSS_MEAN, GAUSS_VAR, SP_RATE)
        reduce_dict = {'L_inf_mean_loss': (partial(L_infinity, mode_shapes[:, ir], OUTPUT_SCALE), MM_IN_METER, 'mean'),
                       'L_inf_max_loss': (partial(L_infinity, mode_shapes[:, ir], OUTPUT_SCALE), MM_IN_METER, 'max'),
                       '3D_20%_mean_loss': (partial(reconstruction_loss_3d, torch.norm, mode_shapes[:, ir],
                                                    OUTPUT_SCALE), MM_IN_METER, 'mean'),
                       '3D_20%_max_loss': (
                           partial(reconstruction_loss_3d, torch.norm, mode_shapes[:, ir], OUTPUT_SCALE),
                           MM_IN_METER, 'max'),
                       '3D_100%': (partial(reconstruction_loss_3d, torch.norm, mode_shapes,
                                           OUTPUT_SCALE), MM_IN_METER, 'mean'),
                       'L1_regression': (lambda y_hat, y: l1_norm(y_hat, y).mean(dim=-1)),
                       'l1_smooth': (lambda y_hat, y: F.smooth_l1_loss(y_hat, y, reduction='none').mean(dim=-1))
                       }
        for i in range(NUM_OUTPUTS):
            reduce_dict[f'l1_scale{i}_regression'] = (partial(l1_norm_indexed, OUTPUT_SCALE, i), 'mean')

        hist_dict = {f'scale{i}_real': partial(y_get_scale_i, OUTPUT_SCALE, scales_mean, scales_std, i) for i in
                     range(NUM_OUTPUTS)}
        hist_dict.update(
            {f'scale{i}_nn': partial(y_hat_get_scale_i, OUTPUT_SCALE, scales_mean, scales_std, i) for i in
             range(NUM_OUTPUTS)})
        text_dict = {'L_inf_max': (partial(L_infinity, mode_shapes, OUTPUT_SCALE), 100, BATCH_SIZE),
                     '3D_mean': (partial(reconstruction_loss_3d, torch.norm, mode_shapes,
                                         OUTPUT_SCALE), 100, BATCH_SIZE)
                     }

        # for i in -np.linspace(-3, 0, 10, endpoint=False):
        #     POISSON_RATE = i*i
        #     EXPERIMENT_NAME=f"noisy pois={i*i}"
        # transform = TRANSFORM(mean_image, POISSON_RATE, GAUSS_MEAN, GAUSS_VAR, SP_RATE)
        run_resnet_synth(NETWORK_CLASS, NUM_INPUT_LAYERS, NUM_OUTPUTS, EXPERIMENT_NAME, TRAINING_DB_PATH,
                         VALIDATION_DB_PATH, VAL_SPLIT, transform, None, reduce_dict, hist_dict, text_dict,
                         resnet_type=RESNET_TYPE,
                         train_cache_size=TRAIN_CACHE_SIZE,
                         val_cache_size=VAL_CACHE_SIZE,
                         batch_size=BATCH_SIZE, subsampler_size=len(VAL_SPLIT), output_scaling=OUTPUT_SCALE,
                         monitor_metric_name=MONITOR, parallel_plotter=parallel_plotter, extra_params=EXTRA_PARAMS)
        parallel_plotter.finalize()
