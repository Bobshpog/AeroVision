import pickle
from functools import partial
from unittest import TestCase

import h5py
import torch
import torch.nn.functional as F

import src.util.image_transforms as my_transforms
from src.models.resnet_synth import run_resnet_synth
from src.util.loss_functions import L_infinity, reconstruction_loss_3d, y_hat_get_scale_i, y_get_scale_i


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

    def test_run_resnet_synth(self):
        BATCH_SIZE = 32  # 16 for Resnet50, 64 for resnet 18
        NUM_EPOCHS = 1000
        NUM_INPUT_LAYERS = 1
        NUM_OUTPUTS = 10
        RESNET_TYPE = '18'  # '18', '50', '34'
        LOSS_FUNC = F.smooth_l1_loss
        EXPERIMENT_NAME = None
        TRAINING_DB_PATH = ""
        VALIDATION_DB_PATH = TRAINING_DB_PATH
        with open('data/validation_splits/2/val_split.pkl', 'rb') as f:
            VAL_SPLIT = pickle.load(f)
        # Total possible cahce is around 6500 total images (640,480,3) total space
        VAL_CACHE_SIZE = len(VAL_SPLIT)
        TRAIN_CACHE_SIZE = 6500 * 3 - VAL_CACHE_SIZE
        TRANSFORM = my_transforms.top_middle_bw
        OUTPUT_SCALE = 1e4
        LEARNING_RATE = 1e-2
        WEIGTH_DECAY = 0
        COSINE_ANNEALING_STEPS = 10
        MAX_CAMERAS = 9
        NORMAL_CAMS = 6
        with h5py.File(TRAINING_DB_PATH, 'r') as hf:
            mean_image = hf['generator metadata']['mean images'][()]
            mode_shapes = hf['generator metadata']['modal shapes'][()]
            ir = hf['generator metadata'].attrs['ir'][()]
            _scales = hf['data']['scales'][()]
            scales_mean = _scales.mean(dim=0)
            scales_std = _scales.std(dim=0)
        transform = TRANSFORM(mean_image)
        reduce_dict = {'L_inf_mean': partial(L_infinity, mode_shapes, OUTPUT_SCALE),
                       'L_inf_max': (partial(L_infinity, mode_shapes, OUTPUT_SCALE), 'max'),
                       '3D_Worst_20%_mean': partial(reconstruction_loss_3d, torch.norm, mode_shapes[:, ir],
                                                    OUTPUT_SCALE),
                       '3D_Worst_20%_max': (
                           partial(reconstruction_loss_3d, torch.norm, mode_shapes[:, ir], OUTPUT_SCALE), 'max'),
                       '3D_mean': partial(reconstruction_loss_3d, torch.norm, mode_shapes,
                                          OUTPUT_SCALE),
                       }
        hist_dict = {f'scale{i}_real': partial(y_get_scale_i, OUTPUT_SCALE, scales_mean, scales_std, i) for i in
                     range(NUM_OUTPUTS)}
        hist_dict.update(
            {f'scale{i}_nn': partial(y_hat_get_scale_i, OUTPUT_SCALE, scales_mean, scales_std, i) for i in
             range(NUM_OUTPUTS)})
        text_dict = {'L_inf_max': (partial(L_infinity, mode_shapes, OUTPUT_SCALE), 5, BATCH_SIZE),
                     '3D_mean': (partial(reconstruction_loss_3d, torch.norm, mode_shapes,
                                         OUTPUT_SCALE), 5, BATCH_SIZE)
                     }
        run_resnet_synth(NUM_INPUT_LAYERS, NUM_OUTPUTS, "test", TRAINING_DB_PATH, VALIDATION_DB_PATH, VAL_SPLIT,
                         transform, reduce_dict, hist_dict, text_dict, train_cache_size=TRAIN_CACHE_SIZE,
                         val_cache_size=VAL_CACHE_SIZE,
                         batch_size=BATCH_SIZE)

    def test_run_resnet_synth_one_camera(self):
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
        TRANSFORM = my_transforms.single_camera_bw
        OUTPUT_SCALE = 1e4
        LEARNING_RATE = 1e-2
        WEIGTH_DECAY = 0
        COSINE_ANNEALING_STEPS = 10
        MAX_CAMERAS = 9
        NORMAL_CAMS = 6

        with h5py.File(TRAINING_DB_PATH, 'r') as hf:
            mean_image = hf['generator metadata']['mean images'][()]
        for i in range(MAX_CAMERAS):
            transform = TRANSFORM(i, mean_image)
            run_resnet_synth(NUM_INPUT_LAYERS, NUM_OUTPUTS, "first experiment one camera: " + str(i),
                             TRAINING_DB_PATH, VALIDATION_DB_PATH, VAL_SPLIT, transform)

    def test_run_resnet_synth_two_cameras(self):
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
        TRANSFORM = my_transforms.many_cameras_bw
        OUTPUT_SCALE = 1e4
        LEARNING_RATE = 1e-2
        WEIGTH_DECAY = 0
        COSINE_ANNEALING_STEPS = 10
        MAX_CAMERAS = 9
        NORMAL_CAMS = 6
        with h5py.File(TRAINING_DB_PATH, 'r') as hf:
            mean_image = hf['generator metadata']['mean images'][()]
        for i in range(NORMAL_CAMS):
            for j in range(i, NORMAL_CAMS):
                transform = TRANSFORM((i, j), mean_image)
                run_resnet_synth(NUM_INPUT_LAYERS, NUM_OUTPUTS,
                                 "first experiment one cameras: " + str(i) + ", " + str(j),
                                 TRAINING_DB_PATH, VALIDATION_DB_PATH, VAL_SPLIT, transform=transform)

    def test_run_resnet_synth_six_cameras(self):
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
        TRANSFORM = my_transforms.many_cameras_bw
        OUTPUT_SCALE = 1e4
        LEARNING_RATE = 1e-2
        WEIGTH_DECAY = 0
        COSINE_ANNEALING_STEPS = 10
        MAX_CAMERAS = 9
        NORMAL_CAMS = 6

        with h5py.File(TRAINING_DB_PATH, 'r') as hf:
            mean_image = hf['generator metadata']['mean images'][()]
        transform = TRANSFORM(list(range(6)), mean_image)
        run_resnet_synth(NUM_INPUT_LAYERS, NUM_OUTPUTS, "first experiment six cameras: ",
                         TRAINING_DB_PATH, VALIDATION_DB_PATH, VAL_SPLIT, transform)
