from functools import partial

import cv2
import numpy as np
from torchvision.transforms import transforms


def slice_many_positions_no_depth(input_photo, ids):
    return input_photo[ids, :, :, :3]


def slice_first_position_no_depth(input_photo):
    return input_photo[0, :, :, :3]


def slice_first_position_with_depth(input_photo):
    return input_photo[0, :, :, :]


def remove_dc_photo(dc_photo, input_photo):
    return input_photo - dc_photo


def last_axis_to_first(input_photo):
    return input_photo.transpose((2, 0, 1))


def double_to_float(input_photo):
    return input_photo.astype('float32')


def _scale_by(scale, x):
    return (scale * x).astype('float32')


def scale_by(scale):
    return partial(_scale_by, scale)


def mul_by_10_power(pow, x):
    return (10 ** pow * x).astype('float32')


def single_rgb_to_bw(img):
    return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=0)


def many_rgb_to_bw(img):
    return np.array([cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY) for i in range(img.shape[0])])


def top_middle_rgb(mean_photos):
    mean_photos = slice_first_position_no_depth(mean_photos)
    remove_mean = partial(remove_dc_photo, mean_photos)
    return transforms.Compose([slice_first_position_no_depth,
                               remove_mean,
                               last_axis_to_first])


def top_middle_bw(mean_photos):
    mean_photos = slice_first_position_no_depth(mean_photos)
    remove_mean = partial(remove_dc_photo, mean_photos)
    return transforms.Compose([slice_first_position_no_depth,
                               remove_mean,
                               single_rgb_to_bw
                               ])


def many_cameras_bw(mean_photos, camera_ids):
    mean_photos = slice_many_positions_no_depth(mean_photos, camera_ids)
    remove_mean = partial(remove_dc_photo, mean_photos)
    return transforms.Compose([slice_first_position_no_depth,
                               remove_mean,
                               many_rgb_to_bw
                               ])


def top_middle_rgbd(mean_photos):
    mean_photos = slice_first_position_with_depth(mean_photos)
    remove_mean = partial(remove_dc_photo, mean_photos)
    return transforms.Compose([slice_first_position_with_depth,
                               remove_mean,
                               last_axis_to_first
                               ])


def whiten_half_picture(img):
    img[:, int(img.shape[1] / 2), 0:3] = np.max(img[:, :, 0:3])
    return img
