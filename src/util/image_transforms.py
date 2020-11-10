from functools import partial

import cv2
import numpy as np
from torchvision.transforms import transforms


def slice_many_positions_no_depth(ids, input_photo):
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
    return (scale * x)


def scale_by(scale):
    return partial(_scale_by, scale)




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


def many_cameras_bw(camera_ids, mean_photos):
    mean_photos = slice_many_positions_no_depth(camera_ids, mean_photos)
    remove_mean = partial(remove_dc_photo, mean_photos)

    def many_cameras_slice_depth(photo):
        return slice_many_positions_no_depth(camera_ids, photo)
    return transforms.Compose([many_cameras_slice_depth,
                               remove_mean,
                               many_rgb_to_bw
                               ])


def single_camera_bw(camera_id, mean_photos):
    mean_photos = slice_many_positions_no_depth(camera_id, mean_photos)
    remove_mean = partial(remove_dc_photo, mean_photos)

    def slice_single_pos_no_depth(photo):
        return slice_many_positions_no_depth(camera_id,photo)
    return transforms.Compose([slice_single_pos_no_depth,
                               remove_mean,
                               single_rgb_to_bw
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
