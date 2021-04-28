from functools import partial

import cv2
import numpy as np
from torchvision.transforms import transforms
import pyvista as pv

from src.util.depth2hha.getHHA import getHHA


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
    return np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in img])[:, np.newaxis, :, :]


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
        return slice_many_positions_no_depth(camera_id, photo)

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


class TransformScaleBy:

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, x):
        return x * self.scale_factor

    def __repr__(self):
        return "SCALE_BY_TRANSFORM_SCALE_FACTOR_" + f'{self.scale_factor: .3f}'


class TransformManyPositionsNoDepth:
    def __init__(self, cam_id):
        self.cam_id = cam_id

    def __call__(self, img):
        return img[self.cam_id, :, :, :3]

    def __repr__(self):
        if isinstance(self.cam_id, int):
            return "MANY_POSITION_NO_DEPTH_TRANSFORM_CAM_ID" + str(self.cam_id)
        to_ret = "MANY_POSITION_NO_DEPTH_TRANSFORM_CAM_ID_("
        for i in self.cam_id:
            to_ret += str(i) + ","
        return to_ret + ")"


class TransformManyPositionsOnlyDepth:
    def __init__(self, cam_id):
        self.cam_id = cam_id

    def __call__(self, img):
        return img[self.cam_id, :, :, -1]

    def __repr__(self):
        if isinstance(self.cam_id, int):
            return "MANY_POSITION_ONLY_DEPTH_TRANSFORM_CAM_ID" + str(self.cam_id)
        to_ret = "MANY_POSITION_ONLY_DEPTH_TRANSFORM_CAM_ID_("
        for i in self.cam_id:
            to_ret += str(i) + ","
        return to_ret + ")"


class TransformManyPositionsHHA:
    def __init__(self, cam_id, camera_pos):
        self.cam_id = cam_id
        p = pv.Plotter()
        p.add_mesh(pv.Sphere())
        p.camera_position = camera_pos
        # Now grab the matrix
        vmtx = p.camera.GetModelViewTransformMatrix()
        mtx = pv.trans_from_matrix(vmtx)
        self.camera_matrix = mtx[:-1, :-1]

    def __call__(self, img):
        return last_axis_to_first(getHHA(self.camera_matrix, img[self.cam_id, :, :, -1], 0))

    def __repr__(self):
        if isinstance(self.cam_id, int):
            return "MANY_POSITION_HHA_TRANSFORM_CAM_ID" + str(self.cam_id)
        to_ret = "MANY_POSITION_HHA_TRANSFORM_CAM_ID_("
        for i in self.cam_id:
            to_ret += str(i) + ","
        return to_ret + ")"


class TransformManyPositions:
    def __init__(self, cam_id):
        self.cam_id = cam_id

    def __call__(self, img):
        return img[self.cam_id]

    def __repr__(self):
        if isinstance(self.cam_id, int):
            return "MANY_POSITION_TRANSFORM_CAM_ID" + str(self.cam_id)
        to_ret = "MANY_POSITION_TRANSFORM_CAM_ID_("
        for i in self.cam_id:
            to_ret += str(i) + ","
        return to_ret + ")"


class TransformRemoveDcPhoto:
    def __init__(self, dc_photo):
        self.dc_photo = dc_photo

    def __call__(self, img):
        return img - self.dc_photo

    def __repr__(self):
        return "REMOVE_DC_PHOTO_TRANSFORM"


class TransformSingleCameraBw:
    def __init__(self, cam_id, mean_photo):
        self.cam_id = cam_id
        many_pos_trans = TransformManyPositionsNoDepth(cam_id)
        mean_photo = many_pos_trans(mean_photo)
        remove_dc_trans = TransformRemoveDcPhoto(mean_photo)
        self.transform = transforms.Compose([many_pos_trans,
                                             remove_dc_trans,
                                             single_rgb_to_bw
                                             ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        return "SINGLE_CAMERA_BW_TRANSFORM_CAM_ID_" + str(self.cam_id)


class TransformSingleCameraRGB:
    def __init__(self, cam_id, mean_photo):
        self.cam_id = cam_id
        many_pos_trans = TransformManyPositionsNoDepth(cam_id)
        mean_photo = many_pos_trans(mean_photo)
        remove_dc_trans = TransformRemoveDcPhoto(mean_photo)
        self.transform = transforms.Compose([many_pos_trans,
                                             remove_dc_trans, last_axis_to_first
                                             ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        return "SINGLE_CAMERA_RGB_TRANSFORM_CAM_ID_" + str(self.cam_id)


class TransformSingleCameraRGBD:
    def __init__(self, cam_id, mean_photo):
        self.cam_id = cam_id
        many_pos_trans = TransformManyPositions(cam_id)
        mean_photo = many_pos_trans(mean_photo)
        remove_dc_trans = TransformRemoveDcPhoto(mean_photo)
        self.transform = transforms.Compose([many_pos_trans,
                                             remove_dc_trans, last_axis_to_first
                                             ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        return "SINGLE_CAMERA_RGBD_TRANSFORM_CAM_ID_" + str(self.cam_id)


class TransformSingleCameraDepth:
    def __init__(self, cam_id, mean_photo):
        self.cam_id = cam_id
        many_pos_trans = TransformManyPositionsOnlyDepth(cam_id)
        mean_photo = many_pos_trans(mean_photo)
        remove_dc_trans = TransformRemoveDcPhoto(mean_photo)
        self.transform = transforms.Compose([many_pos_trans,
                                             remove_dc_trans
                                             ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        return "SINGLE_CAMERA_DEPTH_TRANSFORM_CAM_ID_" + str(self.cam_id)


class TransformManyCameraBw:
    def __init__(self, cam_id, mean_photo):
        self.cam_id = cam_id
        many_pos_trans = TransformManyPositionsNoDepth(cam_id)
        mean_photo = many_pos_trans(mean_photo)
        remove_dc_trans = TransformRemoveDcPhoto(mean_photo)
        self.transform = transforms.Compose([many_pos_trans,
                                             remove_dc_trans,
                                             many_rgb_to_bw
                                             ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        to_ret = "MANY_CAMERA_BW_TRANSFORM_CAM_ID_("
        for i in self.cam_id:
            to_ret += str(i) + ","
        return to_ret + ")"


class TransformManyCamerasRGBD:
    def __init__(self, cam_id, mean_photo):
        self.cam_id = cam_id
        many_pos_trans = TransformManyPositions(cam_id)
        mean_photo = many_pos_trans(mean_photo)
        remove_dc_trans = TransformRemoveDcPhoto(mean_photo)
        self.transform = transforms.Compose([many_pos_trans,
                                             remove_dc_trans,
                                             many_rgb_to_bw
                                             ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        to_ret = "MANY_CAMERA_RGBD_TRANSFORM_CAM_ID_("
        for i in self.cam_id:
            to_ret += str(i) + ","
        return to_ret + ")"


class TranformPoissonNoise:
    def __init__(self, lamda):
        self.lamda = lamda

    def __call__(self, img):
        noise = np.random.poisson(self.lamda, size=img.shape)
        if img.shape[-1] == 4:
            if (len(img.shape) == 4):
                noise[:, :, :, 3] = 0
            elif (len(img.shape == 3)):
                noise[:, :, 3] = 0
            else:
                raise NotImplementedError
        return img + noise

    def __repr__(self):
        return "POISSON_NOISE_TRANSFORM_LAMBDA:_" + str(self.lamda)


class TransformSaltAndPeper:
    def __init__(self, amount, s_vs_p=0.5):  # for 1 cam and n channels: img.shape = [h,w,channels]
        self.amount = amount
        self.s_vs_p = s_vs_p

    def __call__(self, img):
        out = np.copy(img)
        num_salt = np.ceil(self.amount * img.size * self.s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape[1:]]
        out[0][tuple(coords)] = 1
        num_pepper = np.ceil(self.amount * img.size * (1. - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape[1:]]
        out[0][tuple(coords)] = 0
        return out

    def __repr__(self):
        return "S&P_NOISE_TRANFORM_SV:" + str(self.s_vs_p) + "_AMOUNT:_" + str(self.amount)


class TransformGaussian:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.std, size=img.shape)
        if img.shape[-1] == 4:
            if (len(img.shape) == 4):
                noise[:, :, :, 3] = 0
            elif (len(img.shape == 3)):
                noise[:, :, 3] = 0
            else:
                raise NotImplementedError
        return img + noise

    def __repr__(self):
        return "GAUSSIAN_NOISE_TRANFORM_MEAN:" + str(self.mean) + "STD:_" + str(self.std)


class TranformSingleNoisyBW:
    def __init__(self, mean_photo, pois_lamda, gauss_mean, gauss_var, salt_peper_amount, salt_pepper_ratio=0.5,
                 cam_pos=0):
        #   (defult into up middle)
        self.tform = []
        self.tform.append(TransformSingleCameraBw(cam_pos, mean_photo))
        if gauss_var:
            self.tform.append(TransformGaussian(gauss_mean, gauss_var))
        if pois_lamda:
            self.tform.append(TranformPoissonNoise(pois_lamda))
        if salt_peper_amount:
            self.tform.append(TransformSaltAndPeper(salt_peper_amount, salt_pepper_ratio))
        self.tranform = transforms.Compose(self.tform)

    def __call__(self, img):
        return self.tranform(img)

    def __repr__(self):
        to_return = "NOISY ONE IMAGE BW "
        for o in self.tform:
            to_return += '\n' + repr(o)
        return to_return


class TranformSingleNoisyRGB:
    def __init__(self, mean_photo, pois_lamda, gauss_mean, gauss_std, salt_peper_amount, salt_pepper_ratio=0.5,
                 cam_pos=0):
        #   (defult into up middle)
        self.tform = []
        self.tform.append(TransformSingleCameraRGB(cam_pos, mean_photo))
        if gauss_std:
            self.tform.append(TransformGaussian(gauss_mean, gauss_std))
        if pois_lamda:
            self.tform.append(TranformPoissonNoise(pois_lamda))
        if salt_peper_amount:
            self.tform.append(TransformSaltAndPeper(salt_peper_amount, salt_pepper_ratio))
        self.tranform = transforms.Compose(self.tform)

    def __call__(self, img):
        return self.tranform(img)

    def __repr__(self):
        to_return = "NOISY ONE IMAGE RGB "
        for o in self.tform:
            to_return += '\n' + repr(o)
        return to_return


class TranformSingleNoisyRGBD:
    def __init__(self, mean_photo, pois_lamda, gauss_mean, gauss_var, salt_peper_amount, salt_pepper_ratio=0.5,
                 cam_pos=0):
        #   (default into up middle)
        self.tform = []
        self.tform.append(TransformSingleCameraRGBD(cam_pos, mean_photo))
        if gauss_var:
            self.tform.append(TransformGaussian(gauss_mean, gauss_var))
        if pois_lamda:
            self.tform.append(TranformPoissonNoise(pois_lamda))
        if salt_peper_amount:
            self.tform.append(TransformSaltAndPeper(salt_peper_amount, salt_pepper_ratio))
        self.tranform = transforms.Compose(self.tform)

    def __call__(self, img):
        return self.tranform(img)

    def __repr__(self):
        to_return = "NOISY ONE IMAGE RGBD "
        for o in self.tform:
            to_return += '\n' + repr(o)
        return to_return


class TransformManyCamerasBWNoisy:
    def __init__(self, cam_ids, mean_photo, pois_lamda, gauss_mean, gauss_std,
                 salt_peper_amount, salt_pepper_ratio=0.5):
        self.tform = []
        self.tform.append(TransformManyCameraBw(cam_ids, mean_photo))
        if gauss_std:
            self.tform.append(TransformGaussian(gauss_mean, gauss_std))
        if pois_lamda:
            self.tform.append(TranformPoissonNoise(pois_lamda))
        if salt_peper_amount:
            raise NotImplementedError
            # self.tform.append(TransformSaltAndPeper(salt_peper_amount, salt_pepper_ratio))

        self.transform = transforms.Compose(self.tform)

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        to_return = "NOISY MULTI CAM BW"
        for o in self.tform:
            to_return += '\n' + repr(o)
        return to_return


class TransformManyCamerasRGBDNoisy:
    def __init__(self, cam_ids, mean_photo, pois_lamda, gauss_mean, gauss_std,
                 salt_peper_amount, salt_pepper_ratio=0.5):
        self.tform = []
        self.tform.append(TransformManyCamerasRGBD(cam_ids, mean_photo))
        if gauss_std:
            self.tform.append(TransformGaussian(gauss_mean, gauss_std))
        if pois_lamda:
            self.tform.append(TranformPoissonNoise(pois_lamda))
        if salt_peper_amount:
            raise NotImplementedError
            # self.tform.append(TransformSaltAndPeper(salt_peper_amount, salt_pepper_ratio))

        self.transform = transforms.Compose(self.tform)

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        to_return = "NOISY MULTI CAM RGBD"
        for o in self.tform:
            to_return += '\n' + repr(o)
        return to_return


class TransformScales:
    def __init__(self, mode_shape, std=0.00015):  # mode shape in Z axis
        self.inv_modeshape = np.linalg.pinv(mode_shape)
        self.std = std

    def __call__(self, scales):
        rand_noise = np.random.normal(0, self.std, size=self.inv_modeshape.shape[2])
        return scales + np.matmul(self.inv_modeshape, rand_noise)

    def __repr__(self):
        return "SCALE TRANSFORM WITH STD " + str(self.std)
