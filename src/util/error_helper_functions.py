
from src.util.loss_functions import *

def calc_errors(loss_function, mode_shape, pow, ir_indices, x, y):
    """
    return errors as written in the exel file format
    Args:
        loss_function: The loss function between two elements, should deal with vectors and single element.
        ir_indices: tuple of ir indices
        mode_shape: mode shape as read using matlab_reader.read_modal_shapes()
        x: first set of scales (shape = data_points, num_of_scales)
        y: second set of scales (shape = data_points, num_of_scales)

    Returns:
        error values in the following error
        (Average 3D Reconstruction Error,Average 3D IR Reconstruction Error, Average Regression
        	Regression 0,	Regression 1,	Regression 2,	Regression 3,	Regression 4,	Regression 5,	Regression 6,
        		Regression 7,	Regression 8,	Regression 9)
    """
    three_d_loss, ir_loss, avg_regression = 0, 0, 0
    zero_scales = np.zeros(x.shape[1])
    regression_loss = np.zeros((x.shape[1]))
    data_points = x.shape[0]
    num_of_scales = x.shape[1]
    num_verticies = mode_shape.shape[1]
    for i in trange(data_points):
        three_d_loss += threeD_reconstruction_loss(loss_function, mode_shape, pow, x[i], y[i])
        ir_loss += threeD_reconstruction_loss(loss_function, mode_shape[:,ir_indices], pow, x[i], y[i])
        for k in range(num_of_scales):
            regression_loss[k] += loss_function(zero_scales + x[i, k], zero_scales + y[i, k])
        avg_regression += loss_function(x[i], y[i])
    return (three_d_loss / (num_verticies * data_points), ir_loss / (len(ir_indices) * data_points),
            avg_regression/(num_of_scales * data_points)) + tuple(regression_loss / data_points)


def calc_max_errors(loss_function, scales: np.ndarray, ir_indices: tuple, mode_shape):
    """

    Args:
        loss_function: The loss function between two elements, should deal with vectors and single element.
        scales: Scale ndarray in (n x num_scales)
        ir_indices: tuple of ir indices
        mode_shape: mode shape as read using matlab_reader.read_modal_shapes()

    Returns:
        maximum error values in the following error
        (Average 3D Reconstruction Error,Average 3D IR Reconstruction Error, Average Regression
        	Regression 0,	Regression 1,	Regression 2,	Regression 3,	Regression 4,	Regression 5,	Regression 6,
        		Regression 7,	Regression 8,	Regression 9,	)
    """
    return (calc_max_3d_reconstruction_error(loss_function, scales, mode_shape),
            calc_max_ir_reconstruction_error(loss_function, scales, ir_indices, mode_shape),
            calc_max_regression_error(loss_function,scales)) + \
            tuple(calc_max_per_param_error(loss_function,scales,range(scales.shape[1])))


def calc_max_3d_reconstruction_error(loss_function, scales, mode_shape):
    max_3d = 0
    ver = np.zeros(shape=(scales.shape[0], mode_shape.shape[1], mode_shape.shape[0]))
    for i in range(scales.shape[0]):
        #   creating deformation
        ver[i] = (scales[i,:] * mode_shape).sum(axis=2).T

    for i in trange(scales.shape[0]):
        for j in range(scales.shape[0]):
            curr = loss_function(ver[i], ver[j])
            if curr > max_3d:
                max_3d = curr
    print("max 3d/ir "+f'{max_3d: .4e}')
    return max_3d / mode_shape.shape[1]


def calc_max_ir_reconstruction_error(loss_function, scales, ir_indices, mode_shape):
    return calc_max_3d_reconstruction_error(loss_function, scales, mode_shape[:,ir_indices])


def calc_max_per_param_error(loss_function, scales, ids):
    if isinstance(ids, int):
        ids = [ids]
    max_err = np.zeros(len(ids))
    max_scale = np.zeros(scales.shape[0])
    min_scale = np.zeros(scales.shape[0])
    for i in range(scales.shape[0]):
        max_scale[ids] = np.maximum(scales[i, ids], max_scale[ids])
        min_scale[ids] = np.minimum(scales[i, ids], max_scale[ids])
    for i in trange(scales.shape[0]):
        for j in range(scales.shape[0]):
            for num in ids:
                curr = loss_function(scales[i, num], scales[j, num])
                if curr > max_err[num]:
                    max_err[num] = curr
    print("modes error: " + max_err)
    return max_err[ids]


def calc_max_regression_error(loss_function, scales):
    max_scale, min_scale, max_error = 0, 0, 0
    for i in trange(scales.shape[0]):
        for j in range(scales.shape[0]):
            curr = loss_function(scales[i,:], scales[j,:])
            if curr > max_error:
                max_error = curr
    print("regression: "+f'{max_error/scales.shape[1]: .4e}')
    return max_error/scales.shape[1]



def error_to_exel_string(result):
    exel_string = ""
    for res in result:
        exel_string += str(res) + " "

    return exel_string

