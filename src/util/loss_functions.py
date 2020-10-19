from typing import Union
import numpy as np
import torch
from tqdm import trange

def mse_weighted(weights, a, b):
    loss = (a - b) ** 2
    if len(loss.shape) > 1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss = weights * loss
    return torch.sqrt(torch.sum(loss))


def vertex_mean_rms(mode_shapes, pow, x: Union[torch.tensor, np.ndarray], y: Union[torch.tensor, np.ndarray]):
    """
        return loss between movement (based on the scales) we received and the movement we calculated by our own scales
          Args:
              mode_shapes: a [V,n] tensor or np array representing the mode shapes (only the Z axis)
              pow: the power of 10 used to scale the mode scales
              x: first set of scales
              y: second set of scales
           Returns:
              RMS between the vertex positions calculated from scales
           """

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        device = 'cpu'
        return vertex_mean_rms(mode_shapes, pow, torch.tensor(x, device=device),
                               torch.tensor(y, device=device)).detach().numpy()

    num_vertices = mode_shapes.size / (3 * x.shape[-1])

    device = x.device
    with torch.no_grad():
        _x = x * 10 ** -pow
        _y = y * 10 ** -pow
        _x = _x.view(-1, _x.shape[-1]).to(torch.float64).T
        _y = _y.view(-1, _y.shape[-1]).to(torch.float64).T
        mode_shapes = torch.tensor(mode_shapes, device=device, dtype=torch.float64).reshape(-1, len(_x))
        pos_a = (mode_shapes @ _x).sum(dim=1)
        pos_b = (mode_shapes @ _y).sum(dim=1)
        return torch.norm(pos_a - pos_b, 2) / num_vertices


def calc_max_errors(loss_function, scales: np.ndarray, ir_indices: tuple, mode_shape):
    """

    Args:
        loss_function: The loss function between two scale vectors
        scales: Scale ndarray in (n x num_scales)
        ir_indices: tuple of ir indices

    Returns:
        maximum error values in the following error
        (Average 3D Reconstruction Error,Average 3D IR Reconstruction Error,
        	Regression 0,	Regression 1,	Regression 2,	Regression 3,	Regression 4,	Regression 5,	Regression 6,
        		Regression 7,	Regression 8,	Regression 9,	Average Regression)
    """
    return (calc_max_3d_reconstruction_error(loss_function, scales, mode_shape),
            calc_max_ir_reconstruction_error(loss_function, scales, ir_indices, mode_shape)) + \
            tuple(calc_max_per_param_error(loss_function,scales,range(scales.shape[0])).append(
               calc_max_regression_error(loss_function,scales)))


def calc_max_3d_reconstruction_error(loss_function, scales, mode_shape):
    max_3d = 0
    ver = np.zeros(shape=(scales.shape[1], mode_shape.shape[1], mode_shape.shape[0]))
    for i in range(scales.shape[1]):
        #   creating deformation
        ver[i] = (scales[:, i] * mode_shape).sum(axis=2).T

    for i in trange(scales.shape[1]):
        for j in range(scales.shape[1]):
            curr = loss_function(ver[i], ver[j]) / mode_shape.shape[1]
            if curr > max_3d:
                max_3d = curr
    return max_3d


def calc_max_ir_reconstruction_error(loss_function, scales, ir_indices, mode_shape):
    return calc_max_3d_reconstruction_error(loss_function, scales, mode_shape[:,ir_indices])


def calc_max_per_param_error(loss_function, scales, ids):
    if isinstance(ids, int):
        ids = [ids]
    max_err = np.zeros(len(ids))
    max_scale = np.zeros(scales.shape[1])
    min_scale = np.zeros(scales.shape[1])
    for i in range(scales.shape[1]):
        max_scale[ids] = np.maximum(scales[ids, i], max_scale[ids])
        min_scale[ids] = np.minimum(scales[ids, i], max_scale[ids])
    for i in trange(scales.shape[1]):
        for j in range(scales.shape[1]):
            for num in ids:
                curr = loss_function(scales[num, i], scales[num, j]) / (max_scale[num] - min_scale[num])
                if curr > max_err[num]:
                    max_err[num] = curr
    return max_err[ids]


def calc_max_regression_error(loss_function, scales):

    max_scale, min_scale, max_error = 0, 0, 0
    for i in range(scales.shape[1]):
        curr = loss_function(scales[:,i],0)
        #   max based on given norm
        if curr > max_scale:
            max_scale = curr
        if curr < min_scale:
            min_scale = curr
    for i in trange(scales.shape[1]):
        for j in range(scales.shape[1]):
            curr = loss_function(scales[:,i], scales[:,j]) / (max_scale-min_scale)
            if curr > max_error:
                max_error = curr
    return max_error/scales.shape[0]

