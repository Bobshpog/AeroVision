from typing import Union

import numpy as np
import torch

from src.util.memoization.memoization.memoization import cached


def mse_weighted(weights, a, b):
    loss = (a - b) ** 2
    if len(loss.shape) > 1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss = weights * loss
    return torch.sqrt(torch.sum(loss))


def l2_norm(a, b):
    return mse_weighted(1, a, b)


@cached(max_size=2)
def l1_norm(y_hat, y):
    return (y_hat - y).abs()


def l1_norm_indexed(output_scaling,i, y_hat, y):
    return l1_norm(y_hat, y)[:,i]/output_scaling


def vertex_mean_rms(mode_shapes, scale_factor, x: Union[torch.tensor, np.ndarray], y: Union[torch.tensor, np.ndarray]):
    """
        return loss between movement (based on the scales) we received and the movement we calculated by our own scales
          Args:
              mode_shapes: a [V,n] tensor or np array representing the mode shapes (only the Z axis)
              scale_factor: the power of 10 used to scale the mode scales
              x: first set of scales
              y: second set of scales
           Returns:
              RMS between the vertex positions calculated from scales
           """

    l2 = torch.norm
    return reconstruction_loss_3d(l2, mode_shapes, scale_factor, x, y)


def reconstruction_loss_3d(loss_function, mode_shapes: np.ndarray, scale_factor: int,
                           x: Union[torch.tensor, np.ndarray], y: Union[torch.tensor, np.ndarray]):
    """
        return loss between shape (based on the scales) we received and the shape we calculated by our own scales
          Args:
              loss_function: loss function takes two |V| vectors and calclate the loss between them
              mode_shapes: a [V,n] tensor or np array representing the mode shapes (only the Z axis)
              scale_factor: the power of 10 used to scale the mode scales
              x: first set of scales
              y: second set of scales
           Returns:
              RMS between the vertex positions calculated from scales
           """

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        device = 'cpu'
        return reconstruction_loss_3d(loss_function, mode_shapes, scale_factor, torch.tensor(x, device=device),
                                      torch.tensor(y, device=device)).detach().numpy()
    num_vertices = mode_shapes.size / (3 * x.shape[-1])
    device = x.device
    with torch.no_grad():
        _x = x.detach().clone().to(torch.float64)
        _y = y.detach().clone().to(torch.float64)
        _x = _x.T
        _y = _y.T
        mode_shapes = torch.tensor(mode_shapes, device=device, dtype=torch.float64).reshape(-1, len(_x))
        pos_a = (mode_shapes @ _x)
        pos_b = (mode_shapes @ _y)
        return loss_function(pos_a - pos_b, dim=0) / num_vertices/scale_factor


def reconstruction_loss_3d_new(loss_function, mode_shapes: np.ndarray, scale_factor: int,
                           x: Union[torch.tensor, np.ndarray], y: Union[torch.tensor, np.ndarray]):

    """
        return loss between shape (based on the scales) we received and the shape we calculated by our own scales
          Args:
              loss_function: loss function takes two |V| vectors and calclate the loss between them
              mode_shapes: a [V,n] tensor or np array representing the mode shapes (only the Z axis)
              scale_factor: the power of 10 used to scale the mode scales
              x: first set of scales
              y: second set of scales
           Returns:
              RMS between the vertex positions calculated from scales
           """

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        device = 'cpu'
        return reconstruction_loss_3d_new(loss_function, mode_shapes, scale_factor, torch.tensor(x, device=device),
                                      torch.tensor(y, device=device)).detach().numpy()

    with torch.no_grad():
        x_reconst = torch.matmul(torch.tensor(np.swapaxes(mode_shapes, 0, 1)), x.T)
        #   TODO save the mode shape in this format so the transformations wont be done all the time
        y_reconst = torch.matmul(torch.tensor(np.swapaxes(mode_shapes, 0, 1)), y.T)
        return loss_function(x_reconst - y_reconst, dim=[1, 0]) / (scale_factor * mode_shapes.shape[1])


def L_infinity(mode_shapes: np.ndarray, scale_factor: int,
               x: Union[torch.tensor, np.ndarray], y: Union[torch.tensor, np.ndarray]):
    """
        return L_infinity between shape (based on the scales) we received and the shape we calculated by our own scales
          Args:
              mode_shapes: a [V,n] tensor or np array representing the mode shapes (only the Z axis)
              scale_factor: the power of 10 used to scale the mode scales
              x: first set of scales
              y: second set of scales
           Returns:
              L inf between the vertex positions calculated from scales |n_data| tensor

           """

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        device = 'cpu'
        return L_infinity(mode_shapes, scale_factor, torch.tensor(x, device=device),
                          torch.tensor(y, device=device)).detach().numpy()
    device = x.device
    with torch.no_grad():
        _x = x.detach().clone().to(torch.float64)
        _y = y.detach().clone().to(torch.float64)
        _x = _x.T
        _y = _y.T
        mode_shapes = torch.tensor(mode_shapes, device=device, dtype=torch.float64).reshape(-1, len(_x))
        pos_a = (mode_shapes @ _x)
        pos_b = (mode_shapes @ _y)
        diff = (pos_b - pos_a).reshape((int(mode_shapes.shape[0] / 3), 3,-1))
        diff = torch.norm(diff, dim=1)
        return torch.norm(diff, dim=0, p=float('inf'))/scale_factor


def y_hat_get_scale_i(scale, y_mean, y_sd, i, y_hat, y):
    return (y_hat[:, i] - y_mean[i]) / y_sd[i] / scale


def y_get_scale_i(scale, y_mean, y_sd, i, y_hat, y):
    return (y[:, i] - y_mean[i]) / y_sd[i] / scale
