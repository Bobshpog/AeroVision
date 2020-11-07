from typing import Union

import numpy as np
import torch


def mse_weighted(weights, a, b):
    loss = (a - b) ** 2
    if len(loss.shape) > 1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss = weights * loss
    return torch.sqrt(torch.sum(loss))


def l2_norm(a, b):
    return mse_weighted(1, a, b)


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

    l2 = torch.norm
    return reconstruction_loss_3d(l2, mode_shapes, pow, x, y)


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
        _x = x / scale_factor
        _y = y / scale_factor
        _x = _x.view(-1, _x.shape[-1]).to(torch.float64).T
        _y = _y.view(-1, _y.shape[-1]).to(torch.float64).T
        mode_shapes = torch.tensor(mode_shapes, device=device, dtype=torch.float64).reshape(-1, len(_x))
        pos_a = (mode_shapes @ _x)
        pos_b = (mode_shapes @ _y)
        return loss_function(pos_a- pos_b,dim=0)/num_vertices
