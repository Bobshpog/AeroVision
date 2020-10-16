from typing import Union

import numpy as np
import torch


def mse_weighted(weights, a, b):
    loss = (a - b) ** 2
    if len(loss.shape) > 1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss = weights * loss
    return torch.sqrt(torch.sum(loss))


def vertex_mean_rms(mode_shapes, pow, x: Union[torch.tensor, np.array], y: Union[torch.tensor, np.array]):
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
    if x is np.array or y is np.array:
        device = 'cpu'
        return vertex_mean_rms(mode_shapes, pow, torch.tensor(x, device=device),
                               torch.tensor(y, device=device)).detach().numpy()

    num_vertices = mode_shapes.size / (3 * x.shape[-1])

    device = x.device
    with torch.no_grad():
        _x = x * 10 ** -pow
        _y = y * 10 ** -pow
        _x = _x.to(torch.float64).T
        _y = _y.to(torch.float64).T
        mode_shapes = torch.tensor(mode_shapes, device=device, dtype=torch.float64).view(-1, len(_x))
        pos_a = (mode_shapes @ _x).sum(dim=1)
        pos_b = (mode_shapes @ _y).sum(dim=1)
        return torch.norm(pos_a - pos_b, 2) / num_vertices
