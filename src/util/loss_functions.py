import torch


def mse_weighted(weights, a, b):
    loss = (a - b) ** 2
    if len(loss.shape) > 1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss = weights * loss
    return torch.sqrt(torch.sum(loss))


def vertices_mean_rms(mode_shapes, pow, scale_a: torch.tensor, scale_b: torch.tensor):
    """
        return loss between movement (based on the scales) we received and the movement we calculated by our own scales
          Args:
              mode_shapes: a [V,n] tensor representing the mode shapes (only the Z axis)
              pow: the power of 10 used to scale the mode scales
              scale_a: first set of scales
              scale_b: second set of scales
           Returns:
              RMS between the vertex positions calculated from scales
           """
    scale_a *= 10 ** -pow
    scale_b *= 10 ** -pow
    mode_shapes = torch.tensor(mode_shapes, device=scale_a.device, dtype=torch.float64)
    pos_a = (mode_shapes * scale_a).sum(dim=1)
    pos_b = (mode_shapes * scale_b).sum(dim=1)
    return torch.norm(pos_a - pos_b, 2)
