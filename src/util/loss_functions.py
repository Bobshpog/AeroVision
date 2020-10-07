import torch


def MSE_Weighted(weights, a, b):
    loss=(a-b)**2
    if len(loss.shape)>1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss=weights*loss
    return torch.sqrt(torch.sum(loss))


def verticies_L2(mode_shapes, amp, scale_a, scale_b):
    """
        return loss between movement (based on the scales) we received and the movement we calculated by our own scales
          Args:
              mode_shapes: a [V,n] tensor representing the mode shapes (only the Z axis)
              amp: the amplifier of the loss function to compensate between different mode shapes
              scale_a: the scale we received via the network [n] size tensor
              scale_b: ground truth scales
           Returns:
              L2 between the displacement
           """
    disp_made = (mode_shapes * scale_a).sum(axis=1)
    disp_from_scale = (mode_shapes * scale_b).sum(axis=1)
    return torch.norm(disp_made - disp_from_scale, 2) * amp

