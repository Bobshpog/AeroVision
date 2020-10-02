import torch


def MSE_Weighted(weights, a, b):
    loss=(a-b)**2
    if len(loss.shape)>1:
        weights = torch.tensor(weights, dtype=a.dtype, device=a.device)
        loss=weights*loss
    return torch.sqrt(torch.sum(loss))