import torch.nn.functional as F

from src.util.loss_functions import *


def calc_errors(loss_function, mode_shapes: np.ndarray, scaling, ir_indices, x, y):
    """
    return errors as written in the exel file format
    Args:
        loss_function: The loss function between two elements, should deal with vectors and single element,must work on tensors.
        ir_indices: tuple of ir indices
        mode_shapes: mode shape as read using matlab_reader.read_modal_shapes()
        x: first set of scales (shape = num_datapoints, num_scales)
        y: second set of scales (shape = num_datapoints, num_scales)

    Returns:
        error values in the following error
        ( 3D Reconstruction Error Tensor ,3D IR Reconstruction Error Tensor , Average Regression Tensor
        , regression Tensor (num_datapoints,nums_cales))

    """
    num_datapoints, num_scales = x.shape
    # device = x.device
    vertex_loss = reconstruction_loss_3d(loss_function, mode_shapes, scaling, x, y)
    ir_loss = reconstruction_loss_3d(loss_function, mode_shapes[:, ir_indices], scaling, x, y)
    regression_loss = torch.abs(x-y) / scaling
    avg_regression = regression_loss.sum(-1) / (num_scales * scaling)
    return (vertex_loss, ir_loss,
            avg_regression.double(), regression_loss.T.double())#Transpose is important

def calc_mean_errors(loss_function, mode_shapes: np.ndarray, scaling, ir_indices, x, y):
    (vertex_loss,ir_loss,avg_regression,regression_loss)=calc_errors(loss_function, mode_shapes, scaling, ir_indices, x, y)
    return (vertex_loss.mean(), ir_loss.mean(),
        avg_regression.mean(), regression_loss.mean(dim=0))

def calc_max_errors(loss_function, scales: np.ndarray, ir_indices, mode_shape, device='cpu'):
    """

    Args:
        loss_function: The loss function between two elements, should deal with vectors and single element.
        scales: Scale ndarray in (n x num_scales)
        ir_indices: tuple of ir indices
        mode_shape: mode shape as read using matlab_reader.read_modal_shapes()
        device: 'cpu' or 'cuda0'

    Returns:
        maximum error values in the following error
        (Average 3D Reconstruction Error,Average 3D IR Reconstruction Error, Average Regression
        	Regression 0,	Regression 1,	Regression 2,	Regression 3,	Regression 4,	Regression 5,	Regression 6,
        		Regression 7,	Regression 8,	Regression 9,	)
    """
    _scales = torch.tensor(scales, device=device)
    return (calc_max_3d_reconstruction_error(loss_function, _scales, mode_shape),
            calc_max_ir_reconstruction_error(loss_function, _scales, ir_indices, mode_shape),
            calc_max_regression_error(loss_function, _scales),
            *calc_max_per_param_error(loss_function, _scales))


def batch_mat_pdist(a: torch.Tensor, b: torch.Tensor, p=2):
    """
    Compute the pairwise distance_tensor matrix between a and b which both have size [n, d]. The result is a tensor of
    size [n, n] whose entry [i, j] contains the distance_tensor between a[i, :] and b[j, :].
    :param a: A tensor containing m batches of n points of dimension d. i.e. of size [n, d]
    :param b: A tensor containing m batches of n points of dimension d. i.e. of size [n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """
    max_val = torch.tensor(0, dtype=a.dtype, device=a.device)
    size = len(a)
    step = size // 1

    def partial_dist_max(x, y):
        return (x - y.unsqueeze(0)).abs_().pow_(p).sum(2).pow(1 / p).max()

    # pbar = tqdm(enumerate(a), total=size, desc=max_val)
    for idx, val in enumerate(a):
        for i in range(idx, size, step):
            end = max(idx + step, size)
            max_val = torch.max(max_val, partial_dist_max(val, b[i:end]))
        # pbar.set_description(f'max_val={max_val}')
    return max_val


def calc_max_3d_reconstruction_error(loss_function, scales, mode_shapes):
    device = scales.device
    if loss_function in ('l1', 'l2'):
        p = int(loss_function[1:])
        num_scales = scales.shape[-1]
        num_vertices = mode_shapes.size / (3 * num_scales)
        with torch.no_grad():
            _scales = scales.view(-1, num_scales).to(torch.float64).T
            _mode_shapes = torch.tensor(mode_shapes, device=device, dtype=torch.float64).reshape(-1, len(_scales))
            point_clouds = (_mode_shapes @ _scales).T
            max_3d = batch_mat_pdist(point_clouds, point_clouds, p).item() / num_vertices
    else:
        raise NotImplementedError
    return max_3d


def calc_max_ir_reconstruction_error(loss_function, scales, ir_indices, mode_shape):
    return calc_max_3d_reconstruction_error(loss_function, scales, mode_shape[:, ir_indices])


def calc_max_per_param_error(loss_function, scales: torch.Tensor):
    max_scales = scales.max(dim=0)[0]
    min_scales = scales.min(dim=0)[0]
    max_err = []
    if loss_function == 'l1':
        loss_function = lambda x: torch.norm(x,p=1)
    if loss_function == 'l2':
        loss_function = torch.norm
    for i, j in zip(min_scales, max_scales):
        max_err.append(loss_function(i - j))
    max_err = tuple([i.item() for i in max_err])
    return max_err


def calc_max_regression_error(loss_function, scales):
    num_scales = scales.shape[-1]
    if loss_function in ('l1', 'l2'):
        p = int(loss_function[1:])
        with torch.no_grad():
            max_error = (batch_mat_pdist(scales, scales, p).max() / num_scales).item()
    else:
        raise NotImplementedError
    return max_error


def error_to_exel_string(result):
    exel_string = ""
    for res in result:
        exel_string += str(res) + " "

    return exel_string
