from copy import deepcopy
from typing import List, Any
import numpy as np
from torch import optim, nn
from torchvision import models
import pytorch_lightning as pl
import torch
from torchvision.transforms import transforms

from src.models.abstract_resnet import AbstractResnet
from src.models.resnet_synth import CustomInputResnet


class AggResnet(AbstractResnet):
    def __init__(self, num_input_layers, num_outputs, loss_func, reduce_error_func_dict, hist_error_func_dict,
                 text_error_func_dict,
                 output_scaling,
                 resnet_type: str, learning_rate,
                 cosine_annealing_steps,
                 weight_decay, num_pictures, dtype=torch.float32, track_ideal_metrics=False, mode='mean',
                 ):
        super().__init__(num_input_layers, num_outputs, loss_func, reduce_error_func_dict,
                         hist_error_func_dict,
                         text_error_func_dict,
                         output_scaling,
                         resnet_type, learning_rate,
                         cosine_annealing_steps,
                         weight_decay, dtype, track_ideal_metrics)

        self.img_subnets = nn.ModuleList([CustomInputResnet(num_input_layers, num_outputs, loss_func,
                                                            {}, {}, {}, 1,
                                                            resnet_type, learning_rate,
                                                            cosine_annealing_steps,
                                                            weight_decay, dtype) for _ in range(num_pictures)])
        self.num_pictures = num_pictures
        self.mode = mode

    def forward(self, x):
        # x.shape=(N,K,L,H,W)
        # N= Batch size
        # K= Number of input photos
        # L= Number of color layers in photo, 1 for bw, 3 for rgb,
        #      Incremented by 1 if depth is used
        N, K, L, H, W = x.shape
        latent = torch.zeros((N, self.num_pictures, self.num_outputs), dtype=x.dtype, device=x.device)
        for index, resnet in enumerate(self.img_subnets):
            latent[:, index, :] = resnet(x[:, index, :self.num_input_layers])
        if (self.mode == 'mean'):
            return latent.mean(dim=1)
        elif (self.mode == 'min'):
            return latent.min(dim=1)
        elif (self.mode == 'max'):
            return latent.min(dim=1)
        else:
            raise NotImplementedError
