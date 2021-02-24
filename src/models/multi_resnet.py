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


class MultiResnet(AbstractResnet):
    def __init__(self, num_input_layers, num_outputs, loss_func, reduce_error_func_dict, hist_error_func_dict,
                 text_error_func_dict,
                 output_scaling,
                 resnet_type: str, learning_rate,
                 cosine_annealing_steps,
                 weight_decay, num_pictures, dtype=torch.float32, track_ideal_metrics=False, use_depth=False,
                 latent_layer_size_per=128):
        super().__init__(num_input_layers, num_outputs, loss_func, reduce_error_func_dict,
                         hist_error_func_dict,
                         text_error_func_dict,
                         output_scaling,
                         resnet_type, learning_rate,
                         cosine_annealing_steps,
                         weight_decay, dtype, track_ideal_metrics)
        self.latent_layer_size_per = latent_layer_size_per
        self.num_pictures = num_pictures

        self.img_subnets = nn.ModuleList([CustomInputResnet(num_input_layers, latent_layer_size_per, loss_func,
                                                            {}, {}, {}, 1,
                                                            resnet_type, learning_rate,
                                                            cosine_annealing_steps,
                                                            weight_decay, dtype) for _ in range(num_pictures)])

        self.depth_subnets = False
        if use_depth:
            self.depth_subnets = nn.ModuleList([CustomInputResnet(1, latent_layer_size_per, loss_func,
                                                                  {}, {}, {}, 1,
                                                                  resnet_type, learning_rate,
                                                                  cosine_annealing_steps,
                                                                  weight_decay, dtype) for _ in range(num_pictures)])

        latent_layer_size = latent_layer_size_per * num_pictures
        dense1_size = max(latent_layer_size // 2, 256)
        dense2_size = max(latent_layer_size // 4, 256)
        dense3_size = max(latent_layer_size // 8, 128)
        self.densenet = nn.Sequential(nn.Linear(latent_layer_size, dense1_size),
                                      nn.ReLU(),
                                      nn.Linear(dense1_size, dense2_size),
                                      nn.ReLU(),
                                      nn.Linear(dense2_size, dense3_size),
                                      nn.ReLU(),
                                      nn.Linear(dense3_size, self.num_outputs))

    def forward(self, x):
        # x.shape=(N,K,L,H,W)
        # N= Batch size
        # K= Number of input photos
        # L= Number of color layers in photo, 1 for bw, 3 for rgb,
        #      Incremented by 1 if depth is used
        N, K, L, H, W = x.shape
        latent_size = self.latent_layer_size_per * self.num_pictures
        if self.depth_subnets:
            latent_size *= 2
        latent = torch.zeros((N, latent_size), dtype=x.dtype, device=x.device)
        start = 0
        end = self.latent_layer_size_per
        for index, resnet in enumerate(self.img_subnets):
            latent[:, start:end] = resnet(x[:, index, :self.num_input_layers])
            start = end
            end += self.latent_layer_size_per
        if self.depth_subnets:
            for index, resnet in enumerate(self.depth_subnets):
                latent[:, start:end] = resnet(x[:, index, -1])
                start = end
                end += self.latent_layer_size_per
        return self.densenet(latent)
