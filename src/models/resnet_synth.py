import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Any

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from torchvision.models.mobilenet import ConvBNReLU, _make_divisible

from src.model_datasets.image_dataset import ImageDataset
from src.models.abstract_resnet import AbstractResnet
from src.util.general import Functor
from src.util.nn_additions import SubsetChoiceSampler, ReduceMetric, HistMetric, TextMetric


class CustomInputResnet(AbstractResnet):
    def __init__(self, num_input_layers, num_outputs, loss_func, reduce_error_func_dict, hist_error_func_dict,
                 text_error_func_dict,
                 output_scaling,
                 resnet_type: str, learning_rate,
                 cosine_annealing_steps,
                 weight_decay, dtype=torch.float32, track_ideal_metrics=False):
        super().__init__(num_input_layers, num_outputs, loss_func, reduce_error_func_dict, hist_error_func_dict,
                         text_error_func_dict,
                         output_scaling,
                         resnet_type, learning_rate,
                         cosine_annealing_steps,
                         weight_decay, dtype=torch.float32, track_ideal_metrics=False)

        self.model = self.resnet_dict[resnet_type](pretrained=False, num_classes=num_outputs)
        # altering resnet to fit more than 3 input layers
        if resnet_type.startswith('res'):
            self.model.conv1 = nn.Conv2d(num_input_layers, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        if resnet_type.startswith('mobile'):
            self.model.features[0]=ConvBNReLU(self.num_input_layers,_make_divisible(32.0,8),stride=2,norm_layer=nn.BatchNorm2d)
        self.plotter_val_data = None
        self.plotter_train_data = None
        self.type(dst_type=dtype)

    def forward(self, x):
        x = self.model(x)
        return x



def L1_normalized_loss(min, max):
    return lambda x, y: torch.mean(F.l1_loss(x, y, reduction='none'), dim=0) / torch.tensor((max - min),
                                                                                            device=x.device)

