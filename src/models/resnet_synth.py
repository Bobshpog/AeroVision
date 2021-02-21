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


def run_resnet_synth(num_input_layers, num_outputs,
                     comment, train_db_path, val_db_path, val_split, transform, out_transform, mean_error_func_dict,
                     hist_error_func_dict, text_error_func_dict, output_scaling=1e4, lr=1e-2,
                     resnet_type='18', train_cache_size=5500, val_cache_size=1000, batch_size=64, num_epochs=1000,
                     weight_decay=0, cosine_annealing_steps=10, loss_func=F.smooth_l1_loss, subsampler_size=640,
                     dtype=torch.float32, track_ideal_metrics=False, monitor_metric_name=None, parallel_plotter=None):
    if None in [batch_size, num_epochs, resnet_type, train_db_path, val_db_path, val_split, comment]:
        raise ValueError('Config not fully initialized')
    torch_to_np_dtypes = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64
    }
    np_dtype = torch_to_np_dtypes[dtype]
    transform = Functor(transform)
    params = {'batch_size': batch_size, 'train_db': train_db_path.split('/')[-1],
              'val_db': val_db_path.split('/')[-1], 'train-val_split_index': val_split,
              'loss_func': loss_func.__name__, 'num_outputs': num_outputs,
              'output_scaling': output_scaling, 'resnet_type': resnet_type, 'lr': lr,
              'weight_decay': weight_decay, 'cosine_annealing_steps': cosine_annealing_steps,'transform':transform}
    with h5py.File(train_db_path, 'r') as hf:
        db_size = hf['data']['images'].len()
    if isinstance(val_split, int):
        train_dset = ImageDataset(train_db_path,
                                  transform=transform, output_scaling=output_scaling, out_transform=out_transform,
                                  cache_size=train_cache_size,
                                  max_index=val_split, dtype=np_dtype)
        val_dset = ImageDataset(val_db_path,
                                transform=transform, output_scaling=output_scaling, out_transform=out_transform,
                                cache_size=val_cache_size,
                                min_index=val_split, dtype=np_dtype)
    else:
        train_split = set(range(db_size))
        train_split -= set(val_split)
        train_split = tuple(train_split)
        train_dset = ImageDataset(train_db_path,
                                  transform=transform, output_scaling=output_scaling, out_transform=out_transform,
                                  cache_size=train_cache_size,
                                  index_list=train_split, dtype=np_dtype)
        val_dset = ImageDataset(val_db_path,
                                transform=transform, output_scaling=output_scaling, out_transform=out_transform,
                                cache_size=val_cache_size,
                                index_list=val_split, dtype=np_dtype)
    train_loader = DataLoader(train_dset, batch_size, shuffle=False, num_workers=4,
                              sampler=SubsetChoiceSampler(subsampler_size, len(train_dset)))
    val_loader = DataLoader(val_dset, batch_size, shuffle=True, num_workers=4)
    model = CustomInputResnet(num_input_layers, num_outputs, loss_func=loss_func, output_scaling=output_scaling,
                              reduce_error_func_dict=mean_error_func_dict,
                              hist_error_func_dict=hist_error_func_dict,
                              text_error_func_dict=text_error_func_dict,
                              resnet_type=resnet_type, learning_rate=lr,
                              cosine_annealing_steps=10, weight_decay=weight_decay, dtype=dtype,
                              track_ideal_metrics=track_ideal_metrics)

    model.plotter = parallel_plotter
    logger = CometLogger(api_key="sjNiwIhUM0j1ufNwaSjEUHHXh", project_name="AeroVision",
                         experiment_name=comment)
    logger.log_hyperparams(params=params)
    checkpoints_folder = f"./checkpoints/{comment}/"
    if os.path.isdir(checkpoints_folder):
        shutil.rmtree(checkpoints_folder)
    else:
        Path(checkpoints_folder).mkdir(parents=True, exist_ok=True)
    if monitor_metric_name:
        dirpath = f"checkpoints/{comment}"
        dirpath_path = Path(dirpath)
        if dirpath_path.exists():
            shutil.rmtree(dirpath_path)
        mcp = ModelCheckpoint(
            dirpath=dirpath,
            filename="{epoch}_{" + monitor_metric_name + ":.3e}",
            save_last=True,
            save_top_k=1,
            period=1,
            monitor=monitor_metric_name,
            verbose=True)
        callbacks = [mcp]
    else:
        callbacks = None

    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs,
                         callbacks=callbacks,
                         num_sanity_val_steps=0,
                         profiler=True, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    logger.experiment.log_asset_folder(checkpoints_folder)
