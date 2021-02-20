import os
import shutil
from abc import ABC, abstractmethod
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

from src.model_datasets.image_dataset import ImageDataset
from src.util.general import Functor
from src.util.nn_additions import SubsetChoiceSampler, ReduceMetric, HistMetric, TextMetric


class AbstractResnet(pl.LightningModule, ABC):
    def __init__(self, num_input_layers, num_outputs, loss_func, reduce_error_func_dict, hist_error_func_dict,
                 text_error_func_dict,
                 output_scaling,
                 resnet_type: str, learning_rate,
                 cosine_annealing_steps,
                 weight_decay, dtype=torch.float32, track_ideal_metrics=False):
        super().__init__()
        # TODO consider removing pretrained
        if "loss" in list(reduce_error_func_dict.keys()) + list(hist_error_func_dict.keys()):
            raise ValueError("Bad function names")
        self.num_input_layers = num_input_layers
        self.num_outputs = num_outputs
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.resnet_type = resnet_type
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.track_ideal_metrics = track_ideal_metrics
        self.resnet_dict = {'res18': models.resnet18,
                            'res34': models.resnet34,
                            'res50': models.resnet50,
                            'mobile2': models.mobilenet_v2}
        # self.train_step_metrics = {f"train_{name}": ReduceMetric(foo, compute_on_step=True, dist_sync_on_step=True) for
        #                            name, foo in reduce_error_func_dict.items()}
        self.train_epoch_metrics = {f"train_{name}": ReduceMetric(foo, compute_on_step=False, dist_sync_on_step=True)
                                    for name, foo in reduce_error_func_dict.items()}
        self.train_epoch_metrics_noisy_y = {name + "_ideal": deepcopy(foo)
                                            for name, foo in
                                            self.train_epoch_metrics.items()} if track_ideal_metrics else {}

        self.val_metrics = {f"val_{name}": ReduceMetric(foo, compute_on_step=False) for name, foo in
                            reduce_error_func_dict.items()}
        self.val_metrics.update(
            {f"val_hist_{name}": HistMetric(foo) for name, foo in hist_error_func_dict.items()})
        self.val_metrics.update(
            {name: TextMetric(foo, num_outputs, output_scaling) for name, foo in text_error_func_dict.items()})
        self.val_metrics_noisy_y = {name + "_ideal": deepcopy(foo)
                                    for name, foo in self.val_metrics.items()} if track_ideal_metrics else {}
        self.current_step = 0
        self.plotter_val_data = None
        self.plotter_train_data = None
        self.type(dst_type=dtype)

    @abstractmethod
    def forward(self, x):
        pass

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.cosine_annealing_steps:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(adam, self.cosine_annealing_steps,
                                                             self.learning_rate / 1000)
        else:
            return adam
        return [adam], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[-1]
        if self.track_ideal_metrics:
            y_perfect = batch[1]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            #     for name, metric in self.train_step_metrics.items():
            #         metric.update(y_hat, y)
            #         result[name] = metric.compute()
            #         result[f'min_{name}'] = metric.min.cpu().numpy()
            for name, metric in self.train_epoch_metrics.items():
                metric.update(y_hat, y)
            if self.track_ideal_metrics:
                for name, metric in self.train_epoch_metrics_noisy_y.items():
                    metric.update(y_hat, y_perfect)
            if batch_idx == 0:
                self.plotter_train_data = ((x[0].cpu().numpy(), y[0].cpu().numpy(), y_hat[0].cpu().numpy()),
                                           (x[1].cpu().numpy(), y[1].cpu().numpy(), y_hat[1].cpu().numpy()))
        # self.logger.experiment.log_metric('train_loss', loss, step=self.current_step)
        # self.logger.experiment.log_metrics(result, step=self.current_step)
        self.current_step += 1
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        result = {}
        for name, metric in {**self.train_epoch_metrics, **self.train_epoch_metrics_noisy_y}.items():
            result[name] = metric.compute()
            self.log(name, result[name])
            result[f'min_{name}'] = metric.min.cpu().numpy()
        self.logger.experiment.log_metrics(result, step=self.current_epoch, epoch=self.current_epoch)

    # def training_step_end(self, output):
    #     result = {}
    #     with torch.no_grad():
    #         for name, metric in self.train_metrics.items():
    #             result[name] = metric.update(output['y'], output['y_hat'])
    #     self.log_dict(result, on_step=True)
    #
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[-1]
        if self.track_ideal_metrics:
            y_noisy = batch[1]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            for name, metric in self.val_metrics.items():
                metric.update(y_hat, y)
            if self.track_ideal_metrics:
                for name, metric in self.val_metrics_noisy_y.items():
                    metric.update(y_hat, y_noisy)
            if batch_idx == 0:
                self.plotter_val_data = ((x[0].cpu().numpy(), y[0].cpu().numpy(), y_hat[0].cpu().numpy()),
                                         (x[1].cpu().numpy(), y[1].cpu().numpy(), y_hat[1].cpu().numpy()))
                self.plotter.push(self.current_epoch, (self.plotter_train_data, self.plotter_val_data))
        return loss

    # def validation_step_end(self, output):
    #     with torch.no_grad():
    #         for name, metric in self.val_metrics.items():
    #             metric.update(output['y'], output['y_hat'])

    def validation_epoch_end(
            self, outputs: List[Any]) -> None:
        self.logger.experiment.log_metric('val_loss', torch.stack(outputs).mean(), step=self.current_epoch,
                                          epoch=self.current_epoch)
        for name, metric in {**self.val_metrics, **self.val_metrics_noisy_y}.items():
            if isinstance(metric, ReduceMetric):
                metric_res = metric.compute()
                self.logger.experiment.log_metric(name, metric_res, step=self.current_epoch,
                                                  epoch=self.current_epoch)
                self.log(name, metric_res)
                self.logger.experiment.log_metric(f'min_{name}', metric.min.cpu().numpy(), step=self.current_epoch,
                                                  epoch=self.current_epoch)

            if isinstance(metric, HistMetric):
                self.logger.experiment.log_histogram_3d(metric.compute(), name=name, step=self.current_epoch)
            if isinstance(metric, TextMetric):
                values = metric.compute()
                np.save('src/tests/temp/worst.npy', values)
                self.logger.experiment.log_asset('src/tests/temp/worst.npy', file_name=name, step=self.current_epoch)
