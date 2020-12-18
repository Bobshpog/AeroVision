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

from src.model_datasets.image_dataset import ImageDataset
from src.util.general import Functor
from src.util.nn_additions import SubsetChoiceSampler, ReduceMetric, HistMetric, TextMetric


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, reduce_error_func_dict, hist_error_func_dict,
                 text_error_func_dict,
                 output_scaling,
                 resnet_type, learning_rate,
                 cosine_annealing_steps,
                 weight_decay, dtype=torch.float32, track_ideal_metrics=False):
        super().__init__()
        # TODO consider removing pretrained
        resnet_dict = {'18': models.resnet18,
                       '34': models.resnet34,
                       '50': models.resnet50}
        if "loss" in list(reduce_error_func_dict.keys()) + list(hist_error_func_dict.keys()):
            raise ValueError("Bad function names")
        self.num_input_layers = num_input_layers
        self.num_outputs = num_outputs
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.resnet_type = resnet_type
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.track_perfect_y_metrics = track_ideal_metrics
        # self.train_step_metrics = {f"train_{name}": ReduceMetric(foo, compute_on_step=True, dist_sync_on_step=True) for
        #                            name, foo in reduce_error_func_dict.items()}
        self.train_epoch_metrics = {f"train_{name}": ReduceMetric(foo, compute_on_step=False, dist_sync_on_step=True)
                                    for name, foo in reduce_error_func_dict.items()}
        self.train_epoch_metrics_noisy_y = {name + "_ideal": deepcopy(foo)
                                            for name, foo in self.train_epoch_metrics.items()}

        self.val_metrics = {f"val_{name}": ReduceMetric(foo, compute_on_step=False) for name, foo in
                            reduce_error_func_dict.items()}
        self.val_metrics.update(
            {f"val_hist_{name}": HistMetric(foo) for name, foo in hist_error_func_dict.items()})
        self.val_metrics.update(
            {name: TextMetric(foo, num_outputs, output_scaling) for name, foo in text_error_func_dict.items()})
        self.val_metrics_noisy_y = {name + "_ideal": deepcopy(foo)
                                    for name, foo in self.val_metrics.items()}
        self.current_step = 0
        self.resnet = resnet_dict[resnet_type](pretrained=False, num_classes=num_outputs)
        # altering resnet to fit more than 3 input layers
        self.resnet.conv1 = nn.Conv2d(num_input_layers, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        self.type(dst_type=dtype)

    def forward(self, x):
        x = self.resnet(x)
        return x

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
        if self.track_perfect_y_metrics:
            y_perfect = batch[1]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        result = {}
        with torch.no_grad():
            #     for name, metric in self.train_step_metrics.items():
            #         metric.update(y_hat, y)
            #         result[name] = metric.compute()
            #         result[f'min_{name}'] = metric.min.cpu().numpy()
            for name, metric in self.train_epoch_metrics.items():
                metric.update(y_hat, y)
            if self.track_perfect_y_metrics:
                for name, metric in self.train_epoch_metrics_noisy_y.items():
                    metric.update(y_hat, y_perfect)
        # self.logger.experiment.log_metric('train_loss', loss, step=self.current_step)
        # self.logger.experiment.log_metrics(result, step=self.current_step)
        self.current_step += 1
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        result = {}
        for name, metric in {**self.train_epoch_metrics, **self.train_epoch_metrics_noisy_y}:
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
        if self.track_perfect_y_metrics:
            y_noisy = batch[1]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            for name, metric in self.val_metrics.items():
                metric.update(y_hat, y)
            if self.track_perfect_y_metrics:
                for name, metric in self.val_metrics_noisy_y.items():
                    metric.update(y_hat, y_noisy)
        return loss

    # def validation_step_end(self, output):
    #     with torch.no_grad():
    #         for name, metric in self.val_metrics.items():
    #             metric.update(output['y'], output['y_hat'])

    def validation_epoch_end(
            self, outputs: List[Any]) -> None:
        self.logger.experiment.log_metric('val_loss', torch.stack(outputs).mean(), step=self.current_epoch,
                                          epoch=self.current_epoch)
        for name, metric in {**self.val_metrics, **self.val_metrics}:
            if isinstance(metric, ReduceMetric):
                metric_res=metric.compute()
                self.logger.experiment.log_metric(name, metric_res, step=self.current_epoch,
                                                  epoch=self.current_epoch)
                self.log(name,metric_res)
                self.logger.experiment.log_metric(f'min_{name}', metric.min.cpu().numpy(), step=self.current_epoch,
                                                  epoch=self.current_epoch)

            if isinstance(metric, HistMetric):
                self.logger.experiment.log_histogram_3d(metric.compute(), name=name, step=self.current_epoch)
            if isinstance(metric, TextMetric):
                values = metric.compute()
                np.save('src/tests/temp/worst.npy', values)
                self.logger.experiment.log_asset('src/tests/temp/worst.npy', file_name=name, step=self.current_epoch)


#
# class LoggerCallback(Callback):
#     def __init__(self, logger, stopping_metric='val_loss', stopping_patience=10):
#         self.logger = logger
#         self.metrics = {}
#         self.stopping_metric = stopping_metric
#         self.min_counter = MinCounter(stopping_patience)
#         self.max_errors = {'l1_3d_loss': 0.0569622089146808, 'l1_3d_ir_loss': 0.0628303256182133,
#                            'l1_reg_avg': 0.00348652643151581,
#                            'l2_3d_loss': 0.000732383042264162, 'l2_3d_ir_loss': 0.0141162749770687,
#                            'l2_reg_avg': 0.00314825028181076}
#
#     def on_epoch_end(self, trainer, pl_module):
#         # training
#         error_dict = {}
#         error_dict[f'train_loss'] = torch.mean(
#             torch.cat([x.flatten() for x in pl_module.train_batch_list[f'train_loss']]))
#         for i in range(pl_module.num_output_layers):
#             scale_err_hist = torch.cat(
#                 [x.flatten() for x in pl_module.train_batch_list[f'train_l1_scale{i}']]).flatten()
#             error_dict[f'train_scale_err{i}'] = torch.mean(scale_err_hist)
#             scale_hist = torch.cat([x.flatten() for x in pl_module.train_batch_list[f'train_output{i}']]).flatten()
#             self.logger.experiment.log_histogram_3d(scale_hist.cpu().numpy(), name='hist_' + f'train_scale{i}',
#                                                     step=pl_module.current_epoch)
#             self.logger.experiment.log_histogram_3d(scale_err_hist.cpu().numpy(), name='hist_' + f'train_scale_err{i}',
#                                                     step=pl_module.current_epoch)
#         for error_str in pl_module.error_metrics:
#             max_error = self.max_errors[error_str]
#             error_str = f'train_{error_str}'
#             error_tensor = torch.cat([x.flatten() for x in pl_module.train_batch_list[error_str]]).flatten()
#             curr_loss = torch.mean(error_tensor)
#             self.logger.experiment.log_histogram_3d(error_tensor.cpu().numpy() / max_error, name='hist_' + error_str,
#                                                     step=pl_module.current_epoch)
#             error_dict[error_str] = curr_loss
#         self.metrics = {**self.metrics, **error_dict}
#
#         # Both val and training
#         self.logger.experiment.log_metrics(self.metrics, step=pl_module.current_epoch, epoch=pl_module.current_epoch)
#         trainer.should_stop = self.min_counter.add(self.metrics[self.stopping_metric], pl_module.current_epoch)
#
#         # cleanup
#         self.metrics.clear()
#         for i in pl_module.train_batch_list.values():
#             i.clear()
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         error_dict = {}
#         loss_tensor = torch.cat([x.flatten() for x in pl_module.val_batch_list[f'val_l2_3d_ir_loss']])
#         error_dict[f'val_loss'] = torch.mean(torch.cat([x.flatten() for x in pl_module.val_batch_list[f'val_loss']]))
#         worst_indices = torch.argsort(loss_tensor, descending=True)[:5]
#         worst_scales = torch.zeros((2 * 5, pl_module.num_output_layers), dtype=loss_tensor.dtype,
#                                    device=loss_tensor.device)
#         worst_string = ""
#         for i in range(pl_module.num_output_layers):
#             scale_err_hist = torch.cat([x.flatten() for x in pl_module.val_batch_list[f'val_l1_scale{i}']])
#             error_dict[f'val_scale_err{i}'] = torch.mean(scale_err_hist)
#             output_hist = torch.cat([x.flatten() for x in pl_module.val_batch_list[f'val_output{i}']])
#             expected_hist = torch.cat([x.flatten() for x in pl_module.val_batch_list[f'val_expected{i}']])
#             worst_scales[:5, i] = output_hist[worst_indices]
#             worst_scales[5:, i] = expected_hist[worst_indices]
#             self.logger.experiment.log_histogram_3d(output_hist.cpu().numpy(), name='hist_' + f'val_scale{i}',
#                                                     step=pl_module.current_epoch)
#             self.logger.experiment.log_histogram_3d(scale_err_hist.cpu().numpy(), name='hist_' + f'val_scale_err{i}',
#                                                     step=pl_module.current_epoch)
#         for i in worst_scales:
#             for j in i:
#                 worst_string += str(float(j)) + " "
#             worst_string += ";"
#
#         self.logger.experiment.log_text(worst_string, step=pl_module.current_epoch)
#
#         for error_str in pl_module.error_metrics:
#             max_error = self.max_errors[error_str]
#             error_str = f'val_{error_str}'
#             error_tensor = torch.cat([x.flatten() for x in pl_module.val_batch_list[error_str]]).flatten()
#             curr_loss = torch.mean(error_tensor)
#             self.logger.experiment.log_histogram_3d(error_tensor.cpu().numpy() / max_error, name='hist_' + error_str,
#                                                     step=pl_module.current_epoch)
#             error_dict[error_str] = curr_loss
#         self.metrics = {**self.metrics, **error_dict}
#         for i in pl_module.val_batch_list.values():
#             i.clear()


def L1_normalized_loss(min, max):
    return lambda x, y: torch.mean(F.l1_loss(x, y, reduction='none'), dim=0) / torch.tensor((max - min),
                                                                                            device=x.device)


def run_resnet_synth(num_input_layers, num_outputs,
                     comment, train_db_path, val_db_path, val_split, transform, out_transform, mean_error_func_dict,
                     hist_error_func_dict, text_error_func_dict, output_scaling=1e4, lr=1e-2,
                     resnet_type='18', train_cache_size=5500, val_cache_size=1000, batch_size=64, num_epochs=1000,
                     weight_decay=0, cosine_annealing_steps=10, loss_func=F.smooth_l1_loss, subsampler_size=640,
                     dtype=torch.float32, track_ideal_metrics=False):
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
              'weight_decay': weight_decay, 'cosine_annealing_steps': cosine_annealing_steps}
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
    val_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=4)
    model = CustomInputResnet(num_input_layers, num_outputs, loss_func=loss_func, output_scaling=output_scaling,
                              reduce_error_func_dict=mean_error_func_dict,
                              hist_error_func_dict=hist_error_func_dict,
                              text_error_func_dict=text_error_func_dict,
                              resnet_type=resnet_type, learning_rate=lr,
                              cosine_annealing_steps=10, weight_decay=weight_decay, dtype=dtype,
                              track_ideal_metrics=track_ideal_metrics)

    logger = CometLogger(api_key="sjNiwIhUM0j1ufNwaSjEUHHXh", project_name="AeroVision",
                         experiment_name=comment)
    logger.log_hyperparams(params=params)
    checkpoints_folder = f"./checkpoints/{comment}/"
    if os.path.isdir(checkpoints_folder):
        shutil.rmtree(checkpoints_folder)
    else:
        Path(checkpoints_folder).mkdir(parents=True, exist_ok=True)
    mcp = ModelCheckpoint(
        filepath=checkpoints_folder + "{epoch}",
        save_last=True,
        save_top_k=10,
        period=-1,
        monitor='val_loss',
        verbose=True)

    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs,
                         callbacks=[mcp],
                         num_sanity_val_steps=0,
                         profiler=True, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    logger.experiment.log_asset_folder(checkpoints_folder)
