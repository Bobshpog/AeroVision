import os
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import h5py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader

import src.util.image_transforms as my_transforms
from src.model_datasets.image_dataset import ImageDataset
from src.util.error_helper_functions import calc_errors
from src.util.loss_functions import l2_norm


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, error_funcs, output_scaling,
                 resnet_type, learning_rate,
                 cosine_annealing_steps,
                 weight_decay):
        super().__init__()
        # TODO consider removing pretrained
        resnet_dict = {'18': models.resnet18,
                       '34': models.resnet34,
                       '50': models.resnet50}
        self.num_input_layers = num_input_layers
        self.num_output_layers = num_outputs
        self.loss_func = loss_func
        self.l1_error_func, self.l2_error_func = error_funcs
        self.output_scale = output_scaling
        self.learning_rate = learning_rate
        self.resnet_type = resnet_type
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.train_min_errors = defaultdict(lambda: None)
        self.val_min_errors = defaultdict(lambda: None)
        self.train_batch_list = defaultdict(list)
        self.val_batch_list = defaultdict(list)
        self.error_metrics = ['loss', 'l1_3d_loss', 'l2_3d_loss', 'l1_3d_ir_loss', 'l2_3d_ir_loss',
                              'l1_reg_avg', 'l2_reg_avg']
        self.resnet = resnet_dict[resnet_type](pretrained=False, num_classes=num_outputs)
        # altering resnet to fit more than 3 input layers
        self.resnet.conv1 = nn.Conv2d(num_input_layers, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)

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
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            l1_3d_err, l1_3d_ir_err, l1_regression_avg, l1_regression_list = self.l1_error_func(y, y_hat)
            l2_3d_err, l2_3d_ir_err, l2_regression_avg, l2_regression_list = self.l2_error_func(y, y_hat)
            means = torch.mean(y_hat, dim=0) / self.output_scale
            variance = torch.var(y_hat, dim=0) / (self.output_scale ** 2)
            for i in range(self.num_output_layers):
                self.train_batch_list[f'train_l1_scale{i}'].append(l1_regression_list[i] / self.output_scale)
                self.train_batch_list[f'train_l2_scale{i}'].append(l2_regression_list[i] / self.output_scale)
                self.train_batch_list[f'train_mean{i}'].append(means[i])
                self.train_batch_list[f'train_var{i}'].append(variance[i])
            self.train_batch_list['train_loss'].append(loss)
            self.train_batch_list['train_l1_3d_loss'].append(l1_3d_err)
            self.train_batch_list['train_l2_3d_loss'].append(l2_3d_err)
            self.train_batch_list['train_l1_3d_ir_loss'].append(l1_3d_ir_err)
            self.train_batch_list['train_l2_3d_ir_loss'].append(l2_3d_ir_err)
            self.train_batch_list['train_l1_reg_avg'].append(l1_regression_avg)
            self.train_batch_list['train_l2_reg_avg'].append(l2_regression_avg)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            l1_3d_err, l1_3d_ir_err, l1_regression_avg, l1_regression_list = self.l1_error_func(y, y_hat)
            l2_3d_err, l2_3d_ir_err, l2_regression_avg, l2_regression_list = self.l2_error_func(y, y_hat)
            means = torch.mean(y_hat, dim=0) / self.output_scale
            variance = torch.var(y_hat, dim=0) / (self.output_scale ** 2)
            for i in range(self.num_output_layers):
                self.val_batch_list[f'val_l1_scale{i}'].append(l1_regression_list[i] / self.output_scale)
                self.val_batch_list[f'val_l2_scale{i}'].append(l2_regression_list[i] / self.output_scale)
                self.val_batch_list[f'val_mean{i}'].append(means[i])
                self.val_batch_list[f'val_var{i}'].append(variance[i])
            self.val_batch_list['val_loss'].append(loss)
            self.val_batch_list['val_l1_3d_loss'].append(l1_3d_err)
            self.val_batch_list['val_l2_3d_loss'].append(l2_3d_err)
            self.val_batch_list['val_l1_3d_ir_loss'].append(l1_3d_ir_err)
            self.val_batch_list['val_l2_3d_ir_loss'].append(l2_3d_ir_err)
            self.val_batch_list['val_l1_reg_avg'].append(l1_regression_avg)
            self.val_batch_list['val_l2_reg_avg'].append(l2_regression_avg)
        return loss


class LoggerCallback(Callback):
    def __init__(self, logger):
        self.logger = logger
        self.metrics = {}

    def on_train_epoch_end(self, trainer, pl_module: CustomInputResnet):
        error_dict = {}
        means_dict = {}
        var_dict = {}
        for i in range(pl_module.num_output_layers):
            error_dict[f'train_l1_scale{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'train_l1_scale{i}']))
            error_dict[f'train_l2_scale{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'train_l2_scale{i}']))
            means_dict[f'train_mean{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'train_mean{i}']))
            var_dict[f'train_var{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'train_var{i}']))

        # for error_str in pl_module.error_metrics:
        #     error_str = f'train_{error_str}'
        #     curr_loss = torch.mean(torch.stack(pl_module.train_batch_list[error_str]))
        #     min_error_str = f'min_{error_str}'
        #     error_dict[error_str] = curr_loss
        #     old_min = pl_module.train_min_errors[min_error_str]
        #     pl_module.train_min_errors[min_error_str] = torch.min(curr_loss, old_min) if old_min else curr_loss
        # for norm in ['l1', 'l2']:
        #     for i in range(pl_module.num_output_layers):
        #         old_min = pl_module.train_min_errors[f'min_train_{norm}_scale{i}']
        #         curr_error = error_dict[f'train_{norm}_scale{i}']
        #         pl_module.train_min_errors[f'min_train_{norm}_scale{i}'] = torch.min(old_min,
        #                                                                              curr_error) if old_min else curr_error
        self.metrics = {**self.metrics, **error_dict}  # , **pl_module.train_min_errors}

        for i in pl_module.train_batch_list.values():
            i.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        error_dict = {}
        means_dict = {}
        var_dict = {}
        for i in range(pl_module.num_output_layers):
            error_dict[f'val_l1_scale{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'val_l1_scale{i}']))
            error_dict[f'val_l2_scale{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'val_l2_scale{i}']))
            means_dict[f'val_mean{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'val_mean{i}']))
            var_dict[f'val_var{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'val_var{i}']))

        # for error_str in pl_module.error_metrics:
        #     error_str = f'val_{error_str}'
        #     curr_loss = torch.mean(torch.stack(pl_module.val_batch_list[error_str]))
        #     min_error_str = f'min_{error_str}'
        #     error_dict[error_str] = curr_loss
        #     old_min = pl_module.val_min_errors[min_error_str]
        #     pl_module.val_min_errors[min_error_str] = torch.min(curr_loss, old_min) if old_min else curr_loss
        # for norm in ['l1', 'l2']:
        #     for i in range(pl_module.num_output_layers):
        #         old_min = pl_module.val_min_errors[f'min_val_{norm}_scale{i}']
        #         curr_error = error_dict[f'val_{norm}_scale{i}']
        #         pl_module.val_min_errors[f'min_val_{norm}_scale{i}'] = torch.min(old_min,
        #                                                                          curr_error) if old_min else curr_error

        self.metrics = {**self.metrics, **error_dict}  # , **pl_module.val_min_errors}
        for i in pl_module.val_batch_list.values():
            i.clear()

    def on_epoch_end(self, trainer, pl_module):
        self.logger.experiment.log_metrics(self.metrics, step=pl_module.current_epoch, epoch=pl_module.current_epoch)
        self.metrics.clear()


def L1_normalized_loss(min, max):
    return lambda x, y: torch.mean(F.l1_loss(x, y, reduction='none'), dim=0) / torch.tensor((max - min),
                                                                                            device=x.device)


def run_resnet_synth(num_input_layers, num_outputs,
                     comment, train_db_path, val_db_path, val_split, transform, output_scaling=1e-4, lr=1e-2,
                     resnet_type='18', train_cache_size=1000, val_cache_size=5500, batch_size=64, num_epochs=1000,
                     weight_decay=0, cosine_annealing_steps=10, loss_func=F.smooth_l1_loss, camera_ids=None):
    if None in [batch_size, num_epochs, resnet_type, train_db_path, val_db_path, val_split, comment]:
        raise ValueError('Config not fully initialized')
    params = {'batch_size': batch_size, 'train_db': train_db_path.split('/')[-1],
              'val_db': val_db_path.split('/')[-1], 'train-val_split_index': val_split,
              'loss_func': loss_func.__name__, 'img_transform': transform.__name__, 'num_outputs': num_outputs,
              'output_scaling': output_scaling, 'resnet_type': resnet_type, 'lr': lr,
              'weight_decay': weight_decay, 'cosine_annealing_steps': cosine_annealing_steps}
    out_transform = my_transforms.scale_by(output_scaling)
    with h5py.File(train_db_path, 'r') as hf:
        mean_image = hf['generator metadata']['mean images'][()]
        modal_shapes = hf['generator metadata']['modal shapes'][()]
        ir = hf['generator metadata'].attrs['ir'][()]
        db_size = hf['data']['images'].len()
        l1_errors_func = partial(calc_errors, F.l1_loss, modal_shapes, output_scaling, ir)
        l2_errors_func = partial(calc_errors, l2_norm, modal_shapes, output_scaling, ir)
    if camera_ids:
        transform = transform(camera_ids, mean_image)
    else:
        transform = transform(mean_image)
    if isinstance(val_split, int):
        train_dset = ImageDataset(train_db_path,
                                  transform=transform, out_transform=out_transform, cache_size=train_cache_size,
                                  max_index=val_split, camera_ids=camera_ids)
        val_dset = ImageDataset(val_db_path,
                                transform=transform, out_transform=out_transform, cache_size=val_cache_size,
                                min_index=val_split, camera_ids=camera_ids)
    else:
        train_split = set(range(db_size))
        train_split -= set(val_split)
        train_split = tuple(train_split)
        train_dset = ImageDataset(train_db_path,
                                  transform=transform, out_transform=out_transform, cache_size=train_cache_size,
                                  index_list=train_split, camera_ids=camera_ids)
        val_dset = ImageDataset(val_db_path,
                                transform=transform, out_transform=out_transform, cache_size=val_cache_size,
                                min_index=val_split, camera_ids=camera_ids)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=4)
    model = CustomInputResnet(num_input_layers, num_outputs, loss_func=loss_func, output_scaling=output_scaling,
                              error_funcs=(l1_errors_func,
                                           l2_errors_func),
                              resnet_type=resnet_type, learning_rate=lr,
                              cosine_annealing_steps=10, weight_decay=weight_decay)
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

    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, callbacks=[LoggerCallback(logger)],
                         checkpoint_callback=mcp,
                         num_sanity_val_steps=0,
                         profiler=True)
    trainer.fit(model, train_loader, val_loader)
    logger.experiment.log_asset_folder(checkpoints_folder)
