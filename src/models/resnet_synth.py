from collections import defaultdict
from functools import partial

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from pytorch_lightning import Callback
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model_datasets.image_dataset import ImageDataset
import src.util.image_transforms as my_transforms
from src.util.loss_functions import mse_weighted, vertice_mean_rms


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, output_loss_func, vertice_mean_rms_loss_func=None,
                 resnet_type='18', learning_rate=1e-2,
                 cosine_annealing_steps=0,
                 weight_decay=0):
        super().__init__()
        # TODO consider removing pretrained
        resnet_dict = {'18': models.resnet18,
                       '34': models.resnet34,
                       '50': models.resnet50}
        self.num_input_layers = num_input_layers
        self.num_output_layers = num_outputs
        self.loss_func = loss_func
        self.output_loss_func = output_loss_func
        self.vertice_mean_rms_loss_func = vertice_mean_rms_loss_func
        self.learning_rate = learning_rate
        self.resnet_type = resnet_type
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.min_train_loss = None
        self.min_val_loss = None
        self.min_train_vertice_rms = None
        self.min_val_vertice_rms=None
        self.train_min_errors = defaultdict(lambda: None)
        self.val_min_errors = defaultdict(lambda: None)
        self.train_batch_list = defaultdict(list)
        self.val_batch_list = defaultdict(list)

        self.resnet = resnet_dict[resnet_type](pretrained=False, num_classes=num_outputs)
        # altering resnet to fit more than 3 input layers
        self.resnet.conv1 = nn.Conv2d(num_input_layers, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        # TODO  Consider adding scrubbed FC layer

    def forward(self, x):
        x = self.resnet(x)
        return x

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.cosine_annealing_steps:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(adam, self.cosine_annealing_steps,
                                                             self.learning_rate / 1000)
        else:
            scheduler = None
        return [adam], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            output_loss = self.output_loss_func(y_hat, y)
            means = torch.mean(y_hat, dim=0)
            variance = torch.var(y_hat, dim=0)
            for i in range(self.num_output_layers):
                self.train_batch_list[f'scale{i}'].append(output_loss[i])
                self.train_batch_list[f'mean{i}'].append(means[i])
                self.train_batch_list[f'var{i}'].append(variance[i])
            self.train_batch_list['loss'].append(loss)
            if self.vertice_mean_rms_loss_func:
                self.train_batch_list['mean_vertice_rms'].append(self.vertice_mean_rms_loss_func(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            output_loss = self.output_loss_func(y_hat, y)
            means = torch.mean(y_hat, dim=0)
            variance = torch.var(y_hat, dim=0)
            for i in range(self.num_output_layers):
                self.val_batch_list[f'scale{i}'].append(output_loss[i])
                self.val_batch_list[f'mean{i}'].append(means[i])
                self.val_batch_list[f'var{i}'].append(variance[i])
            self.val_batch_list['loss'].append(loss)
            if self.vertice_mean_rms_loss_func:
                self.val_batch_list['mean_vertice_rms'].append(self.vertice_mean_rms_loss_func(y_hat, y))
        return loss


class LoggerCallback(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module: CustomInputResnet):
        curr_loss = torch.mean(torch.stack(pl_module.train_batch_list['loss']))
        if pl_module.vertice_mean_rms_loss_func:
            curr_vertice_rms = torch.mean(torch.stack(pl_module.train_batch_list['mean_vertice_rms']))
        error_dict = {}
        means_dict = {}
        var_dict = {}
        for i in range(pl_module.num_output_layers):
            error_dict[f'scale{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'scale{i}']))
            means_dict[f'mean{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'mean{i}']))
            var_dict[f'var{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'var{i}']))

        pl_module.min_train_loss = torch.min(curr_loss,
                                             pl_module.min_train_loss) if pl_module.min_train_loss else curr_loss
        pl_module.min_train_vertice_rms = torch.min(curr_vertice_rms,
                                                    pl_module.min_train_vertice_rms) if pl_module.min_train_vertice_rms else curr_vertice_rms

        for i in range(pl_module.num_output_layers):
            pl_module.train_min_errors[f'scale{i}'] = torch.min(pl_module.train_min_errors[f'scale{i}'],
                                                                error_dict[f'scale{i}']) if pl_module.train_min_errors[
                f'scale{i}'] else error_dict[f'scale{i}']

        self.logger.experiment.add_scalars('loss', {'train_loss': curr_loss}, pl_module.current_epoch)
        self.logger.experiment.add_scalars('min_loss', {'train': pl_module.min_train_loss}, pl_module.current_epoch)
        self.logger.experiment.add_scalars('train_error',
                                           error_dict, pl_module.current_epoch)
        self.logger.experiment.add_scalars('train_min_error',
                                           pl_module.train_min_errors, pl_module.current_epoch)
        self.logger.experiment.add_scalars('train_means',
                                           means_dict, pl_module.current_epoch)
        self.logger.experiment.add_scalars('train_variance',
                                           var_dict, pl_module.current_epoch)
        if pl_module.vertice_mean_rms_loss_func:
            self.logger.experiment.add_scalars('mean_vertice_rms', {'train': curr_vertice_rms}, pl_module.current_epoch)
            self.logger.experiment.add_scalars('min_mean_vertice_rms', {'train': pl_module.min_train_vertice_rms},
                                               pl_module.current_epoch)

        for i in pl_module.train_batch_list.values():
            i.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        curr_loss = torch.mean(torch.stack(pl_module.val_batch_list['loss']))

        if pl_module.vertice_mean_rms_loss_func:
            curr_vertice_rms = torch.mean(torch.stack(pl_module.val_batch_list['mean_vertice_rms']))
        error_dict = {}
        means_dict = {}
        var_dict = {}
        for i in range(pl_module.num_output_layers):
            error_dict[f'scale{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'scale{i}']))
            means_dict[f'mean{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'mean{i}']))
            var_dict[f'var{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'var{i}']))
        pl_module.min_val_loss = torch.min(curr_loss,
                                           pl_module.min_val_loss) if pl_module.min_val_loss else curr_loss
        pl_module.min_val_vertice_rms = torch.min(curr_vertice_rms,
                                                  pl_module.min_val_vertice_rms) if pl_module.min_val_vertice_rms else curr_vertice_rms
        for i in range(pl_module.num_output_layers):
            pl_module.val_min_errors[f'scale{i}'] = torch.min(pl_module.val_min_errors[f'scale{i}'],
                                                              error_dict[f'scale{i}']) if pl_module.val_min_errors[
                f'scale{i}'] else error_dict[f'scale{i}']

        self.logger.experiment.add_scalars('loss', {'val_loss': curr_loss}, pl_module.current_epoch)
        self.logger.experiment.add_scalars('min_loss', {'val': pl_module.min_val_loss}, pl_module.current_epoch)
        self.logger.experiment.add_scalars('val_error',
                                           error_dict, pl_module.current_epoch)
        self.logger.experiment.add_scalars('val_means',
                                           means_dict, pl_module.current_epoch)
        self.logger.experiment.add_scalars('val_variance',
                                           var_dict, pl_module.current_epoch)
        self.logger.experiment.add_scalars('val_min_error',
                                           pl_module.val_min_errors, pl_module.current_epoch)

        if pl_module.vertice_mean_rms_loss_func:
            self.logger.experiment.add_scalars('mean_vertice_rms', {'val': curr_vertice_rms}, pl_module.current_epoch)
            self.logger.experiment.add_scalars('min_mean_vertice_rms', {'val': pl_module.min_val_vertice_rms},
                                               pl_module.current_epoch)
        for i in pl_module.val_batch_list.values():
            i.clear()


def L1_normalized_loss(min, max):
    return lambda x, y: torch.mean(F.l1_loss(x, y, reduction='none'), dim=0) / torch.tensor((max - min),
                                                                                            device=x.device)


if __name__ == '__main__':
    BATCH_SIZE = None  # 16 for Resnet50, 64 for resnet 18
    NUM_EPOCHS = None
    VAL_CACHE_SIZE = 1000
    TRAIN_CACHE_SIZE = 5500  # around 6500 total images (640,480,3) total space
    NUM_INPUT_LAYERS = 3
    NUM_OUTPUTS = 5
    RESNET_TYPE = None  # '18', '50', '34'
    OUTPUT_LOSS_FUNC = None  # Initialized later
    LOSS_FUNC = F.smooth_l1_loss
    EXPERIMENT_NAME = ""
    TRAINING_DB_PATH = None
    VALIDATION_DB_PATH = None
    VAL_SPLIT = None
    TRANSFORM = my_transforms.top_middle_rgb
    OUTPUT_SCALING = 3

    if None in [BATCH_SIZE, NUM_EPOCHS, RESNET_TYPE, TRAINING_DB_PATH, VALIDATION_DB_PATH, VAL_SPLIT]:
        raise ValueError('Config not fully initialized')
    out_transform = transforms.Compose([partial(my_transforms.mul_by_10_power, OUTPUT_SCALING)])
    with h5py.File(TRAINING_DB_PATH, 'r') as hf:
        mean_image = hf['generator metadata']['mean images'][()]
        vertice_mean_rms_loss_func = partial(vertice_mean_rms, hf['generator metadata']['modal shapes'][()],
                                             OUTPUT_SCALING)
        min_scales = out_transform(np.min(hf['data']['scales'], axis=0))
        max_scales = out_transform(np.max(hf['data']['scales'], axis=0))
    OUTPUT_LOSS_FUNC = L1_normalized_loss(min_scales, max_scales)
    transform = TRANSFORM(mean_image)
    train_dset = ImageDataset(TRAINING_DB_PATH,
                              transform=transform, out_transform=out_transform, cache_size=TRAIN_CACHE_SIZE,
                              max_index=VAL_SPLIT)
    val_dset = ImageDataset(VALIDATION_DB_PATH,
                            transform=transform, out_transform=out_transform, cache_size=VAL_CACHE_SIZE,
                            min_index=VAL_SPLIT)
    train_loader = DataLoader(train_dset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, BATCH_SIZE, shuffle=False, num_workers=4)
    model = CustomInputResnet(NUM_INPUT_LAYERS, NUM_OUTPUTS, loss_func=LOSS_FUNC, output_loss_func=OUTPUT_LOSS_FUNC,
                              vertice_mean_rms_loss_func=vertice_mean_rms_loss_func,
                              resnet_type=RESNET_TYPE,
                              cosine_annealing_steps=10)
    logger = TensorBoardLogger('lightning_logs', name=EXPERIMENT_NAME)
    mcp = ModelCheckpoint(
        filepath=f"{logger.log_dir}/checkpoints/"
                 + "{epoch}",
        save_last=True,
        save_top_k=-1,
        period=5,
        verbose=True)

    trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS, callbacks=[LoggerCallback(logger)],
                         checkpoint_callback=mcp,
                         num_sanity_val_steps=0,
                         profiler=True, logger=logger)
    trainer.fit(model, train_loader, val_loader)
