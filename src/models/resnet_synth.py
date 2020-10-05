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

from src.model_datasets.resnet_sin_func import ImageDataset
import src.util.image_transforms as my_transforms
from src.util.loss_functions import MSE_Weighted


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, output_loss_func, resnet_type='18', learning_rate=1e-2,
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
        self.learning_rate = learning_rate
        self.resnet_type = resnet_type
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.min_train_loss = None
        self.min_val_loss = None
        self.train_min_errors = defaultdict(None)
        self.val_min_scales = defaultdict(None)
        self.train_batch_list = {}
        self.val_batch_list = {}
        for i in range(self.num_output_layers):
            self.train_batch_list[f'scale{i}'] = []
            self.val_batch_list[f'scale{i}'] = []
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
            for i in range(self.num_output_layers):
                self.train_batch_list[f'scale{i}'].append(self.output_loss_func(y_hat[:, i], y[:, i]))
            self.train_batch_list['loss'].append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            for i in range(self.num_output_layers):
                self.val_batch_list[f'scale{i}'].append(self.output_loss_func(y_hat[:, i], y[:, i]))
            self.val_batch_list['loss'].append(loss)
        return loss


class LoggerCallback(Callback):
    def __init__(self, logger, min_scales, max_scales):
        self.logger = logger
        self.range_scales = torch.tensor(max_scales - min_scales + np.finfo(np.float32).eps).cuda()
        # self.min_scales = min_scales
        # self.max_Scales = max_scales

    def on_train_epoch_end(self, trainer, pl_module: CustomInputResnet):
        curr_loss = torch.mean(torch.stack(pl_module.train_batch_list['loss']))
        error_dict = {}
        for i in range(pl_module.num_output_layers):
            error_dict[f'scale{i}'] = torch.mean(torch.stack(pl_module.train_batch_list[f'scale{i}'])) / \
                                      self.range_scales[i]

        pl_module.min_train_loss = torch.min(curr_loss,
                                             pl_module.min_train_loss) if pl_module.min_train_loss else curr_loss

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

        for i in pl_module.train_batch_list.values():
            i.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        curr_loss = torch.mean(torch.stack(pl_module.val_batch_list['loss']))
        error_dict = {}
        for i in range(pl_module.num_output_layers):
            error_dict[f'scale{i}'] = torch.mean(torch.stack(pl_module.val_batch_list[f'scale{i}'])) / \
                                      self.range_scales[i]

        pl_module.min_val_loss = torch.min(curr_loss,
                                           pl_module.min_val_loss) if pl_module.min_val_loss else curr_loss

        for i in range(pl_module.num_output_layers):
            pl_module.val_min_errors[f'scale{i}'] = torch.min(pl_module.val_min_errors[f'scale{i}'],
                                                              error_dict[f'scale{i}']) if pl_module.val_min_errors[
                f'scale{i}'] else error_dict[f'scale{i}']

        self.logger.experiment.add_scalars('loss', {'val_loss': curr_loss}, pl_module.current_epoch)
        self.logger.experiment.add_scalars('min_loss', {'val': pl_module.min_val_loss}, pl_module.current_epoch)
        self.logger.experiment.add_scalars('val_error',
                                           error_dict, pl_module.current_epoch)
        self.logger.experiment.add_scalars('val_min_error',
                                           pl_module.val_min_errors, pl_module.current_epoch)

        for i in pl_module.val_batch_list.values():
            i.clear()


if __name__ == '__main__':
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    VAL_CACHE_SIZE = 1000
    TRAIN_CACHE_SIZE = 5500
    NUM_INPUT_LAYERS = 3
    NUM_OUTPUTS = 5
    EXPERIMENT_NAME = ""
    TRAINING_DB_PATH = "data/databases/20201002-083303__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
    VALIDATION_DB_PATH = "data/databases/20201002-095619__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
    with h5py.File(TRAINING_DB_PATH, 'r') as hf:
        mean_image = my_transforms.slice_first_position_no_depth(hf['generator metadata']['mean images'])
        min_scales = np.min(hf['data']['scales'], axis=0)
        max_scales = np.max(hf['data']['scales'], axis=0)
    remove_mean = partial(my_transforms.remove_dc_photo, mean_image)
    transform = transforms.Compose([my_transforms.slice_first_position_no_depth,
                                    remove_mean,
                                    my_transforms.last_axis_to_first])
    train_dset = ImageDataset(TRAINING_DB_PATH,
                              transform=transform, cache_size=TRAIN_CACHE_SIZE, max_index=900)
    val_dset = ImageDataset(VALIDATION_DB_PATH,
                            transform=transform, cache_size=VAL_CACHE_SIZE, min_index=900)
    train_loader = DataLoader(train_dset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, BATCH_SIZE, shuffle=False, num_workers=4)
    model = CustomInputResnet(NUM_INPUT_LAYERS, NUM_OUTPUTS, loss_func=F.mse_loss, output_loss_func=F.l1_loss,
                              resnet_type='18',
                              cosine_annealing_steps=10)
    logger = TensorBoardLogger('lightning_logs', name=EXPERIMENT_NAME)
    mcp = ModelCheckpoint(
        filepath=f"{logger.log_dir}/checkpoints/"
                 + "{epoch}",
        save_last=True,
        save_top_k=-1,
        period=5,
        verbose=True)

    trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS, callbacks=[LoggerCallback(logger, min_scales, max_scales)],
                         checkpoint_callback=mcp,
                         num_sanity_val_steps=0,
                         profiler=True, logger=logger)
    trainer.fit(model, train_loader, val_loader)
