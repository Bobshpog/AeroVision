from functools import partial

import h5py
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

from src.model_datasets.resnet_sin_func import SinFunctionDataset
import src.util.image_transforms as my_transforms
from src.util.loss_functions import MSE_Weighted


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, resnet_type='18', learning_rate=1e-2,
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
        self.min_train_amp_err = None
        self.min_train_decay_err = None
        self.min_train_freq_err = None
        self.min_val_loss = None
        self.min_val_amp_err = None
        self.min_val_decay_err = None
        self.min_val_freq_err = None
        self.train_batch_list = {'loss': [], 'amp_err': [], 'decay_err': [], 'freq_err': []}
        self.val_batch_list = {'loss': [], 'amp_err': [], 'decay_err': [], 'freq_err': []}
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
            one = torch.ones(y.shape[0], dtype=y.dtype, device=y.device)
            self.train_batch_list['loss'].append(loss)
            self.train_batch_list['amp_err'].append(self.loss_func(y_hat[:, 0] / y[:, 0], one))
            self.train_batch_list['decay_err'].append(self.loss_func(y_hat[:, 1] / y[:, 1], one))
            self.train_batch_list['freq_err'].append(self.loss_func(y_hat[:, 2] / y[:, 2], one))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        with torch.no_grad():
            one = torch.ones(y.shape[0], dtype=y.dtype, device=y.device)
            self.val_batch_list['loss'].append(loss)
            self.val_batch_list['amp_err'].append(self.loss_func(y_hat[:, 0] / y[:, 0], one))
            self.val_batch_list['decay_err'].append(self.loss_func(y_hat[:, 1] / y[:, 1], one))
            self.val_batch_list['freq_err'].append(self.loss_func(y_hat[:, 2] / y[:, 2], one))
        return loss


class LoggerCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule):
        curr_loss = torch.mean(torch.stack(pl_module.train_batch_list['loss']))
        curr_amp_err = torch.mean(torch.stack(pl_module.train_batch_list['amp_err']))
        curr_decay_err = torch.mean(torch.stack(pl_module.train_batch_list['decay_err']))
        curr_freq_err = torch.mean(torch.stack(pl_module.train_batch_list['freq_err']))
        pl_module.min_train_loss = torch.min(curr_loss,
                                             pl_module.min_train_loss) if pl_module.min_train_loss else curr_loss
        pl_module.min_train_amp_err = torch.min(curr_amp_err,
                                                pl_module.min_train_amp_err) if pl_module.min_train_amp_err else curr_amp_err
        pl_module.min_train_decay_err = torch.min(curr_decay_err,
                                                  pl_module.min_train_decay_err) if pl_module.min_train_decay_err else curr_decay_err
        pl_module.min_train_freq_err = torch.min(curr_freq_err,
                                                 pl_module.min_train_freq_err) if pl_module.min_train_freq_err else curr_freq_err
        metrics = {
            'train min loss': pl_module.min_train_loss,
            'train min amplitude error': pl_module.min_train_amp_err,
            'train min decay error': pl_module.min_train_decay_err,
            'train min frequency error': pl_module.min_train_freq_err,
            'train loss': curr_loss,
            'train amplitude error': curr_amp_err,
            'train decay error': curr_decay_err,
            'train frequency error': curr_freq_err,
        }
        pl_module.logger.log_metrics(metrics, pl_module.current_epoch)
        for i in pl_module.train_batch_list.values():
            i.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        curr_loss = torch.mean(torch.stack(pl_module.val_batch_list['loss']))
        curr_amp_err = torch.mean(torch.stack(pl_module.val_batch_list['amp_err']))
        curr_decay_err = torch.mean(torch.stack(pl_module.val_batch_list['decay_err']))
        curr_freq_err = torch.mean(torch.stack(pl_module.val_batch_list['freq_err']))
        pl_module.min_val_loss = torch.min(curr_loss,
                                           pl_module.min_val_loss) if pl_module.min_val_loss else curr_loss
        pl_module.min_val_amp_err = torch.min(curr_amp_err,
                                              pl_module.min_val_amp_err) if pl_module.min_val_amp_err else curr_amp_err
        pl_module.min_val_decay_err = torch.min(curr_decay_err,
                                                pl_module.min_val_decay_err) if pl_module.min_val_decay_err else curr_decay_err
        pl_module.min_val_freq_err = torch.min(curr_freq_err,
                                               pl_module.min_val_freq_err) if pl_module.min_val_freq_err else curr_freq_err
        metrics = {
            'val min loss': pl_module.min_val_loss,
            'val min amplitude error': pl_module.min_val_amp_err,
            'val min decay error': pl_module.min_val_decay_err,
            'val min frequency error': pl_module.min_val_freq_err,
            'val loss': curr_loss,
            'val amplitude error': curr_amp_err,
            'val decay error': curr_decay_err,
            'val frequency error': curr_freq_err,
        }
        pl_module.logger.log_metrics(metrics, pl_module.current_epoch)
        for i in pl_module.val_batch_list.values():
            i.clear()


if __name__ == '__main__':
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    VAL_CACHE_SIZE = 1000
    TRAIN_CACHE_SIZE = 5500
    EXPERIMENT_NAME = ""
    TRAINING_DB_PATH = "data/databases/20201002-083303__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
    VALIDATION_DB_PATH = "data/databases/20201002-095619__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
    with h5py.File(TRAINING_DB_PATH, 'r') as hf:
        mean_image = my_transforms.slice_first_position_no_depth(hf['generator metadata']['mean images'])
    remove_mean = partial(my_transforms.remove_dc_photo, mean_image)
    transform = transforms.Compose([my_transforms.slice_first_position_no_depth,
                                    remove_mean,
                                    my_transforms.last_axis_to_first])
    train_dset = SinFunctionDataset(TRAINING_DB_PATH,
                                    transform=transform, cache_size=TRAIN_CACHE_SIZE)
    val_dset = SinFunctionDataset(VALIDATION_DB_PATH,
                                  transform=transform, cache_size=VAL_CACHE_SIZE)
    train_loader = DataLoader(train_dset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, BATCH_SIZE, shuffle=False, num_workers=4)
    model = CustomInputResnet(3, 3, loss_func=F.mse_loss, resnet_type='18', cosine_annealing_steps=10)
    logger = TensorBoardLogger('lightning_logs',name=EXPERIMENT_NAME)
    mcp = ModelCheckpoint(
        filepath=f"{logger.log_dir}/checkpoints" + "{epoch}_tl_{train_batch_list['loss']:.3f}_vl_{train_batch_list['loss']:.3f}",
        save_last=True, mode='min', )

    trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS, callbacks=[LoggerCallback()], checkpoint_callback=mcp,
                         num_sanity_val_steps=0,
                         profiler=True, logger=logger)
    trainer.fit(model, train_loader, val_loader)
