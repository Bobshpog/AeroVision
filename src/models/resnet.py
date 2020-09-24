from functools import partial

import h5py
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms

from model_datasets.resnet_sin_func import SinFunctionDataset
import src.models.transforms as my_transforms


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, learning_rate=1e-2, cosine_annealing_steps=0,
                 weight_decay=0):
        super().__init__()
        # TODO consider removing pretrained
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.resnet = models.resnet18(pretrained=False, num_classes=num_outputs)
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
        result = pl.TrainResult(loss)
        diff = y_hat - y
        d_amp = diff[0]
        d_decay = diff[1]
        d_freq = diff[2]
        result.log('train amp distance', d_amp.mean())
        result.log('train decay distance', d_decay.mean())
        result.log('train frequency distance', d_freq.mean())
        result.log('train amp accuracy', (d_amp / y[0]).mean())
        result.log('train decay accuracy', (d_decay / y[1]).mean())
        result.log('train frequency accuracy', (d_freq / y[2]).mean())
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        diff = y_hat - y
        d_amp = diff[0]
        d_decay = diff[1]
        d_freq = diff[2]
        result.log('val_loss', loss)
        result.log('val amp distance', d_amp.mean())
        result.log('val decay distance', d_decay.mean())
        result.log('val frequency distance', d_freq.mean())
        result.log('val amp accuracy', (d_amp / y[0]).mean())
        result.log('val decay accuracy', (d_decay / y[1]).mean())
        result.log('val frequency accuracy', (d_freq / y[2]).mean())
        return result


if __name__ == '__main__':
    BATCH_SIZE = 90
    NUM_EPOCHS = 50
    TRAINING_DB_PATH = "data/databases/20200923-101734__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
    VALIDATION_DB_PATH = "data/databases/20200922-125422__SyntheticSineDecayingGen(mesh_wing='finished_fem_without_tip', mesh_tip='fem_tip', resolution=[640, 480], texture_path='checkers2.png'.hdf5"
    with h5py.File(TRAINING_DB_PATH, 'r') as hf:
        mean_image = my_transforms.slice_first_position_no_depth(hf['generator metadata']['mean images'])
    remove_mean = partial(my_transforms.remove_dc_photo, mean_image)
    transform = transforms.Compose([my_transforms.slice_first_position_no_depth,
                                    remove_mean,
                                    my_transforms.double_to_float,
                                    my_transforms.last_axis_to_first])
    train_dset = SinFunctionDataset(TRAINING_DB_PATH,
                                    transform=transform)
    val_dset = SinFunctionDataset(VALIDATION_DB_PATH,
                                  transform=transform)
    train_loader = DataLoader(train_dset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, BATCH_SIZE, shuffle=False, num_workers=4)
    model = CustomInputResnet(3, 3, F.mse_loss, cosine_annealing_steps=10)
    trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS)
    trainer.fit(model, train_loader, val_loader)
