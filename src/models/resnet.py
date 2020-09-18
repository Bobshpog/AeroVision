import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

# constants
NUM_EPOCHS = 50
BATCH_SIZE = 128


class Autoencoder(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs,loss_func, learning_rate=1e-3):
        super().__init__()
        # TODO consider removing pretrained
        self.learning_rate = learning_rate
        self.loss_func=loss_func
        self.resnet = models.resnet50(pretrained=True, num_classes=num_outputs)
        # altering resnet to fit more than 3 input layers
        self.resnet.conv1 = nn.Conv2d(num_input_layers, self.resnet.inplanes, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        # TODO  Consider adding scrubbed FC layer

    def forward(self, x):
        x = self.resnet(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat,y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result
