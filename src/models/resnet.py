import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# constants
NUM_EPOCHS = 50
BATCH_SIZE = 128


class CustomInputResnet(pl.LightningModule):
    def __init__(self, num_input_layers, num_outputs, loss_func, learning_rate=1e-2, cosine_annealing_steps=0,
                 weight_decay=0):
        super().__init__()
        # TODO consider removing pretrained
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.cosine_annealing_steps = cosine_annealing_steps
        self.weight_decay = weight_decay
        self.resnet = models.resnet18(pretrained=True, num_classes=num_outputs)
        # altering resnet to fit more than 3 input layers
        self.resnet.conv1 = nn.Conv2d(num_input_layers, self.resnet.inplanes, kernel_size=7, stride=2, padding=3,
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result
