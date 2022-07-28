import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        self.train_acc(y_hat, y)
        self.log('train_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        self.val_acc(y_hat, y)
        self.log('val_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc, on_epoch=True, on_step=False)

    def test_step(self):
        pass
    
    def predict_step(self):
        pass
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

