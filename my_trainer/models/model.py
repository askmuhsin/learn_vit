import torch
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
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        total = len(y)
        correct = sum(y_hat.argmax(axis=1)==y).item()
        acc = round(100 * correct / total, 2)
        
        self.log('training_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('training_acc', acc, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        total = len(y)
        correct = sum(y_hat.argmax(axis=1)==y).item()
        acc = round(100 * correct / total, 2)
        
        self.log('val_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('val_acc', acc, on_epoch=True, on_step=False)

    def test_step(self):
        pass
    
    def predict_step(self):
        pass
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

