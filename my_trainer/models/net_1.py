import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(pl.LightningModule):
    def __init__(
        self
    ):
        super().__init__()
        # self.model = model
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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

