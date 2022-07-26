import torch
import torchvision
import pytorch_lightning as pl
import torchvision.transforms as transforms

class CifarDataset(pl.LightningDataModule):
        def __init__(
            self,
            train_batch_size=120,
            test_batch_size=120,
            data_dir='./data'
        ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def prepare_data(self):
        self.trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True,
            download=True, transform=self.transform
        )
        self.testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False,
            download=True, transform=self.transform
        )
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.train_batch_size, 
            shuffle=True, num_workers=32
        )
        return trainloader

    def val_dataloader(self):
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.test_batch_size, 
            shuffle=False, num_workers=32
        )
        return testloader
