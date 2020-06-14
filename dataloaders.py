import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from typing import Sequence

def get_STL10_dataloaders(batch_size=128):
    """STL-10 dataloader with (96,96) sized images"""
    all_transforms = transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor()
    ])

    unlabeled_train_data = datasets.STL10('../stl10_data', split='unlabeled', download=True, transform=all_transforms)
    train_data = datasets.STL10('../stl10_data', split='train', transform=all_transforms)
    test_data = datasets.STL10('../stl10_data',  split='test', transform=all_transforms)

    unlabeled_train_loader = DataLoader(unlabeled_train_data, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return unlabeled_train_loader,train_loader, test_loader

