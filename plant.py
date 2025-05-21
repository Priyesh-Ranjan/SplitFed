from __future__ import print_function

import pickle

import torch
from torchvision import datasets, transforms
#import torchvision.models as models
#import pytorch_lightning as pl
#import torch.nn.functional as F

from dataloaders import iidLoader, byLabelLoader, dirichletLoader

IMG_SIZE = (224, 224)

def getDataset():
    dataset = datasets.ImageFolder(root='./data/plant/small',
                               transform = transforms.Compose([
    transforms.Resize((IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()]))
    return dataset


def basic_loader(num_clients, loader_type, dist):
    dataset = getDataset()
    if loader_type == dirichletLoader :
        return loader_type(num_clients, dataset, alpha=dist)
    else :
        return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, dist=0.9, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel',
                           'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type, dist)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type, dist)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/plant/small', transform = transforms.Compose([
    transforms.Resize((IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])), batch_size=test_batch_size, shuffle=True)
    return test_loader