import torch
from torchvision import transforms, datasets
import pickle

from dataloaders import iidLoader, dirichletLoader, byLabelLoader

import numpy as np

def getDataset():
    dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([transforms.Resize((256, 256)),transforms.CenterCrop(224),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
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
            loader = basic_loader(num_clients, loader_type)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.Resize((256, 256)),transforms.CenterCrop(224),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                   (0.5, 0.5, 0.5))])),
        batch_size=test_batch_size, shuffle=True)
    return test_loader
