import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from dataloaders import *

def getDataset():
    dataset = datasets.MNIST('./data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.Resize((32, 32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'], 'Loader has to be either \'iid\' or \'non_overlap_label \''
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
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                             transform=transforms.Compose(
                                                                 [transforms.Resize((32, 32)), transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))])),
                                              batch_size=test_batch_size, shuffle=True)
    return test_loader