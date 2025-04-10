import torch
from torchvision import transforms, datasets
import pickle

import numpy as np


class Partition(torch.utils.data.Dataset):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.classes = 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        data_idx = self.index[i]
        return self.data[data_idx]


class customDataLoader():
    """ Virtual class: load a particular partition of dataset"""

    def __init__(self, size, dataset, bsz):
        '''
        size: number of paritions in the loader
        dataset: pytorch dataset
        bsz: batch size of the data loader
        '''
        self.size = size
        self.dataset = dataset
        self.classes = np.unique(dataset.targets).tolist()
        self.bsz = bsz
        self.partition_list = self.getPartitions()
        num_unique_items = len(np.unique(np.concatenate(self.partition_list)))
        if (len(dataset) != num_unique_items):
            print(
                f"Number of unique items in partitions ({num_unique_items}) is not equal to the size of dataset ({len(dataset)}), some data may not be included")

    def getPartitions(self):
        raise NotImplementedError()

    def __len__(self):
        return self.size

    def __getitem__(self, rank):
        assert rank < self.size, 'partition index should be smaller than the size of the partition'
        partition = Partition(self.dataset, self.partition_list[rank])
        partition.classes = self.classes
        train_set = torch.utils.data.DataLoader(partition, batch_size=int(self.bsz), shuffle=True,
                                                drop_last=True)  # drop last since some network requires batchnorm
        return train_set


class iidLoader(customDataLoader):
    def __init__(self, size, dataset, bsz=128):
        super(iidLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)
        indexes = [x for x in range(0, data_len)]
        np.random.shuffle(indexes)
        # fractions of data in each partition
        partition_sizes = [1.0 / self.size for _ in range(self.size)]

        partition_list = []
        for frac in partition_sizes:
            part_len = int(frac * data_len)
            partition_list.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        return partition_list


class byLabelLoader(customDataLoader):
    def __init__(self, size, dataset, bsz=128):
        super(byLabelLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)

        partition_list = []
        self.labels = np.unique(self.dataset.targets).tolist()
        label = self.dataset.targets
        label = torch.tensor(np.array(label))
        for i in self.labels:
            label_iloc = (label == i).nonzero(as_tuple=False).squeeze().tolist()
            partition_list.append(label_iloc)
        return partition_list


class dirichletLoader(customDataLoader):
    def __init__(self, size, dataset, alpha=0.9, bsz=128):
        # alpha is used in getPartition,
        # and getPartition is used in parent constructor
        # hence need to initialize alpha first
        self.alpha = alpha
        super(dirichletLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)

        partition_list = [[] for j in range(self.size)]
        self.labels = np.unique(self.dataset.targets).tolist()
        label = self.dataset.targets
        label = torch.tensor(np.array(label))
        for i in self.labels:
            label_iloc = (label == i).nonzero(as_tuple=False).squeeze().numpy()
            np.random.shuffle(label_iloc)
            p = np.random.dirichlet([self.alpha] * self.size)
            # choose which partition a data is assigned to
            assignment = np.random.choice(range(self.size), size=len(label_iloc), p=p.tolist())
            part_list = [(label_iloc[(assignment == k)]).tolist() for k in range(self.size)]
            for j in range(self.size):
                partition_list[j] += part_list[j]
        return partition_list

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
