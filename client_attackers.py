from torch.utils.data import DataLoader
import torch
from utils import DatasetSplit
from server import evaluate_server, train_server
from random import random
from clients import *

class Attacker_LF(Client):
    def __init__(self, PDR, flip, net_client_model, idx, lr, device, local_ep = 1, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        super(Attacker_LF, self).__init__(net_client_model, idx, lr, device, local_ep = 1, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None)
        self.PDR = PDR
        self.flip = flip

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: self.flip[int(x)] if (x in list(self.flip.keys()) and random() <= self.PDR) else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_