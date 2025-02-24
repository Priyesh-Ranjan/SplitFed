#from torch.utils.data import DataLoader
import torch
#from utils import DatasetSplit
#from server import evaluate_server, train_server
from random import random
from client import Client

def label_flipping_setup(attack, label_flipping) :
    sources = [int(i.strip()) for i in attack.split("->")[0].split(",")]
    targets = [int(i.strip()) for i in attack.split("->")[1].split(",")]
    
    flip = {}
    
    for i, val in enumerate(sources) :
        flip[val] = targets[i]
        if label_flipping == "bi" :
            flip[targets[i]] = val
    
    return flip  

class Attacker_LF(Client):
    def __init__(self, PDR, flip, idx, lr, device, dataset_train, dataset_test, idxs, idxs_test, local_ep = 1):
        super(Attacker_LF, self).__init__(idx, lr, device, dataset_train, dataset_test, idxs, idxs_test, local_ep = 1)
        self.PDR = PDR
        self.flip = flip

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: self.flip[int(x)] if (x in list(self.flip.keys()) and random() <= self.PDR) else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_
