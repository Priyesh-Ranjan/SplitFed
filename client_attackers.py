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

def poisoning_setup(attack) :
    mu = float(attack.split(" ")[1].split(",")[0])
    std = float(attack.split(" ")[1].split(",")[1])
    
    return mu, std

class Attacker_LF(Client):
    def __init__(self, net_glob_client, PDR, flip, idx, lr, device, optimizer, trainData, testData, local_ep):
        super(Attacker_LF, self).__init__(net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep)
        self.PDR = PDR
        self.flip = flip

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: self.flip[int(x)] if (x in list(self.flip.keys()) and random() <= self.PDR) else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_
    
    def model_transform(self, model) :
        pass


class Attacker_Random(Client):
    def _init(self, net_glob_client, PDR, idx, lr, device, optimizer, trainData, testData, local_ep):
        super(Attacker_Random, self).__init__(net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep)
        self.PDR = PDR
        self.labels = range(len(self.ldr_test.dataset.classes))
    
    def data_transform(self, data, target) :
        target_ = torch.tensor(list(map(lambda x: random.choice(self.labels) if random() <= self.PDR else x, target)))
        return data, target_ 
    
    def model_transform(self, model) :
        pass
    
class Attacker_SignFlipping(Client):
    def _init(self, net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep):
        super(Attacker_SignFlipping, self).__init__(net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep)
    
    def data_transform(self, data, target) :
        return data, target
    
    def model_transform(self) :
        state_dict = self.model.state_dict()

        for key in state_dict:
            state_dict[key] = -1 * state_dict[key]

        self.model.load_state_dict(state_dict)
        
class Attacker_ModelPoisoning(Client):
    def _init(self, net_glob_client, std, mu, idx, lr, device, optimizer, trainData, testData, local_ep):
        super(Attacker_ModelPoisoning, self).__init__(net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep)
        self.std = std
        self.mu = mu
    def data_transform(self, data, target) :
        return data, target
    
    def model_transform(self) :
        state_dict = self.model.state_dict()

        for key in state_dict:
            noise = torch.randn_like(state_dict[key]) * self.std + self.mu
            state_dict[key] = state_dict[key] + noise

        self.model.load_state_dict(state_dict) 
        
class Attacker_DataPoisoning(Client):
    def _init(self, net_glob_client, PDR, mu, std, idx, lr, device, optimizer, trainData, testData, local_ep):
        super(Attacker_DataPoisoning, self).__init__(net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep)
        self.std = std
        self.mu = mu
        self.PDR = PDR
    def data_transform(self, data, target) :
        data_ = torch.tensor(list(map(lambda x: x + torch.randn_like(x) * self.std + self.mu if random() <= self.PDR else x, data)))
        return data_, target
    
    def model_transform(self) :
        pass