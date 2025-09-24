#from torch.utils.data import DataLoader
import torch
#from dataset import DatasetSplit
import numpy as np
from copy import deepcopy
from utils import calculate_accuracy
from torch import nn
from codecarbon import EmissionsTracker
import copy
import logging
from collections import deque

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_glob_client, idx, lr, device, optimizer, trainData, testData, local_ep = 1):
        self.model = net_glob_client
        self.idx = idx
        self.device = device
        self.lr = lr
        self.optimizer_client = optimizer
        self.local_ep = local_ep
        self.criterion = nn.CrossEntropyLoss()
        #self.selected_clients = []
        self.ldr_train = trainData
        self.ldr_test = testData
        self.init_parameters()
        self.K_avg = 3
        self.hog_avg = deque(maxlen=self.K_avg)
        #self.ldr_glob = DataLoader(DatasetSplit(dataset_test, range(len(dataset_test))), batch_size = 256, shuffle = True)
        
    def init_parameters(self) :
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states
        self.avg_delta = deepcopy(states)
        self.sum_hog = deepcopy(states)
        
    def data_transform(self, images, labels) :
        return images, labels
    
    def model_transform(self) :
        pass
    
    def setModelParameter(self, states):
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()
        agg_emissions: float = tracker.stop()
        return agg_emissions
    
    def get_sum_hog(self):
        #return utils.net2vec(self.sum_hog)
        return torch.cat([v.detach().flatten() for v in self.sum_hog.values()])
    
    def get_avg_grad(self):
        return torch.cat([v.detach().flatten() for v in self.avg_delta.values()])
    
    def compute_hogs(self):
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        newState = self.model.state_dict()
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            self.sum_hog[p] += self.stateChange[p]
            K_ = len(self.hog_avg)
            if K_ == 0:
                self.avg_delta[p] = self.stateChange[p]
            elif K_ < self.K_avg:
                self.avg_delta[p] = (self.avg_delta[p]*K_ + self.stateChange[p])/(K_+1)
            else:
                self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p])/self.K_avg
        self.hog_avg.append(deepcopy(self.stateChange))
        em: float = tracker.stop()
        return em
        
    
    def train(self, server):
        client_train_emissions = 0; server_train_emissions = 0
        up = 0; down = 0
        self.model.to(self.device)
        self.model.train()
        for ep in range(self.local_ep):
            #len_batch = len(self.ldr_train)
            loss = []; acc = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.data_transform(images, labels)
                images, labels = images.to(self.device), labels.to(self.device)
                
                tracker = EmissionsTracker(log_level=logging.CRITICAL)
                tracker.start()
                self.optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = self.model(images)
                fx = fx.view(fx.size(0), -1)
                client_fx = fx.clone().detach().requires_grad_(True)
                te_1: float = tracker.stop()
                client_train_emissions+=te_1
                up += fx.element_size() * fx.nelement() + labels.element_size() * labels.nelement()
                # Sending activations to server and receiving gradients from server
                
                dfx, batch_loss, batch_acc, emissions = server.train_server(client_fx, labels, self.idx)
                down += dfx.element_size() * dfx.nelement()
                server_train_emissions += emissions
                tracker = EmissionsTracker(log_level=logging.CRITICAL)
                tracker.start()
                loss.append(batch_loss.item())
                acc.append(batch_acc.item())
                
                #--------backward prop -------------
                fx.backward(dfx)
                self.optimizer_client.step()
                te_2: float = tracker.stop()
            
                client_train_emissions += te_2
            prRed('Client{} Train => Local Epoch: {} / {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx, ep, self.local_ep, 
                                                                                          np.average(acc), np.average(loss)))
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
        
        self.model_transform()    
        self.model.cpu()
        #client_train_emissions = train_emissions_1 + train_emissions_2
        return np.average(loss), np.average(acc), self.model.state_dict(), client_train_emissions, server_train_emissions, up, down
    
    #def update(self):
    #    tracker = EmissionsTracker(log_level=logging.CRITICAL)
    #    newState = self.model.state_dict()
    #    for param in self.originalState:
    #        self.stateChange[param] = newState[param] - self.originalState[param]
    #    em: float = tracker.stop()
    #    return em
    
    def train_federated(self):
        emissions = 0
        self.model.to(self.device)
        self.model.train()
       	for ep in range(self.local_ep):
            #len_batch = len(self.ldr_train)
            loss = []; acc = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.data_transform(images, labels)
                images, labels = images.to(self.device), labels.to(self.device)
                tracker = EmissionsTracker(log_level=logging.CRITICAL)
                tracker.start()
                self.optimizer_client.zero_grad()
                #---------forward prop-------------
                out = self.model(images)
                #client_fx = fx.clone().detach().requires_grad_(True)
                
                batch_loss = self.criterion(out, labels)
                batch_acc = calculate_accuracy(out, labels)
                
                #--------backward prop--------------
                
                loss.append(batch_loss.item())
                acc.append(batch_acc.item())
                
                batch_loss.backward()
                self.optimizer_client.step()
                
                em: float = tracker.stop()
                emissions += em
            prRed('Client{} Train => Local Epoch: {} / {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx, ep, self.local_ep, 
                                                                                          np.average(acc), np.average(loss)))
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
        
        self.model_transform()    
        self.model.cpu()
        return np.average(loss), np.average(acc), self.model.state_dict(), emissions, 0.0, 0, 0
    
    def evaluate(self, server, ell, test):
        self.model.to(self.device);self.model.eval()
        loss = []; acc= []  
        if test == "local" : ldr = self.ldr_test
        #elif test == 'global' : ldr = self.ldr_glob
        with torch.no_grad():
            len_batch = len(ldr)
            for batch_idx, (images, labels) in enumerate(ldr):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = self.model(images)
                fx = fx.view(fx.size(0), -1)
                # Sending activations to server 
                batch_loss, batch_acc = server.eval_server(fx, labels, self.idx, len_batch, ell)
                loss.append(batch_loss.item())
                acc.append(batch_acc.item())
                        
        prGreen(' Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx, np.average(acc), np.average(loss)))
        self.model.cpu()
        return np.average(loss), np.average(acc) 
    
    def evaluate_federated(self, test) :
        temp_model = copy.deepcopy(self.model)
        temp_model.to(self.device);temp_model.eval()
        if test == "local" : ldr = self.ldr_test
        #elif test == 'global' : ldr = self.ldr_glob
        loss = []; acc= []   
        with torch.no_grad():
            #len_batch = len(ldr)
            for batch_idx, (images, labels) in enumerate(ldr):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                out = temp_model(images)
                
                batch_loss = self.criterion(out, labels)
                # calculate accuracy
                batch_acc = calculate_accuracy(out, labels)
                
                loss.append(batch_loss.item())
                acc.append(batch_acc.item())
                        
        prGreen(' Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx, np.average(acc), np.average(loss)))
        temp_model.cpu()
        return np.average(loss), np.average(acc) 