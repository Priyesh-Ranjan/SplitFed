import copy
import torch
from utils import FedAvg, calculate_accuracy
#import numpy as np
#import torch
#from torch import nn
from model import Net
from codecarbon import EmissionsTracker
import logging
from collections import defaultdict, deque
from copy import deepcopy
from mudhog import MuDHoG

#====================================================================================================
#                                  Server Side Program
#====================================================================================================

# To print in color -------test/train of the client side

# Server-side function associated with Training 
class Server() :
    def __init__(self, criterion, device, lr, n, AR) :
        self.device = device
        self.net_model_server = []
        #self.net_glob_server = net_server
        self.lr = lr
        self.optimizer_server = []
        self.criterion = criterion
        self.clients = n
        self.AR = AR
        self.originalState = defaultdict(int)
        self.stateChange = defaultdict(int)
        self.K_avg = 3
        self.sum_hog = defaultdict(int)
        self.avg_delta = defaultdict(int)
        self.hog_avg = defaultdict(int)
        self.init_model()
    
    def get_num_clients(self) :
        return self.clients
    
    def init_model(self):
        for i in range(self.clients) :
            net_glob_server = Net().classifier
            
                #if torch.cuda.device_count() > 1:
                #    net_glob_server = nn.DataParallel(net_glob_server)
   
            self.net_model_server.append(net_glob_server) #self.net_model_server.append(net_glob_server.to(self.device))
            self.optimizer_server.append(torch.optim.Adam(net_glob_server.parameters(), lr = self.lr))
            
            states = deepcopy(net_glob_server.state_dict())
            for param, values in states.items():
                values *= 0
            self.stateChange[i] = states
            self.sum_hog[i] = deepcopy(states)
            self.avg_delta[i] = deepcopy(states)
            self.hog_avg[i] = deque(maxlen=self.K_avg)
            
    def compute_hogs(self):
        """Update sum_hog, avg_delta, and rolling history for all clients."""
        for i in range(self.clients):
            newState = self.net_model_server[i].state_dict()
            for p in newState:
                self.stateChange[i][p] = newState[p] - self.originalState[i][p]
                self.sum_hog[i][p] += self.stateChange[i][p]

                K_ = len(self.hog_avg[i])
                if K_ == 0:
                    self.avg_delta[i][p] = self.stateChange[i][p]
                elif K_ < self.K_avg:
                    self.avg_delta[i][p] = (self.avg_delta[i][p] * K_ + self.stateChange[i][p]) / (K_ + 1)
                else:
                    self.avg_delta[i][p] += (self.stateChange[i][p] - self.hog_avg[i][0][p]) / self.K_avg

            # append stateChange to rolling buffer
            self.hog_avg[i].append(deepcopy(self.stateChange[i]))        
        
    def train_server(self, fx_client, y, idx):
        net_server = copy.deepcopy(self.net_model_server[idx]) #net_server = copy.deepcopy(self.net_model_server[idx].to(self.device))
        net_server.to(self.device);net_server.train()
        
        # train and update
        self.optimizer_server[idx].zero_grad()
        
        fx_client = fx_client.to(self.device)
        y = y.to(self.device)
        
        #---------forward prop-------------
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()
        fx_server = net_server(fx_client)
        
        # calculate loss
        loss = self.criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        #--------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        self.optimizer_server[idx].step()
        
        # Update the server-side model for the current batch
        self.net_model_server[idx] = copy.deepcopy(net_server);net_server.cpu()
        server_train_emissions: float = tracker.stop()
        
        return dfx_client, loss, acc, server_train_emissions
    
    def setModelParameter(self, w_glob_server):
        tracker = EmissionsTracker()
        tracker.start()
        for idx in range(self.clients) :
            self.net_model_server[idx].load_state_dict(copy.deepcopy(w_glob_server))
            self.originalState[idx] = copy.deepcopy(w_glob_server)
            self.net_model_server[idx].zero_grad()
        agg: float = tracker.stop()
        return agg
            
    #def return_delta(self):
        #self.stateChange = defaultdict(int)
    #    for idx in range(self.clients) :
    #        newState = self.net_model_server[idx].state_dict()
    #        for p in self.originalState[idx]:
    #            self.stateChange[idx][p] = newState[p] - self.originalState[idx][p]
    #    return self.stateChange          
    
    def get_sum_hog(self, client_id):
        """Return concatenated HoG sum for a client."""
        return torch.cat([v.detach().flatten() for v in self.sum_hog[client_id].values()])

    def get_avg_grad(self, client_id):
        """Return concatenated average gradient for a client."""
        return torch.cat([v.detach().flatten() for v in self.avg_delta[client_id].values()])
    
    def aggregation(self):
        print("------------------------------------------------")
        print("------ Federation process at Server-Side ------- ")
        print("------------------------------------------------")
        
        w_locals_server = []
        
        for idx in range(self.clients) :
            w_server = self.net_model_server[idx].state_dict()    
            w_locals_server.append(copy.deepcopy(w_server))
            
        #if self.AR == "fedavg" :
        #    w_glob_server, t = FedAvg(w_locals_server)
        #if self.AR == "mudhog" :
        #    w_glob_server, t = MuDHoG(w_locals_server)
        
        #for i in range(self.clients) :
        #    self.net_model_server[i].load_state_dict(w_glob_server)
        #self.setModelParameter(w_glob_server)
        
        return w_locals_server

# Server-side functions associated with Testing
    def eval_server(self, fx_client, y, idx, len_batch, ell):
        net = copy.deepcopy(self.net_model_server[idx]).to(self.device)
        net.eval()
      
        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device) 
            #---------forward prop-------------
            fx_server = net(fx_client)
            
            # calculate loss
            loss = self.criterion(fx_server, y)
            # calculate accuracy
            acc = calculate_accuracy(fx_server, y)
            fx_server.cpu();y.cpu()
            return loss, acc
