import copy
import torch
from utils import FedAvg, calculate_accuracy
import numpy as np
import torch
from torch import nn
from model import VGG16_Server_Side, Net
from codecarbon import EmissionsTracker

#====================================================================================================
#                                  Server Side Program
#====================================================================================================

# To print in color -------test/train of the client side

# Server-side function associated with Training 
class Server() :
    def __init__(self, criterion, device, lr, n) :
        self.device = device
        self.net_model_server = []
        #self.net_glob_server = net_server
        self.lr = lr
        self.optimizer_server = []
        self.criterion = criterion
        self.clients = n
        self.init_model()
        
    def init_model(self):
        for i in range(self.clients) :
            net_glob_server = Net().classifier
            
                #if torch.cuda.device_count() > 1:
                #    net_glob_server = nn.DataParallel(net_glob_server)
   
            self.net_model_server.append(net_glob_server) #self.net_model_server.append(net_glob_server.to(self.device))
            self.optimizer_server.append(torch.optim.Adam(net_glob_server.parameters(), lr = self.lr))
        
    def train_server(self, fx_client, y, idx):
        net_server = copy.deepcopy(self.net_model_server[idx]) #net_server = copy.deepcopy(self.net_model_server[idx].to(self.device))
        net_server.to(self.device);net_server.train()
        
        # train and update
        self.optimizer_server[idx].zero_grad()
        
        fx_client = fx_client.to(self.device)
        y = y.to(self.device)
        
        #---------forward prop-------------
        tracker = EmissionsTracker()
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
        for idx in range(self.clients) :
            self.net_model_server[idx].load_state_dict(copy.deepcopy(w_glob_server))
            self.originalState = copy.deepcopy(w_glob_server)
            self.net_model_server[idx].zero_grad()
    
    def aggregation(self):
        print("------------------------------------------------")
        print("------ Federation process at Server-Side ------- ")
        print("------------------------------------------------")
        
        w_locals_server = []
        
        for idx in range(self.clients) :
            w_server = self.net_model_server[idx].state_dict()    
            w_locals_server.append(copy.deepcopy(w_server))
            
        w_glob_server = FedAvg(w_locals_server)
        
        #for i in range(self.clients) :
        #    self.net_model_server[i].load_state_dict(w_glob_server)
        self.setModelParameter(w_glob_server)

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
