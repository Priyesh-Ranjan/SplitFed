import copy
import torch
from utils import FedAvg, calculate_accuracy
import numpy as np

#====================================================================================================
#                                  Server Side Program
#====================================================================================================

# To print in color -------test/train of the client side

# Server-side function associated with Training 
class Server() :
    def __init__(self, net_model_server, net_glob_server, criterion, device, lr, n) :
        self.device = device
        self.net_model_server = [net_glob_server for i in range(n)]
        self.net_glob_server = net_glob_server
        self.lr = lr
        self.criterion = criterion
        self.clients = []
        
    def train_server(self, fx_client, y, idx):
        net_server = copy.deepcopy(self.net_model_server[idx]).to(self.device)
        net_server.train()
        optimizer_server = torch.optim.Adam(net_server.parameters(), lr = self.lr)
        
        # train and update
        optimizer_server.zero_grad()
        
        fx_client = fx_client.to(self.device)
        y = y.to(self.device)
        
        #---------forward prop-------------
        fx_server = net_server(fx_client)
        
        # calculate loss
        loss = self.criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        #--------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer_server.step()
        
        # Update the server-side model for the current batch
        self.net_model_server[idx] = copy.deepcopy(net_server)
        
        return dfx_client, loss, acc
    
    def aggregation(self):
        print("------------------------------------------------")
        print("------ Federation process at Server-Side ------- ")
        print("------------------------------------------------")
        
        w_locals_server = []
        
        for idx in self.clients :
            w_server = self.net_model_server[idx].state_dict()    
            w_locals_server.append(copy.deepcopy(w_server))
            
        w_glob_server = FedAvg(w_locals_server)
        
        self.net_glob_server.load_state_dict(w_glob_server)    
        self.net_model_server = [self.net_glob_server for i in range(self.clients)]
    
    def eval_train(self, ell, acc_train_collect_user, loss_train_collect_user) :
        # server-side global model update and distribute that model to all clients ------------------------------
        loss_train, acc_train = np.average(loss_train_collect_user), np.average(acc_train_collect_user)
        
        print("====================== SERVER V1==========================")
        print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_train, loss_train))
        
        return loss_train, acc_train

# Server-side functions associated with Testing
    def evaluate_server(self, fx_client, y, idx, len_batch, ell):
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
            
            return loss, acc
    
    def eval_fed(self, ell, acc_test_collect_user, loss_test_collect_user) :        
        
        loss_test, acc_test = np.average(loss_test_collect_user), np.average(acc_test_collect_user)
                      
        
        print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_test, loss_test))
        print("==========================================================") 
        
        return loss_test, acc_test