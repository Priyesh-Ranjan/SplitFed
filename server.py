import copy
import torch
import gc
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
from defense import LatentTrustAnalyzer
from defense_lab import ServerLatentMMDAnalyzer
from defense_gac import GradientActivationCorrelationAnalyzer
from utils import FedAvg_Trust




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
        if self.AR.lower() == "new" :
            self.trust_analyzer = LatentTrustAnalyzer(
            device=self.device,
            trust_threshold=0.5,
            log_dir=f"logs/trust_{self.AR}")
        if self.AR.lower() == "plr" :
           self.trust_analyzer = ServerLatentMMDAnalyzer(
               num_classes=16,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               kernel_gamma=None,            # use median heuristic
               max_samples_per_class=64,
               subsample_per_mmd=32)
        if self.AR.lower() == 'gac' :
            self.trust_analyzer = GradientActivationCorrelationAnalyzer(tau=0.5)

    
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
        
        if self.AR.lower() == "plr":
            self.trust_analyzer.update_client_latent(client_id=str(idx), latent=fx_client.detach(), labels=y.detach())
            gc.collect(); torch.cuda.empty_cache()
        
        #--------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        self.optimizer_server[idx].step()
        
        if self.AR.lower() == "gac":
            self.trust_analyzer.add_batch(idx, fx_client.detach(), dfx_client.detach(), y)
        
        # Update the server-side model for the current batch
        self.net_model_server[idx] = copy.deepcopy(net_server);net_server.cpu()
        server_train_emissions: float = tracker.stop()
        
        if self.AR.lower() == "new":
            self.trust_analyzer.update(fx_client.detach(), idx)
        
        return dfx_client, loss, acc, server_train_emissions
    
    def aggregate(self, models, round_idx, identity = None):
        """Aggregate client models using the selected algorithm and trust weights."""
        # compute final trust per client
        if identity.lower() == "client":
            trust_scores = {
            cid: self.trust_analyzer.finalize_client(cid, round_idx, "client")
            for cid in range(self.clients)}
        elif identity.lower() == "server":
            trust_scores = {
            cid: self.trust_analyzer.finalize_client(cid, round_idx, "server")
            for cid in range(self.clients)}

        # Normalize trust scores (prevent division by zero)
        weights = torch.tensor(list(trust_scores.values()), device=self.device)
        weights = torch.softmax(weights, dim=0)

        print(f"[Round {round_idx}] Trust Weights:", weights.cpu().numpy())

        # Call the appropriate aggregator
        
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()
        
        global_model = FedAvg_Trust(models, weights)
        
        agg : float = tracker.stop()

        return global_model, agg
    
    def plr_aggregate(self, models, round_idx, identity = None):
        """Aggregate client models using the selected algorithm and trust weights."""
        # compute final trust per client
        if identity.lower() == "client":
            trust_scores, diag = self.trust_analyzer.compute_trust_scores(round_id=round_idx)
        elif identity.lower() == "server":
            trust_scores, diag = self.trust_analyzer.compute_trust_scores(round_id=round_idx)

        # Normalize trust scores (prevent division by zero)
        #tables = class_mmd_table(diag["normalized_class_pair_mmds"])

        weights = [trust_scores.get(str(i), 0.0) for i in range(self.clients)]
        #global_w, abi = FedAvg(client_state_dicts, trust_scores=weights)
        #weights = torch.tensor(list(trust_scores.values()), device=self.device)
        #weights = torch.softmax(weights, dim=0)

        print(f"[Round {round_idx}] Trust Weights:", weights)

        # Call the appropriate aggregator
        
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()
        
        global_model = FedAvg_Trust(models, weights)
        
        agg : float = tracker.stop()
        
        if identity.lower == "server" : self.trust_analyzer.reset()
        
        self.cleanup_after_round()
        
        return global_model, agg
    
    def gac_aggregate(self, models, round_idx, identity = None):
        """Aggregate client models using the selected algorithm and trust weights."""
        # compute final trust per client
        if identity.lower() == "client":
            trust_scores, diag = self.trust_analyzer.finalize_scores()
        elif identity.lower() == "server":
            trust_scores, diag = self.trust_analyzer.finalize_scores()

        # Normalize trust scores (prevent division by zero)
        #tables = class_mmd_table(diag["normalized_class_pair_mmds"])

        weights = [trust_scores.get(i, 0.0) for i in range(self.clients)]
        #global_w, abi = FedAvg(client_state_dicts, trust_scores=weights)
        #weights = torch.tensor(list(trust_scores.values()), device=self.device)
        #weights = torch.softmax(weights, dim=0)

        print(f"[Round {round_idx}] Trust Weights:", weights)

        # Call the appropriate aggregator
        
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()
        
        global_model = FedAvg_Trust(models, weights)
        
        agg : float = tracker.stop()
        
        if identity.lower == "server" : self.trust_analyzer.reset()
        
        self.cleanup_after_round()
        
        return global_model, agg
    
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
        
    def cleanup_after_round(self):
        """Free latent features, clear cache, and release models."""
    # Clear stored latents from analyzer
        #if self.trust_analyzer is not None:
        #    self.trust_analyzer.client_latents.clear()
    
    # Clear CUDA memory
        torch.cuda.empty_cache()
        gc.collect()

    # Move server models to CPU explicitly and delete temp references
        for i in range(len(self.net_model_server)):
            if self.net_model_server[i] is not None:
                self.net_model_server[i].cpu()
                torch.cuda.empty_cache()
        gc.collect()

