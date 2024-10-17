from torch.utils.data import DataLoader
import torch
from dataset import DatasetSplit
import numpy as np

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, idx, lr, device, local_ep = 1, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = local_ep
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 256, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 256, shuffle = True)
        
    def data_transform(self, images, labels) :
        return images, labels
    
    def train(self, server, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for ep in range(self.local_ep):
            #len_batch = len(self.ldr_train)
            loss = []; acc = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.data_transform(images, labels)
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                dfx, batch_loss, batch_acc = server.train_server(client_fx, labels, self.idx)
                loss.extend(batch_loss.item())
                acc.extend(batch_acc.item())
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
            
            prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx, self.local_ep, 
                                                                                          np.average(acc), np.average(loss)))
        
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    
    def evaluate(self, server, net, ell):
        net.eval()
        loss = []; acc= []   
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                batch_loss, batch_acc = server.evaluate_server(fx, labels, self.idx, len_batch, ell)
                loss.extend(batch_loss.item())
                acc.extend(batch_acc.item())
                        
        prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx, np.average(acc), np.average(loss)))
        return np.average(loss), np.average(acc)   