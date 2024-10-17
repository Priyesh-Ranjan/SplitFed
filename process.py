#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from pandas import DataFrame

import random
import numpy as np


import matplotlib
matplotlib.use('Agg')
import copy

from server import Server
from model import ResNet18_client_side, ResNet18_server_side, Baseblock
from dataset import prepare_dataset
from client import Client
from client_attackers import Attacker_LF, label_flipping_setup
from utils import FedAvg

def main(args) :
    SEED = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))    

    #===================================================================
    program = args.experiment_name
    print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr

    dataset_train, dataset_test, dict_users, dict_users_test = prepare_dataset(num_users, args.dataset, args.loader_type)    

    net_glob_client = ResNet18_client_side()
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)    

    net_glob_client.to(device)
    print(net_glob_client) 

    net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 7) #7 is my numbr of classes
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

    net_glob_server.to(device)
    print(net_glob_server)      

    #client idx collector
    
    # Initialization of net_model_server and net_server (server-side model)
    
    server = Server(net_glob_server, nn.CrossEntropyLoss(), device, lr, num_users)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)    
    
    idxs_users = range(num_users)
    clients = []
    
    flip = label_flipping_setup(args.attack, args.label_flipping)
    
    for idx in idxs_users :
        if idx < args.scale :
            clients.append(Attacker_LF(args.PDR, flip, idx, lr, device, local_epochs, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx]))
        else :    
            clients.append(Client(idx, lr, device, local_epochs, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx]))
    
    
    #------------ Training And Testing  -----------------
    net_glob_client.train()
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    loss_train = []; acc_train = []
    loss_test = []; acc_test = []
    for i in range(epochs):
        w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        for client in clients:
            # Training ------------------
            train_loss, train_acc, w_client = client.train(server, net = copy.deepcopy(net_glob_client).to(device))
            w_locals_client.append(copy.deepcopy(w_client))
            
            # Testing -------------------
            test_loss, test_acc = client.evaluate(server, net = copy.deepcopy(net_glob_client).to(device), ell= i)
            
            loss_clients_train.append(train_loss); acc_clients_train.append(train_acc)    
            loss_clients_test.append(test_loss); acc_clients_test.append(test_acc)    
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
        
        print("-----------------------------------------------------------")
        print("------ FedServer: Federation process at Client-Side ------- ")
        print("-----------------------------------------------------------")
        w_glob_client = FedAvg(w_locals_client)   
        
        # Update client-side global model 
        net_glob_client.load_state_dict(w_glob_client)   
        
        l, a = server.eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = server.evaluate_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        
    #===================================================================================     

    print("Training and Evaluation completed!")    

    #===============================================================================
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(acc_train)+1)]
    df = DataFrame({'round': round_process,'loss_train':loss_train,'acc_train':acc_train, 
                    'loss_test':loss_test, 'acc_test':acc_test})     
    file_name = program+".xlsx"    
    df.to_excel(file_name, sheet_name= "v1_test", index = False)     

    #=============================================================================
    #                         Program Completed
    #=============================================================================