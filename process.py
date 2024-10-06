#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

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
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr

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

    #===================================================================================
    # For Server Side Loss and Accuracy 
    loss_train_collect = []; acc_train_collect = []; loss_test_collect = []; acc_test_collect = []
    batch_acc_train = []; batch_loss_train = []; batch_acc_test = []; batch_loss_test = []

    criterion = nn.CrossEntropyLoss()
    count1 = 0; count2 = 0
    #====================================================================================================
    #                                  Server Side Program
    #====================================================================================================

    # to print train - test together in each round-- these are made global
    acc_avg_all_user_train = 0; loss_avg_all_user_train = 0
    loss_train_collect_user = []; acc_train_collect_user = []
    loss_test_collect_user = []; acc_test_collect_user = []

    w_glob_server = net_glob_server.state_dict()
    w_locals_server = []

    #client idx collector
    idx_collect = []
    l_epoch_check = False
    fed_check = False
    # Initialization of net_model_server and net_server (server-side model)
    net_model_server = [net_glob_server for i in range(num_users)]
    net_server = copy.deepcopy(net_model_server[0]).to(device)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)    
        
    dict_users, dict_users_test = load_dataset(num_users, args.dataset, args.loader_type)    

    #------------ Training And Testing  -----------------
    net_glob_client.train()
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    for iter in range(epochs):
        m = max(int(num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace = False)
        w_locals_client = []
          
        for idx in idxs_users:
            local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # Training ------------------
            w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
            w_locals_client.append(copy.deepcopy(w_client))
            
            # Testing -------------------
            local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
            
                
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
        print("-----------------------------------------------------------")
        print("------ FedServer: Federation process at Client-Side ------- ")
        print("-----------------------------------------------------------")
        w_glob_client = FedAvg(w_locals_client)   
        
        # Update client-side global model 
        net_glob_client.load_state_dict(w_glob_client)    
        
    #===================================================================================     

    print("Training and Evaluation completed!")    

    #===============================================================================
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(acc_train_collect)+1)]
    df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
    file_name = program+".xlsx"    
    df.to_excel(file_name, sheet_name= "v1_test", index = False)     

    #=============================================================================
    #                         Program Completed
    #=============================================================================