import torch
from torch import nn

from model import ResNet18_client_side, ResNet18_server_side, Baseblock, Net
import copy
from server import Server
from client import Client
from client_attackers import Attacker_LF, label_flipping_setup
from utils import FedAvg, eval_train, eval_fed#, eval_glob

from codecarbon import track_emissions

#@track_emissions(offline=True, country_iso_code="USA", project_name="split", output_dir = "/carbon/", output_file = "split.csv")
def Split(args, trainData, testData):   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr

    net_glob_client = ResNet18_client_side()
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)    

    net_glob_client.to(device)
    print(net_glob_client) 
    optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr)

    if args.dataset.upper() == "CIFAR" :    
        net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 10) #10 is my numbr of classes
    if args.dataset.upper() == "PLANT" :
        net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 38)
        
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

    net_glob_server.to(device)
    print(net_glob_server)      
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)

    #client idx collector
    
    # Initialization of net_model_server and net_server (server-side model)
    
    server = Server(net_glob_server, nn.CrossEntropyLoss(), optimizer_server, device, lr, num_users)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)    
    
    idxs_users = range(num_users)
    clients = []

    for idx in idxs_users :
        clients.append(Client(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))  
    
    #------------ Training And Testing  -----------------
    #net_glob_client.train()
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    for client in clients :
        client.setModelParameter(w_glob_client)
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    loss_train = []; acc_train = []
    loss_test = []; acc_test = []
    #loss_glob = []; acc_glob = []
    for i in range(epochs):
        #w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        #loss_clients_glob = []; acc_clients_glob = []
        for client in clients:
            # Training ------------------
            train_loss, train_acc, w_client = client.train(server)
            #w_locals_client.append(copy.deepcopy(w_client))
            
            # Testing -------------------
            test_loss, test_acc = client.evaluate(server, ell= i, test = "local")
            #glob_loss, glob_acc = client.evaluate(server, ell= i, test = "global")
            loss_clients_train.append(train_loss); acc_clients_train.append(train_acc)    
            loss_clients_test.append(test_loss); acc_clients_test.append(test_acc)  
            #loss_clients_glob.append(glob_loss); acc_clients_glob.append(glob_acc)
            #w_glob_client = w_locals_client  
            client.setModelParameter(w_client)
            #net_glob_client.load_state_dict(w_locals_client)    
        
        l, a = eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = eval_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        #l, a = eval_glob(i, acc_clients_glob, loss_clients_glob)
        #loss_glob.append(l); acc_glob.append(a)
    return loss_train, acc_train, loss_test, acc_test#, loss_glob, acc_glob

#@track_emissions(offline=True, country_iso_code="USA", project_name="split_fed", output_dir = "/carbon/", output_file = "split_fed.csv")
def Split_Fed(args, trainData, testData):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr
    
    net_glob_client = ResNet18_client_side()
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)    

    net_glob_client.to(device)
    print(net_glob_client) 
    optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr) 


    if args.dataset.upper() == "CIFAR" :    
        net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 10) #10 is my numbr of classes
    if args.dataset.upper() == "PLANT" :
        net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 38)
        
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

    net_glob_server.to(device)
    print(net_glob_server)      
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)
    #client idx collector
    
    # Initialization of net_model_server and net_server (server-side model)
    
    server = Server(net_glob_server, nn.CrossEntropyLoss(), optimizer_server, device, lr, num_users)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)    
    
    idxs_users = range(num_users)
    clients = []
    
    if args.attack.upper() == "Label_Flipping" :
        flip = label_flipping_setup(args.attack, args.label_flipping)
    
    for idx in idxs_users :
        if idx < args.scale :
            clients.append(Attacker_LF(net_glob_client, args.PDR, flip, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
        else :    
            clients.append(Client(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))  
    
    
    #------------ Training And Testing  -----------------
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    loss_train = []; acc_train = []
    loss_test = []; acc_test = []
    #loss_glob = []; acc_glob = []
    
    for i in range(epochs):
        w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        #loss_clients_glob = []; acc_clients_glob = []
        for client in clients:
            # Training ------------------
            client.setModelParameter(w_glob_client) 
            train_loss, train_acc, w_client = client.train(server)
            w_locals_client.append(copy.deepcopy(w_client))
            
            # Testing -------------------
            test_loss, test_acc = client.evaluate(server, ell= i, test = "local")
            #glob_loss, glob_acc = client.evaluate(server, ell= i, test = "global")
        
            loss_clients_train.append(train_loss); acc_clients_train.append(train_acc)    
            loss_clients_test.append(test_loss); acc_clients_test.append(test_acc)    
            #loss_clients_glob.append(glob_loss); acc_clients_glob.append(glob_acc)
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
        
        print("-----------------------------------------------------------")
        print("------ FedServer: Federation process at Client-Side ------- ")
        print("-----------------------------------------------------------")
        w_glob_client = FedAvg(w_locals_client)   
        
        # Update client-side global model 
        
        l, a = eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = eval_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        #l, a = eval_glob(i, acc_clients_glob, loss_clients_glob)
        #loss_glob.append(l); acc_glob.append(a)
    return loss_train, acc_train, loss_test, acc_test#, loss_glob, acc_glob

#@track_emissions(offline=True, country_iso_code="USA", project_name="fed", output_file = "fed.csv")
def Fed(args, trainData, testData) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr
    
    if args.dataset.upper() == "CIFAR" :    
        net_glob_client = Net(10) #10 is my numbr of classes
    if args.dataset.upper() == "PLANT" :
        net_glob_client = Net(38)
    
    #net_glob_client = Net(10)
    #net_glob_client = ResNet18_client_side()
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)    

    net_glob_client.to(device)
    print(net_glob_client) 
    optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr) 

    #net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 7) #7 is my numbr of classes
    #if torch.cuda.device_count() > 1:
    #    print("We use",torch.cuda.device_count(), "GPUs")
    #    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

    #net_glob_server.to(device)
    #print(net_glob_server)      
    #optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)
    #client idx collector
    
    # Initialization of net_model_server and net_server (server-side model)
    
    #server = Server(net_glob_server, nn.CrossEntropyLoss(), optimizer_server, device, lr, num_users)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)    
    
    idxs_users = range(num_users)
    clients = []
    
    if args.attack.upper() == "Label_Flipping" :
        flip = label_flipping_setup(args.attack, args.label_flipping)
    
    for idx in idxs_users :
        if idx < args.scale :
            clients.append(Attacker_LF(net_glob_client, args.PDR, flip, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
        else :    
            clients.append(Client(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))  
    
    
    #------------ Training And Testing  -----------------
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    loss_train = []; acc_train = []
    loss_test = []; acc_test = []
    #loss_glob = []; acc_glob = []
    for i in range(epochs):
        w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        #loss_clients_glob = []; acc_clients_glob = []
        for client in clients:
            client.setModelParameter(w_glob_client) 
            # Training ------------------
            train_loss, train_acc, w_client = client.train_federated()
            w_locals_client.append(copy.deepcopy(w_client))
            
            # Testing -------------------
            test_loss, test_acc = client.evaluate_federated(test = "local")
            #glob_loss, glob_acc = client.evaluate_federated(test = "global")
            
            loss_clients_train.append(train_loss); acc_clients_train.append(train_acc)    
            loss_clients_test.append(test_loss); acc_clients_test.append(test_acc)    
            #loss_clients_glob.append(glob_loss); acc_clients_glob.append(glob_acc)
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
        
        print("-----------------------------------------------------------")
        print("------ FedServer: Federation process at Client-Side ------- ")
        print("-----------------------------------------------------------")
        w_glob_client = FedAvg(w_locals_client)   
        
        # Update client-side global model 
        
        l, a = eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = eval_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        #l, a = eval_glob(i, acc_clients_glob, loss_clients_glob)
        #loss_glob.append(l); acc_glob.append(a)
    return loss_train, acc_train, loss_test, acc_test#, loss_glob, acc_glob