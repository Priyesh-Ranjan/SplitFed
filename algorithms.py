import torch
from torch import nn

from model import Net, VGG16_Client_Side
import copy
from server import Server
from client import Client
from client_attackers import Attacker_LF, label_flipping_setup, Attacker_SignFlipping, Attacker_Random, Attacker_BD
from client_attackers import Attacker_DataPoisoning, Attacker_ModelPoisoning, poisoning_setup, backdoor_setup
from utils import FedAvg, eval_train, eval_fed#, eval_glob
import io
from mudhog import MuDHoG
from flame import FLAME

from codecarbon import EmissionsTracker

def Split(args, trainData, testData):   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr
    AR = args.AR
    
    #if torch.cuda.device_count() > 1:
    #    print("We use",torch.cuda.device_count(), "GPUs")
    
    idxs_users = range(num_users)
    clients = []

    for idx in idxs_users :
        net_glob_client = Net().features
        
        #if torch.cuda.device_count() > 1:
        #    net_glob_client = nn.DataParallel(net_glob_client)    

        #net_glob_client.to(device) 
        optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr)
        clients.append(Client(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))  
        
    server = Server(nn.CrossEntropyLoss(), device, lr, num_users, AR)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)    
    
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
    client_carbon = []; server_carbon = []
    uplink = []; downlink = []
    #loss_glob = []; acc_glob = []
    for i in range(epochs):
        #w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        client_temp = 0; server_temp = 0
        up = 0; down = 0
        #loss_clients_glob = []; acc_clients_glob = []
        for client in clients:
            # Training ------------------
            train_loss, train_acc, w_client, client_emissions, server_emissions, u, d = client.train(server)
            #w_locals_client.append(copy.deepcopy(w_client))
            
            client_temp += client_emissions; server_temp += server_emissions
            up+= u; down+= d
            # Testing -------------------
            test_loss, test_acc = client.evaluate(server, ell= i, test = "local")
            #glob_loss, glob_acc = client.evaluate(server, ell= i, test = "global")
            loss_clients_train.append(train_loss); acc_clients_train.append(train_acc)    
            loss_clients_test.append(test_loss); acc_clients_test.append(test_acc)  
            #loss_clients_glob.append(glob_loss); acc_clients_glob.append(glob_acc)
            #w_glob_client = w_locals_client  
            client_temp += client.setModelParameter(w_client)
            #net_glob_client.load_state_dict(w_locals_client)    
        
        l, a = eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = eval_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        client_carbon.append(client_temp); server_carbon.append(server_temp)
        uplink.append(u); downlink.append(d)
        #l, a = eval_glob(i, acc_clients_glob, loss_clients_glob)
        #loss_glob.append(l); acc_glob.append(a)
    return loss_train, acc_train, loss_test, acc_test, client_carbon, server_carbon, 0.0, 0.0, uplink, downlink

def Split_Fed(args, trainData, testData):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using "+str(device))

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr
    #AR = args.AR
    
    #if torch.cuda.device_count() > 1:
    #    print("We use",torch.cuda.device_count(), "GPUs")
    
    idxs_users = range(num_users)
    clients = []
    
    if args.attack.upper().count("LABEL_FLIPPING") :
        att = "LF"
        flip = label_flipping_setup(args.attack, args.label_flipping)
    if args.attack.upper().count("BACKDOOR") :
        att = "BD"
        pattern, target = backdoor_setup(args.attack)    
    if args.attack.upper().count("SIGN_FLIPPING") :
        att = "SF"
    if args.attack.upper().count("RANDOM_FLIPPING") :
        att = "RLF"
    if args.attack.upper().count("DATA_POISONING") :
        att = "DP"
        mu, std = poisoning_setup(args.attack)
    if args.attack.upper().count("MODEL_POISONING") :
        att = "MP"
        mu, std = poisoning_setup(args.attack)
    
    for idx in idxs_users :
        net_glob_client = Net().features
        
        #if torch.cuda.device_count() > 1:
        #    net_glob_client = nn.DataParallel(net_glob_client)    

        #net_glob_client.to(device)
        optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr) 
 
        if idx < args.scale :
            if att == "SF": clients.append(Attacker_SignFlipping(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_epochs))
            if att == "RLF": clients.append(Attacker_Random(net_glob_client, args.PDR, idx, lr, device, optimizer_client, trainData[idx], testData, local_epochs))
            if att == "LF": clients.append(Attacker_LF(net_glob_client, args.PDR, flip, idx, lr, device, optimizer_client, trainData[idx], testData, local_epochs))
            if att == "BD": clients.append(Attacker_BD(net_glob_client, args.PDR, pattern, target, idx, lr, device, optimizer_client, trainData[idx], testData, local_epochs))
            if att == "DP": clients.append(Attacker_DataPoisoning(net_glob_client, args.PDR, mu, std, idx, lr, device, optimizer_client, trainData[idx], testData, local_epochs))
            if att == "MP": clients.append(Attacker_ModelPoisoning(net_glob_client, mu, std, idx, lr, device, optimizer_client, trainData[idx], testData, local_epochs))
        else :    
            clients.append(Client(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))  
    
    server = Server(nn.CrossEntropyLoss(), device, lr, num_users, args.AR)
    
    if args.AR == "mudhog" :
        if args.side == "client" or args.side == "server":
            mudhog_object = MuDHoG()
        elif args.side == "both":
            mudhog_server = MuDHoG()
            mudhog_client = MuDHoG()
    if args.AR == "flame" :
        if args.side == "client" or args.side == "server":
            flame_object = FLAME()
        elif args.side == "both" :
            flame_server = FLAME()
            flame_client = FLAME()
    
    #------------ Training And Testing  -----------------
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    w_glob_server = Net().classifier.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    loss_train = []; acc_train = []
    loss_test = []; acc_test = []
    client_carbon = []; server_carbon = []
    client_agg_carbon = []; server_agg_carbon = []
    uplink = []; downlink = []
    #loss_glob = []; acc_glob = []
    
    for i in range(epochs):
        w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        client_temp = 0; server_temp = 0
        up = 0; down = 0
        server_temp += server.setModelParameter(w_glob_server)
        #loss_clients_glob = []; acc_clients_glob = []
        for client in clients:
            # Training ------------------
            client_temp += client.setModelParameter(w_glob_client) 
            buffer = io.BytesIO()
            torch.save(w_glob_client, buffer)
            down += buffer.tell()
            #down += w_glob_client.element_size() * w_glob_client.nelement()            
            train_loss, train_acc, w_client, client_emissions, server_emissions, u, d = client.train(server)
            up += u; down += d
            w_locals_client.append(copy.deepcopy(w_client))
            
            client_temp += client_emissions; server_temp += server_emissions
            
            buffer = io.BytesIO()
            torch.save(w_client, buffer)
            up += buffer.tell()
            
            #up += w_locals_client.element_size() * w_locals_client.nelement()

            # Testing -------------------
            test_loss, test_acc = 0, 0
            test_loss, test_acc = client.evaluate(server, ell= i, test = "local")
            #glob_loss, glob_acc = client.evaluate(server, ell= i, test = "global")
        
            loss_clients_train.append(train_loss); acc_clients_train.append(train_acc)    
            loss_clients_test.append(test_loss); acc_clients_test.append(test_acc)    
            #loss_clients_glob.append(glob_loss); acc_clients_glob.append(glob_acc)
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
        
        w_locals_server = server.aggregation()
        
        print("-----------------------------------------------------------")
        print("------ FedServer: Federation process at Client-Side ------- ")
        print("-----------------------------------------------------------")
        #w_glob_client, c = FedAvg(w_locals_client) 
        if args.AR == "fedavg" :
            w_glob_client, c = FedAvg(w_locals_client)  
            w_glob_server, t = FedAvg(w_locals_server)
        elif args.AR == "mudhog" :
            if args.side == "server" :
                w_glob_server, t = mudhog_object.aggregator(w_locals_server, server)
            elif args.side == "client" :    
                w_glob_client, c = mudhog_object.aggregator(w_locals_client, clients)
            elif args.side == "both" :    
                w_glob_client, c = mudhog_client.aggregator(w_locals_client, clients)
                w_glob_server, t = mudhog_server.aggregator(w_locals_server, server)
        elif args.AR == "flame" :
            if args.side == "client" :
                w_glob_prev = copy.deepcopy(w_glob_client)
                w_glob_client, c = flame_object.aggregator(w_glob_prev, w_locals_client,"clients") 
            elif args.side == "server" :    
                w_glob_prev = copy.deepcopy(w_glob_server)
                w_glob_server, t = flame_object.aggregator(w_glob_prev, w_locals_server, "server")   
            elif args.side == "both" :
                w_glob_prev = copy.deepcopy(w_glob_client)
                w_glob_client, c = flame_client.aggregator(w_glob_prev, w_locals_client,"clients") 
                w_glob_prev = copy.deepcopy(w_glob_server)
                w_glob_server, t = flame_server.aggregator(w_glob_prev, w_locals_server, "server") 
        elif args.AR == "new" :
            w_glob_client, c = server.aggregate(w_locals_client,i,"client")
            w_glob_server, t = server.aggregate(w_locals_server,i,"server")
        elif args.AR == "plr" :
            w_glob_client, c = server.plr_aggregate(w_locals_client,i,"client")
            w_glob_server, t = server.plr_aggregate(w_locals_server,i,"server")
        elif args.AR == "gac" :
            w_glob_client, c = server.gac_aggregate(w_locals_client,i,"client")
            w_glob_server, t = server.gac_aggregate(w_locals_server,i,"server")
                
            
        # Update client-side global model 
        
        l, a = eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = eval_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        client_carbon.append(client_temp); server_carbon.append(server_temp)
        client_agg_carbon.append(c); server_agg_carbon.append(t)
        uplink.append(up); downlink.append(down)
        #l, a = eval_glob(i, acc_clients_glob, loss_clients_glob)
        #loss_glob.append(l); acc_glob.append(a)
    return loss_train, acc_train, loss_test, acc_test, client_carbon, server_carbon, client_agg_carbon, server_agg_carbon, uplink, downlink

def Fed(args, trainData, testData) :
    device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================================================
    # No. of users
    num_users = args.num_clients
    epochs = args.epochs
    local_epochs = args.inner_epochs
    #frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
    lr = args.lr
    
    #net_glob_client = Net(10)
    #net_glob_client = ResNet18_client_side() 

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
    
    if args.attack.upper().count("LABEL_FLIPPING") :
        att = "LF"
        flip = label_flipping_setup(args.attack, args.label_flipping)
    if args.attack.upper().count("SIGN_FLIPPING") :
        att = "SF"
    if args.attack.upper().count("RANDOM_LABEL_FLIPPING") :
        att = "RLF"
    if args.attack.upper().count("DATA_POISONING") :
        att = "DP"
        mu, std = poisoning_setup(args.attack)
    if args.attack.upper().count("MODEL_POISONING") :
        att = "MP"
        mu, std = poisoning_setup(args.attack)
    
    for idx in idxs_users :
        net_glob_client = Net(38)
            
        #if torch.cuda.device_count() > 1:
        #    print("We use",torch.cuda.device_count(), "GPUs")
        #    net_glob_client = nn.DataParallel(net_glob_client)  
            
        #net_glob_client.to(device)

        optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = lr)    
        
        if idx < args.scale :
            if att == "SF": clients.append(Attacker_SignFlipping(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
            if att == "RLF": clients.append(Attacker_Random(net_glob_client, args.PDR, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
            if att == "LF": clients.append(Attacker_LF(net_glob_client, args.PDR, flip, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
            if att == "DP": clients.append(Attacker_DataPoisoning(net_glob_client, args.PDR, mu, std, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
            if att == "MP": clients.append(Attacker_ModelPoisoning(net_glob_client, mu, std, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))
        else :    
            clients.append(Client(net_glob_client, idx, lr, device, optimizer_client, trainData[idx], testData, local_ep = local_epochs))  
    
    if args.AR == "mudhog" :
        mudhog_object = MuDHoG()
    if args.AR == "flame" :
        flame_object = FLAME()
    
    #------------ Training And Testing  -----------------
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    loss_train = []; acc_train = []
    loss_test = []; acc_test = []
    client_carbon = []; server_carbon = []
    client_agg_carbon = []; server_agg_carbon = []
    uplink = []; downlink = []
    #loss_glob = []; acc_glob = []
    for i in range(epochs):
        w_locals_client = []
        loss_clients_train = []; acc_clients_train = []
        loss_clients_test = []; acc_clients_test = []
        client_temp = 0; server_temp = 0
        up = 0; down = 0
        #loss_clients_glob = []; acc_clients_glob = []
        for client in clients:
            buffer = io.BytesIO()
            torch.save(w_glob_client, buffer)
            down += buffer.tell()
            #down += w_glob_client.element_size() * w_glob_client.nelement()
            client_temp += client.setModelParameter(w_glob_client) 
            # Training ------------------
            train_loss, train_acc, w_client, client_emissions, server_emissions, u, d = client.train_federated()
            w_locals_client.append(copy.deepcopy(w_client))
            client_temp += client_emissions; server_temp += server_emissions
            
            buffer = io.BytesIO()
            torch.save(w_client, buffer)
            up += buffer.tell()
            #up += w_client.element_size() * w_client.nelement()
            
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
        if args.AR == "fedavg" :
            w_glob_client, c = FedAvg(w_locals_client)  
        elif args.AR == "mudhog" :
            w_glob_client, c = mudhog_object.aggregator(w_locals_client, clients)
        elif args.AR == "flame" :
            w_glob_prev = copy.deepcopy(w_glob_client)
            w_glob_client, c = flame_object.aggregator(w_glob_prev, w_locals_client,"clients")
        
        # Update client-side global model 
        
        l, a = eval_train(i, acc_clients_train, loss_clients_train)
        loss_train.append(l); acc_train.append(a)
        l, a = eval_fed(i, acc_clients_test, loss_clients_test)
        loss_test.append(l); acc_test.append(a)
        client_agg_carbon.append(0.0); server_agg_carbon.append(c)
        client_carbon.append(client_temp); server_carbon.append(server_temp)
        uplink.append(up); downlink.append(down)
        #l, a = eval_glob(i, acc_clients_glob, loss_clients_glob)
        #loss_glob.append(l); acc_glob.append(a)
    return loss_train, acc_train, loss_test, acc_test, client_carbon, server_carbon, client_agg_carbon, server_agg_carbon, uplink, downlink