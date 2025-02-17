#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from pandas import DataFrame

import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
from dataset import prepare_dataset
from algorithms import Split, Fed, Split_Fed

def main(args) :
    SEED = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))    

    #===================================================================
    program = args.experiment_name
    print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

    num_users = args.num_clients
    dataset_train, dataset_test, dict_users, dict_users_test = prepare_dataset(num_users, args.dataset, args.loader_type)    
        
    #===================================================================================     
    
    if args.setup.upper() == "SPLIT" :
        loss_train, acc_train, loss_test, acc_test = Split(args, dataset_train, dataset_test, dict_users, dict_users_test)
    elif args.setup.upper() == "FED" :
        loss_train, acc_train, loss_test, acc_test = Fed(args, dataset_train, dataset_test, dict_users, dict_users_test)
    elif args.setup.upper() == "SPLIT_FED" :
        loss_train, acc_train, loss_test, acc_test = Split_Fed(args, dataset_train, dataset_test, dict_users, dict_users_test)
    
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