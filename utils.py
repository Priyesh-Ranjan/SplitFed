import torch
import copy
import numpy as np
from codecarbon import EmissionsTracker
#from mudhog import mud_hog_aggregation
import logging

# Federated averaging: FedAvg
def FedAvg(w):
    tracker = EmissionsTracker(log_level=logging.CRITICAL)
    tracker.start()
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    agg: float = tracker.stop()
    return w_avg, agg       

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc


def eval_train(ell, acc_train_collect_user, loss_train_collect_user) :
    # server-side global model update and distribute that model to all clients ------------------------------
    loss_train, acc_train = np.average(loss_train_collect_user), np.average(acc_train_collect_user)
        
    #print("====================== SERVER V1==========================")
    print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_train, loss_train))
        
    return loss_train, acc_train


def eval_fed(ell, acc_test_collect_user, loss_test_collect_user) :        
        
    loss_test, acc_test = np.average(loss_test_collect_user), np.average(acc_test_collect_user)
                      
        
    print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_test, loss_test))
    print("==========================================================") 
        
    return loss_test, acc_test


def eval_glob(ell, acc_glob_collect_user, loss_glob_collect_user) :        
        
    loss_glob, acc_glob = np.average(loss_glob_collect_user), np.average(acc_glob_collect_user)
                      
        
    print(' Global: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_glob, loss_glob))
    print("==========================================================") 
        
    return loss_glob, acc_glob