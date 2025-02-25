import argparse
#from utils.decide_attack import attack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate of models")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--scale",type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar", "fmnist"], default="cifar")
    parser.add_argument("--loader_type", type=str, choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--AR", type=str, default="fedavg")
    parser.add_argument("--PDR",type=float,default=1.0)
    parser.add_argument("--attack", type=str)
    parser.add_argument("--label_flipping",nargs='?',type=str,choices=["uni","bi"], default="uni")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--inner_epochs", type=int, default=1)
    parser.add_argument("--setup", type=str, choices=["split","fed","split_fed"], default="split_fed")

    args = parser.parse_args()
    return args
