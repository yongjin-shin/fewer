# wandb
group = 'test'
name = 'test'
tags = ['test']

configdict = {
    # Experiment information
    "exp_info":{
        "project_name": "redpill_bluepill",
        "name": name, "group": group, "tags": tags, "notes": ''
    },
    
    # Federated Learning setups
    "fed_setups":{
        "model": 'cifarcnn', # cifarcnn, res8
        "server_params":{
            "n_rounds" : 300, "n_clients" : 100, "sample_ratio": 0.1,
            "agg_alg" : 'fedavg',
            "server_location" : 'cpu',        
            "device": 'cuda:0'
        },
        "local_params":{
            "local_ep" : 5, "local_bs" : 50,
            "global_loss" : 'none', # none, l2
            "global_alpha" : 0
        }
    },
    
    # Data setups
    "data_setups": {
        "root": './data',
        "dataset": 'cifar10', # mnist, fmnist, cifar10, cifar100
        "param": {"alg": 'fedavg', "max_class_num": 2, "dir_alpha": 0.5} # alg: fedavg, fedma
    },
    
    # Training setups
    "criterion": {
        "param": {"mode": 'CE', "smoothing": 0, "beta": 0} # mode: CE, LS, KD
    },
    "optimizer": {
        "param": {"lr": 0.01, "momentum":0.9, "weight_decay": 0}
    },
    "scheduler": {
        "enabled": False,
        "type": 'cosine', # cosine, multistep
        "param": {"T_max": 300}
    },
    "seed": 2021
}
