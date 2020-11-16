# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from server import *
from train_tools import *
from utils import *

import numpy as np
import wandb, argparse, json, os
import warnings

warnings.filterwarnings('ignore')

# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description='Process Config Dicts')
parser.add_argument('--config_path', default='./config/default.py', type=str)
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

# Load a configuration file
with open(args.config_path) as f:
    config_code = f.read()
    exec(config_code) 

torch.set_printoptions(10)

MODEL = {}

################################################################### 
SCHEDULER = {'step': lr_scheduler.StepLR,
            'multistep': lr_scheduler.MultiStepLR,
            'cosine': lr_scheduler.CosineAnnealingLR}


def _get_setups(opt):
    # datasets
    datasetter = DataSetter(root='./data', dataset=opt.data_setups.dataset)
    datasets = datasetter.data_distributer(**opt.data_setups.param.__dict__)
    
    # train setups
    model = create_nets(model=opt.fed_setups.model, 
                        dataset=datasetter.dataset, 
                        num_classes=datasetter.num_classes)
    
    criterion = OverhaulLoss(**opt.criterion.param.__dict__)
    optimizer = optim.SGD(model.parameters(), **opt.optimizer.param.__dict__)
    
    if opt.scheduler.enabled:
        scheduler = SCHEDULER[opt.scheduler.type](optimizer, **opt.scheduler.param.__dict__)
    else:
        scheduler = None
    
    return datasets, model, criterion, optimizer, scheduler


################################################################################################################

def main():
    # Fix randomness
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    
    # Setups
    datasets, model, criterion, optimizer, scheduler = _get_setups(opt)
    server = Server(datasets, model, criterion, optimizer, scheduler,
                    local_args=opt.fed_setups.local_params,
                    **opt.fed_setups.server_params.__dict__)
        
    #wandb.watch(server.model, log='parameters') # inspects server model
    
    save_path = os.path.join('./results', opt.exp_info.name)
    directory_setter(save_path, make_dir=True)

    # Federeted Learning
    total_result = server.train()
    

    # Save results
    result_path = os.path.join(save_path, 'results.json')
    
    with open(result_path, 'w') as f:
        json.dump(result_path, f)
    
    model_path = os.path.join(save_path, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Upload to wandb
    wandb.save(model_path)
    wandb.save(result_path)

    
    
if __name__ == '__main__':
    opt = objectview(configdict)
    
    # Initialize wandb
    wandb.init(project=opt.exp_info.project_name, 
               name=opt.exp_info.name, 
               tags=opt.exp_info.tags,
               group=opt.exp_info.group,
               notes=opt.exp_info.notes, 
               config=configdict)
    
    main()
