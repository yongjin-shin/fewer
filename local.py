import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_tools import *

import copy, random
import numpy as np

__all__ = ['Local']


class Local:
    def __init__(self, model, criterion, testset=None, local_ep=5, local_bs=50, 
                 global_loss=['none', 0], server_location='cpu', device='cuda:0'):

        # model & optimizer
        self.model = model
        self.criterion = criterion
        self.local_ep, self.local_bs = local_ep, local_bs
        self.global_loss_type, self.global_loss_alpha = global_loss[0], global_loss[1]
        
        self.server_location = server_location
        self.device = device
        
        if testset is not None:
            self.testloader = DataLoader(testset, batch_size=local_bs)
        
        # initialize local client
        self.reset()

        
    def train(self, test_evaluation=True):
        self.model.to(self.device)
        
        if (self.global_loss_type != 'none') or ('D' in self.criterion.mode):
            self._keep_global()
        
        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        t_logits, t_features = None, None

        for ep in range(self.local_ep):
            for itr, (data, target) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                # forward pass
                data, target = data.to(self.device), target.to(self.device)
                output, features = self.model(data, get_features=True)
                
                if 'D' in self.criterion.mode:  # 'D' stands for distillation
                    with torch.no_grad():
                        t_logits, t_features = self.round_global(data, get_features=True)
                
                loss = self.criterion(output, target, t_logits, features, t_features)

                if self.global_loss_type != 'none' and self.global_alpha > 0:
                    loss += (self.global_alpha * self._loss_to_round_global())
                
                # backward pass
                loss.backward()
                self.optimizer.step()

            
            if self.scheduler is not None:
                self.scheduler.step()
        
        # get stats from local trainset
        train_loss, train_acc = model_evaluator(self.model, self.dataloader, self.criterion, self.device)
        
        # get stats from testset
        if (self.testloader is not None) and test_evaluation:
            test_loss, test_acc =  model_evaluator(self.model, self.testloader, self.criterion, self.device)

        result = {
            'train_loss': train_loss, 'train_acc' : train_acc, 
            'test_loss' : test_loss, 'test_acc' : test_acc
        }

        self.model.to(self.server_location)
        
        return result
            
        
    def download_global(self, client_dataset, server_weights, server_optimizer):
        self.dataloader = DataLoader(client_dataset, batch_size=self.local_bs, shuffle=True)
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        
        
    def upload_local(self):
        return copy.deepcopy(self.model.state_dict())

    
    def reset(self):
        self.dataloader = None
        self.round_global = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.scheduler = None

        
    def _keep_global(self):
        self.round_global = copy.deepcopy(self.model)
        for params in self.round_global.parameters():
            params.requires_grad = False
        
        
    def _loss_to_round_global(self):
        vec = []
        if self.global_loss_type == 'l2':
            for i, ((name1, param1), (name2, param2)) in enumerate(zip(self.model.named_parameters(),
                                                                       self.round_global.named_parameters())):
                if name1 != name2:
                    raise RuntimeError

                else:
                    vec.append((param1 - param2).view(-1, 1))

            all_vec = torch.cat(vec)
            loss = torch.norm(all_vec)  # (all_vec** 2).sum().sqrt()
        else:
            raise NotImplemented
        
        return loss * 0.5
