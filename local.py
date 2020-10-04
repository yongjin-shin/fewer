import torch, math, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_tools.utils import create_nets
from train_tools.criterion import OverhaulLoss
from train_tools.SparsityController.recovering_utils import *

__all__ = ['Local']


class Local:
    def __init__(self, args):
        self.args = args
        self.data_loader = None

        # model & optimizer
        self.model = create_nets(self.args, 'LOCAL').to(self.args.server_location)
        self.optim, self.lr_scheduler = None, None
        self.criterion = OverhaulLoss(self.args)
        self.epochs = self.args.local_ep

    def train(self):
        if (self.args.global_loss_type != 'none'):
            self.keep_global()
            
        if (self.args.mode == 'KD') or (self.args.mode =='FedLSD'):
            self.keep_global()
            t_logits = None

        train_loss, train_acc, itr, ep = 0, 0, 0, 0
        self.model.to(self.args.device)

        for ep in range(self.epochs):        
            for itr, (data, target) in enumerate(self.data_loader):
                # forward pass
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                output = self.model(data)
                if (self.args.mode == 'KD') or (self.args.mode =='FedLSD'):
                    with torch.no_grad():
                        t_logits = self.round_global(data)
                        
                loss = self.criterion(output, target, t_logits)
                
                if self.args.global_loss_type != 'none':
                    loss += (self.args.global_alpha * self.loss_to_round_global())
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_loss += loss.detach().item()
        
        
        with torch.no_grad():
            correct, data_num = 0, 0
        
            for itr, (data, target) in enumerate(self.data_loader):
                data_num += data.size(0)
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                output = self.model(data)
                pred = torch.max(output, dim=1)[1]
                correct += (pred == target).sum().item()
            local_acc = round(correct/data_num, 4)
        
        #local_acc = 0

        self.model.to(self.args.server_location)
        local_loss = train_loss / ((self.args.local_ep) * (itr+1))
        
        return local_loss, local_acc

    def stack_grad(self):
        """stack gradient to local parameters"""
        self.model.to(self.args.device)
        self.optim.zero_grad()
            
        for data, target in self.data_loader:
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
        self.model.to(self.args.server_location)

    def get_dataset(self, client_dataset):
        if client_dataset.__len__() <= 0:
            raise RuntimeError
        else:
            self.data_loader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True)
            
    def get_model(self, server_model):
        self.model.load_state_dict(server_model)
        
    def get_lr(self, server_lr):
        if server_lr < 0:
            raise RuntimeError("Less than 0")
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=server_lr,
                                     momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
        
    def upload_model(self):
        return copy.deepcopy(self.model.state_dict())
    
    def reset(self):
        self.data_loader = None
        self.optim = None
        self.lr_scheduler = None
        self.round_global = None

    def keep_global(self):
        self.round_global = copy.deepcopy(self.model)
        for params in self.round_global.parameters():
            params.requires_grad = False
        
    def loss_to_round_global(self):
        vec = []
        if self.args.global_loss_type == 'l2':
            for i, (param1, param2) in enumerate(zip(self.model.parameters(), self.round_global.parameters())):
                if self.args.no_reg_to_recover:
                    raise NotImplemented
                else:
                    vec.append((param1-param2).view(-1, 1))

            all_vec = torch.cat(vec)
            loss = torch.norm(all_vec)  # (all_vec** 2).sum().sqrt()
        else:
            raise NotImplemented
        
        return loss * 0.5
