import torch, math, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from train_tools.utils import create_nets
from train_tools.SparsityController.recovering_utils import *
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Local']


class Local:
    def __init__(self, args):
        self.args = args
        self.data_loader = None

        # Dataset
        self.valid_loader = None
        self.valid_dataset = None

        # model & optimizer
        self.model = create_nets(self.args, 'LOCAL').to(self.args.server_location)
        self.optim, self.lr_scheduler = None, None
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = self.args.local_ep

    def train(self):
        if self.args.global_loss_type != 'none':
            self.keep_global()

        train_loss, train_acc, itr, ep = 0, 0, 0, 0
        self.model.to(self.args.device)

        for ep in range(self.epochs):        
            for itr, (data, target) in enumerate(self.data_loader):
                # forward pass
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if self.args.global_loss_type != 'none':
                    loss += (self.args.global_alpha * self.loss_to_round_global())
                
                # backward pass
                self.optim.zero_grad()
                # self.apply_partial_gradient(itr)
                loss.backward()
                self.optim.step()

                train_loss += loss.detach().item()

        # with torch.no_grad():
        #     correct, data_num = 0, 0
        #     # t = []
        #     for itr, (data, target) in enumerate(self.data_loader):
        #         data_num += data.size(0)
        #         data = data.to(self.args.device)
        #         target = target.to(self.args.device)
        #         # t.append(target)
        #         output = self.model(data)
        #         pred = torch.max(output, dim=1)[1]
        #         correct += (pred == target).sum().item()
        #     local_acc = round(correct/data_num, 4)
        #     # print(f"Local: {torch.unique(torch.cat(t))}")
        #
        #     valid_correct, data_num = 0, 0
        #     # t = []
        #     for itr, (data, target) in enumerate(self.valid_loader):
        #         data_num += data.size(0)
        #         data = data.to(self.args.device)
        #         target = target.to(self.args.device)
        #         # t.append(target)
        #         output = self.model(data)
        #         pred = torch.max(output, dim=1)[1]
        #         valid_correct += (pred == target).sum().item()
        #     valid_acc = round(valid_correct/data_num, 4)
        #     # print(f"Valid: {torch.unique(torch.tensor(t))}")

        local_acc, valid_acc = None, None
        self.model.to(self.args.server_location)
        local_loss = train_loss / ((self.args.local_ep) * (itr+1))
        
        return local_loss, local_acc, valid_acc

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

    def get_dataset(self, client_dataset, valid_idx):
        if client_dataset.__len__() <= 0:
            raise RuntimeError
        else:
            # self.plotter(client_dataset)
            # self.plotter(Subset(self.valid_dataset, valid_idx))
            self.data_loader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True)
            self.valid_loader = DataLoader(Subset(self.valid_dataset, valid_idx), shuffle=True)

    def plotter(self, dataset):
        loader = DataLoader(dataset, batch_size=1)
        X, Y = [], []
        for x, y in loader:
            X.append(x.numpy())
            Y.append(y.numpy())

        x = np.concatenate(X).squeeze()
        y = np.concatenate(Y).squeeze()
        lb = np.unique(y)

        fig = plt.figure(figsize=(10, 2))

        for _l in lb:
            _idx = np.random.choice(np.where(y == _l)[0], 5)
            print(y[_idx])
            for j, _i in enumerate(_idx):
                fig.add_subplot(j+1, 1, j+1)
                plt.imshow(x[_i])
            plt.show()
            plt.close()

    def get_validset(self, valid_dataset):
        self.valid_dataset = valid_dataset

    def get_model(self, server_model):
        self.model.load_state_dict(server_model)
        
    def get_lr(self, server_lr):
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
        loss = 0

        for i, (param1, param2) in enumerate(zip(self.model.parameters(), self.round_global.parameters())):
            if self.args.global_loss_type == 'smooth_l1':
                if self.args.no_reg_to_recover:
                    global_mask = (param2 != 0).int()
                    loss += F.smooth_l1_loss(param1*global_mask, param2*global_mask, reduction='sum')
                else:
                    loss += F.smooth_l1_loss(param1, param2, reduction='sum')
                
            elif self.args.global_loss_type == 'cosine_similarity':
                loss += -F.cosine_similarity(param1.view(-1), param2.view(-1), dim=0)
        
        return loss
