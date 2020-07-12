import copy
import torch
from torch.utils.data import DataLoader
from pruning import *
from networks import create_nets
from misc import model_location_switch_uploading
import math


class Local:
    def __init__(self, args):
        self.args = args
        self.data_loader = None

        # model & optimizer
        self.model = create_nets(self.args, 'LOCAL').to(self.args.device)
        self.optim, self.lr_scheduler = None, None
        if 'adam' == self.args.optimizer:
            raise NotImplementedError

        self.loss_func = torch.nn.NLLLoss(reduction='mean')
        self.epochs = self.args.local_ep

    def train(self):
        train_loss, itr, ep = 0, 0, 0
        
        for ep in range(self.epochs):            
            for itr, (x, y) in enumerate(self.data_loader):
                self.optim.zero_grad()

                logprobs = self.model(x.to(self.args.device))
                loss = self.loss_func(logprobs, y.to(self.args.device))
                loss.backward()

                train_loss += loss

                self.optim.step()

        return train_loss / ((self.args.local_ep) * (itr+1))
    
    def adapt_half_epochs(self, mode='half1'):
        if mode == 'half1':
            self.epochs = math.ceil(self.args.local_ep/2)
        elif mode == 'half2':
            self.epochs = math.floor(self.args.local_ep/2)        

    def get_dataset(self, client_dataset):
        if client_dataset.__len__() <= 0:
            raise RuntimeError
        else:
            self.data_loader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True)

    def get_model(self, server_model):
        self.model.load_state_dict(server_model)

    def get_lr(self, server_lr):

        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=server_lr,
                                     momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
        # self.optim.load_state_dict(server_optim)  # Todo: Momentum은 제대로 안 먹힘. 이거 aggregation 할 때랑 연결 지어서 해야함.

    def upload_model(self):
        return model_location_switch_uploading(model=self.model,
                                               args=self.args)

    def upload_optim(self):
        return [copy.deepcopy(self.optim.state_dict()),
                copy.deepcopy(self.lr_scheduler.state_dict())]

    def reset(self):
        self.data_loader = None
        self.optim = None
        self.lr_scheduler = None
