import torch, math, copy
import torch.nn as nn
from torch.utils.data import DataLoader
from train_tools.utils import create_nets

__all__ = ['Local']


class Local:
    def __init__(self, args):
        self.args = args
        self.data_loader = None

        # model & optimizer
        self.model = create_nets(self.args, 'LOCAL').to(self.args.server_location)
        self.optim, self.lr_scheduler = None, None
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = self.args.local_ep

    def train(self):
        train_loss, itr, ep = 0, 0, 0
        self.model.to(self.args.device)

        for ep in range(self.epochs):        
            for itr, (data, target) in enumerate(self.data_loader):
                # forward pass
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_loss += loss.detach().item()

        self.model.to(self.args.server_location)
        local_loss = train_loss / ((self.args.local_ep) * (itr+1))
        
        return local_loss    
   
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
        
    def upload_model(self):
        return copy.deepcopy(self.model.state_dict())
    
    def upload_optim(self):
        return [copy.deepcopy(self.optim.state_dict()),
                copy.deepcopy(self.lr_scheduler.state_dict())]

    def reset(self):
        self.data_loader = None
        self.optim = None
        self.lr_scheduler = None
