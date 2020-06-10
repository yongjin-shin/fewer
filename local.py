import copy
import torch
from torch.utils.data import DataLoader
from pruning import *
from networks import create_nets
from misc import model_location_switch_uploading


class Local:
    def __init__(self, args):
        self.args = args
        self.data_loader = None

        # model & optimizer
        self.model = create_nets(self.args, 'LOCAL').to(self.args.device)
        self.optim = None
        self.loss_func = torch.nn.NLLLoss(reduction='mean')

    def train(self):
        train_loss, itr, ep = 0, 0, 0
        for ep in range(self.args.local_ep):
            for itr, (x, y) in enumerate(self.data_loader):
                self.optim.zero_grad()

                logprobs = self.model(x.to(self.args.device))
                loss = self.loss_func(logprobs, y.to(self.args.device))
                loss.backward()

                train_loss += loss

                self.optim.step()

        return train_loss / ((ep+1) * (itr+1))

    def train_with_recovery(self, keeped_masks):
        train_loss, itr, ep = 0, 0, 0

        for ep in range(self.args.local_ep):
            for itr, (x, y) in enumerate(self.data_loader):
                # reset w_dense gradients
                self.optim.zero_grad()
                
                # w_tilda <- mask * w_dense
                with torch.no_grad():
                    masked_model = copy.deepcopy(self.model)
                    mask_adder(masked_model, keeped_masks)
                    mask_merger(masked_model)

                # compute gradient from w_tilda
                logprobs = masked_model(x.to(self.args.device))
                loss = self.loss_func(logprobs, y.to(self.args.device))
                loss.backward()
                train_loss += loss

                # deliver gradients from w_tilda to w_dense
                for dense_param, mask_param in zip(self.model.parameters(), masked_model.parameters()):
                    dense_param.grad = mask_param.grad
                
                # update w_dense weights
                self.optim.step()

        return train_loss / ((ep+1) * (itr+1))

    def get_dataset(self, client_dataset):
        self.data_loader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True)
        if client_dataset.__len__() <= 0:
            raise RuntimeError

    def get_model(self, server_model):
        self.model.load_state_dict(server_model)

        if 'sgd' == self.args.optimizer:
            self.optim = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)

        elif 'adam' == self.args.optimizer:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        else:
            raise NotImplementedError

    def reset(self):
        self.data_loader = None

    def upload_model(self):
        return model_location_switch_uploading(model=self.model,
                                               args=self.args)
