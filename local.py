import copy
import torch
from data import Mydataset
from torch.utils.data import DataLoader


class Local:
    def __init__(self, args, c_id):
        # important arguments
        self.c_id = c_id
        self.args = args

        # dataset
        self.data_loader = None
        self.dataset = None
        self.nb_unique_label = 0

        # model & optimizer
        self.model = None
        self.optim = None
        self.loss_func = torch.nn.NLLLoss(reduction='mean')
        
        # recovery signal
        self.recovery = None

    def train(self):
        train_loss, itr, ep = 0, 0, 0
        for ep in range(self.args.local_ep):
            # correct = 0
            for itr, (x, y) in enumerate(self.data_loader):
                logprobs = self.model(x)
                loss = self.loss_func(logprobs, y)

                # y_pred = torch.argmax(torch.exp(logprobs), dim=1)
                # correct += torch.sum(y_pred.view(-1) == y.view(-1)).cpu().item()
                train_loss += loss

                self.model.zero_grad()
                loss.backward()
                self.optim.step()

            # print(f"{self.c_id}th Client | Train loss: {train_loss / (itr + 1):.3f} | Train Acc: {correct * 100 / len(self.dataset['y']):.3f}@ {ep}")
        return train_loss / ((ep+1) * (itr+1))
    
    def get_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(Mydataset(self.dataset, self.args), batch_size=self.args.local_bs, shuffle=True)
        if len(dataset['y']) <= 0:
            raise RuntimeError

    def set_unique(self, nb_unique):
        self.nb_unique_label = nb_unique

    def get_model(self, model):
        self.model = copy.deepcopy(model)
        if 'sgd' == self.args.optimizer:
            self.optim = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        elif 'adam' == self.args.optimizer:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError

    def upload_model(self):
        return copy.deepcopy(self.model.state_dict())
    
    def upload_recovery_signal(self):
        return copy.deepcopy(self.recovery)
    
    def _recoverer(self):
        """To be implemented"""
        self.recovery = None
