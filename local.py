
import copy
import torch
from torch.utils.data import DataLoader
from networks import MLP, MnistCNN, CifarCnn, TestCNN

class Local:
    def __init__(self, args, c_id,update_per_iter):
        # important arguments
        self.c_id = c_id
        self.args = args
        self.update_per_iter = update_per_iter
        # dataset
        self.data_loader = None
        self.dataset = None

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
                logprobs = self.model(x.to(self.args.device))
                loss = self.loss_func(logprobs, y.to(self.args.device))

                # y_pred = torch.argmax(torch.exp(logprobs), dim=1)
                # correct += torch.sum(y_pred.view(-1) == y.view(-1)).cpu().item()
                train_loss += loss

                self.model.zero_grad()

                loss.backward()

                self.optim.step()

            # print(f"{self.c_id}th Client | Train loss: {train_loss / (itr + 1):.3f} | Train Acc: {correct * 100 / len(self.dataset['y']):.3f}@ {ep}")
        return train_loss / ((ep+1) * (itr+1))


    def train_with_recovery(self, keeped_masks):
        train_loss, itr, ep = 0, 0, 0
        self.keeped_masks = keeped_masks
        # for mask_param,dense_param in zip(self.model.named_parameters(),self.model_dense.named_parameters()):
        #     print(mask_param[0])
        #     print(dense_param[0])
        #     print(mask_param[1].size())
        #     print(dense_param[1].size())


        '''get masked model for initial step, for now search mask fit to param by shape'''

        for mask_param in self.model.parameters():
            for keeped_mask in keeped_masks:
                if mask_param.size() == keeped_masks[keeped_mask].size():
                    mask_param = torch.mul(mask_param,keeped_masks[keeped_mask])


        for ep in range(self.args.local_ep):
            for itr, (x, y) in enumerate(self.data_loader):

                if itr % self.update_per_iter:
                    '''mask dense model per update_per_itr'''
                    for dense_param in self.model_dense.parameters():
                        for keeped_mask in keeped_masks:
                            if dense_param.size() == keeped_masks[keeped_mask].size():
                                dense_param = torch.mul(dense_param, keeped_masks[keeped_mask])
                    '''update masked dense model (m*w_{t+1}) to masked model (m*(w_{t}) '''
                    self.model = copy.deepcopy(self.model_dense)

                logprobs = self.model(x)
                loss = self.loss_func(logprobs, y)

                train_loss += loss

                self.model.zero_grad()
                self.model_dense.zero_grad()
                '''get gradient of masked model'''
                loss.backward()

                '''deliver gradient of masked model parameter to dense model '''
                for mask_param, dense_param in zip(self.model.parameters(),self.model_dense.parameters()):
                    dense_param.grad = copy.deepcopy(mask_param.grad)

                '''update dense model from gradient of masked model'''
                self.optim_dense.step()

        return train_loss / ((ep+1) * (itr+1))

    def get_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=self.args.local_bs, shuffle=True)
        if self.dataset.__len__() <= 0:
            raise RuntimeError

    def get_model(self, model):
        self.model = copy.deepcopy(model)

        import pickle
        self.model_dense = copy.deepcopy(pickle.loads(pickle.dumps(model)))

        if 'sgd' == self.args.optimizer:
            self.optim = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
            self.optim_dense = torch.optim.SGD(self.model_dense.parameters(),
                                         lr=self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        elif 'adam' == self.args.optimizer:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
            self.optim_dense = torch.optim.Adam(self.model_dense.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError

    def upload_model(self):
        return copy.deepcopy(self.model.state_dict())

    def upload_recovery_model(self):
        return copy.deepcopy(self.model_dense.state_dict())
    
    def upload_recovery_signal(self):
        return copy.deepcopy(self.recovery)
    
    def _recoverer(self):
        """To be implemented"""
        self.recovery = None
