
import copy
import torch
from data import Mydataset
from torch.utils.data import DataLoader
from networks import MLP, MnistCNN, CifarCnn, TestCNN

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


    def train_with_recovery(self):
        train_loss, itr, ep = 0, 0, 0
        for ep in range(self.args.local_ep):
            # correct = 0
            for itr, (x, y) in enumerate(self.data_loader):
                logprobs = self.model(x)
                temp_probs = self.model_dense(x)
                loss = self.loss_func(logprobs, y)

                train_loss += loss

                self.model.zero_grad()
                self.model_dense.zero_grad()
                loss.backward()

                # print('start')

                for mask_param, dense_param in zip(self.model.parameters(),self.model_dense.parameters()):
                    print('============================')
                    print(dense_param.grad)
                    dense_param.grad = mask_param.grad
                    print(dense_param.grad)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                # self.optim.step()
                self.optim_dense.step()

        return train_loss / ((ep+1) * (itr+1))

    def get_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(Mydataset(self.dataset, self.args), batch_size=self.args.local_bs, shuffle=True)
        if len(dataset['y']) <= 0:
            raise RuntimeError

    def set_unique(self, nb_unique):
        self.nb_unique_label = nb_unique

    def get_model(self, model,model_name,model_reference):
        self.model = copy.deepcopy(model)

        import pickle
        self.model_dense = copy.deepcopy(pickle.loads(pickle.dumps(model)))

        # self.model_dense = copy.deepcopy(model_reference)
        #
        # params1 = self.model.named_parameters()
        # params2 = self.model_dense.named_parameters()
        #
        #
        # import pickle
        #
        # for name1, param1 in params1:
        #     print(name1)
        #     for name2, param2 in params2:
        #         print('==========')
        #         print(name1)
        #         print(name2)
                # if name1 == name2:

                    # param2.data.copy_(params)


        # # model.attribute = list(model.attribute)
        # self.model_dense  = copy.deepcopy(model_reference)
        #

        # dict_params2 = dict(params2)
        #
        # for name1, param1 in params1:
        #     if name1 in dict_params2:
        #         # print(name1)
        #         dict_params2[name1].data.copy_(param1.data)


        # self.model_dense= copy.deepcopy(model_reference)
        # print(self.model.state_dict())
        # self.model_dense.load_state_dict(torch.from_numpy(self.model.state_dict().cpu().numpy()))
        # self.model_dense.load_state_dict(dict_params2)


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
