from torch.utils.data import DataLoader
import numpy as np
import copy
import torch

# Related Classes
from local import Local
from data import Mydataset
from networks import MLP, MnistCNN, CifarCnn, TestCNN, VGG
from aggregation import get_aggregation_func
from pruning import *

class Server:
    def __init__(self, args):
        """ N개의 Local은 여기서 만들어진다!!!"""
        # important variables
        self.args = args
        self.locals = [Local(args=args, c_id=i) for i in range(self.args.nb_devices)]
        
        # pruning handler
        self.pruning_handler = PruningHandler(args)
        self.sparsity = 0
        
        # dataset
        self.dataset_train, self.dataset_test = None, None
        self.test_loader = None

        # about model
        self.model = None
        self.loss_func = torch.nn.NLLLoss(reduction='mean')
        self.aggregate_model_func = get_aggregation_func(self.args.aggregation_alg)

        # misc
        self.container = []
        self.nb_unique_label = 0
        self.nb_client_per_round = max(int(self.args.ratio_clients_per_round * self.args.nb_devices), 1)
        self.sampling_clients = lambda nb_samples: np.random.choice(self.args.nb_devices, nb_samples, replace=False)  
        
    def get_data(self, dataset_server, dataset_locals, dataset_test):
        """raw data를 받아와서 server와 local에 데이터를 분배함"""
        self.dataset_train, self.dataset_test = dataset_server, dataset_test

        dataset_test = Mydataset(self.dataset_test, self.args)
        self.test_loader = DataLoader(dataset_test, batch_size=100, shuffle=True)
        self.nb_unique_label = dataset_test.unique()

        for i in range(self.args.nb_devices):
            """set unique must be prior to get_dataset"""
            self.locals[i].set_unique(self.nb_unique_label)
            self.locals[i].get_dataset(dataset_locals[i])

    def make_model(self):
        if self.args.model == 'mlp':
            model = MLP(self.dataset_test['x'][0].shape.numel(), self.args.hidden,
                        torch.unique(self.dataset_test['y']).numel()).to(self.args.device)
        elif self.args.model == 'mnistcnn':
            model = MnistCNN(1, torch.unique(self.dataset_test['y']).numel()).to(self.args.device)
        elif self.args.model == 'cifarcnn':
            model = CifarCnn(3, torch.unique(self.dataset_test['y']).numel()).to(self.args.device)
        elif self.args.model == 'testcnn':
            model = TestCNN(3 if 'cifar' in self.args.dataset else 1,
                            torch.unique(self.dataset_test['y']).numel()).to(self.args.device)
        elif self.args.model == 'vgg':
            model = VGG(3 if 'cifar' in self.args.dataset else 1,
                            torch.unique(self.dataset_test['y']).numel()).to(self.args.device)
        else:
            raise NotImplementedError

        self.model = model
        print(model)

    def train(self, exp_id=None):
        """Distribute, Train, Aggregation and Test"""
        for r in range(self.args.nb_rounds):
            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            
            # pruning step
            self.model, keeped_masks = self.pruning_handler.pruner(self.model, r)

            # distribution step
            self.distribute_models(sampled_devices, self.model)
            
            # client training & upload models
            train_loss, updated_locals, recovery_signals = self.clients_training(sampled_devices, keeped_masks)
            
            # recovery step
            self.pruning_handler.recoverer(self.model, recovery_signals, r)
            
            # aggregation step
            self.aggregation_models(updated_locals)
            
            # test & log results
            test_loss, test_acc = self.test()
            self.logging(train_loss.item(), test_loss, test_acc, r, exp_id)

        return self.container

    def clients_training(self, sampled_devices, keeped_masks=None):
        """Local의 training 하나씩 실행함. multiprocessing은 구현하지 않았음."""
        updated_locals = []
        recovery_signals = []
        train_loss = 0

        for i in sampled_devices:
            if keeped_masks is not None:    
                # get and apply pruned mask from global
                self.pruning_handler.mask_adder(self.locals[i].model, keeped_masks)
            
            train_loss += self.locals[i].train()
            
            # merge mask of local (remove masks but pruned weights are still zero)
            self.pruning_handler.mask_merger(self.locals[i].model)
            
            updated_locals.append(self.locals[i].upload_model())
            recovery_signals.append(self.locals[i].upload_recovery_signal())

        train_loss /= len(sampled_devices)
        return train_loss, updated_locals, recovery_signals

    def distribute_models(self, sampled_devices, model):
        for i in sampled_devices:
            self.locals[i].get_model(copy.deepcopy(model))
        # print(f"Devices will be training: {sampled_devices}")

    def aggregation_models(self, updated_locals):
        self.model.load_state_dict(copy.deepcopy(self.aggregate_model_func(updated_locals)))

    def test(self):
        """ Aggregation 되어있는 Global Model로 Test 진행"""
        self.model.eval()
        test_loss, correct, itr = 0, 0, 0
        for itr, (x, y) in enumerate(self.test_loader):
            logprobs = self.model(x)
            test_loss += self.loss_func(logprobs, y).item()
            y_pred = torch.argmax(torch.exp(logprobs), dim=1)
            correct += torch.sum(y_pred.view(-1) == y.view(-1)).cpu().item()
            
        current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
        print('Current Sparsity: %0.4f' % current_sparsity)

        self.model.train()
        return test_loss / (itr + 1), 100 * float(correct) / float(len(self.dataset_test['y']))

    def logging(self, train_loss, test_loss, test_acc, r, exp_id=None):
        self.container.append([train_loss, test_loss, test_acc, r, exp_id])
        print(f"{r}th round | Train loss: {train_loss:.3f} Test loss: {test_loss:.3f} | acc: {test_acc:.3f} ")

