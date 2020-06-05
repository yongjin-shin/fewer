from torch.utils.data import DataLoader
import numpy as np
import copy
import torch

# Related Classes
from local import Local
from networks import MLP, MnistCNN, CifarCnn, TestCNN, VGG
from aggregation import get_aggregation_func
from pruning import *

# from pynvml import *


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
        self.model, self.model_reference = None, None
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

        self.test_loader = DataLoader(self.dataset_test, batch_size=100, shuffle=True)
        for i in range(self.args.nb_devices):
            self.locals[i].get_dataset(dataset_locals[i])

    def make_model(self):
        if 'mnist' in self.args.dataset:
            _in_dim = 1
        elif 'cifar' in self.args.dataset:
            _in_dim = 3
        else:
            raise NotImplementedError

        if self.args.model == 'mlp':
            model = MLP(784, self.args.hidden, 10).to(self.args.device)
            model_refer = MLP(784, self.args.hidden, 10).to(self.args.device)
        elif self.args.model == 'mnistcnn':
            model = MnistCNN(1, 10).to(self.args.device)
            model_refer = MnistCNN(1, 10).to(self.args.device)
        elif self.args.model == 'cifarcnn':
            model = CifarCnn(3, 10).to(self.args.device)
            model_refer = CifarCnn(3, 10).to(self.args.device)
        elif self.args.model == 'testcnn':
            model = TestCNN(_in_dim, 10).to(self.args.device)
            model_refer = TestCNN(_in_dim, 10).to(self.args.device)
        elif self.args.model == 'vgg':
            model = VGG(_in_dim, 10).to(self.args.device)
            model_refer = VGG(_in_dim, 10).to(self.args.device)
        else:
            raise NotImplementedError

        self.model = model
        self.model_reference = model_refer
        # print(model)

    def train(self, exp_id=None):
        """Distribute, Train, Aggregation and Test"""
        for r in range(self.args.nb_rounds):
            print('==================================================')
            print('Epoch [%d/%d]'%(r+1, self.args.nb_rounds))
            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            
            # global pruning step
            self.model, keeped_masks = self.pruning_handler.pruner(self.model, r)
            
            # distribution step
            current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            print('Downloading Sparsity : %0.4f' % current_sparsity)
            self.distribute_models(sampled_devices, self.model, self.args.model, self.model_reference)
        
            # client training & upload models
            train_loss, updated_locals = self.clients_training(sampled_devices,
                                                               keeped_masks=keeped_masks,
                                                               recovery=self.args.recovery,
                                                               model=self.args.model)
            
            # # recovery step
            local_sparsity = []
            for i in sampled_devices:
                _, keeped_local_mask = self.pruning_handler.pruner(self.locals[i].model, r)
                local_sparsity.append(self.pruning_handler.global_sparsity_evaluator(self.locals[i].model))
            print('Avg Uploading Sparsity : %0.4f' % (round(sum(local_sparsity)/len(local_sparsity), 4)))

            # aggregation step
            self.aggregation_models(updated_locals)
            current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            
            # test & log results
            test_loss, test_acc = self.test()
            self.logging(train_loss.item(), test_loss, test_acc, r, exp_id)
            print('==================================================')
            
        return self.container, self.model

    def clients_training(self, sampled_devices, keeped_masks=None, recovery=False, model=None):
        """Local의 training 하나씩 실행함. multiprocessing은 구현하지 않았음."""
        updated_locals = []
        train_loss = 0

        for i in sampled_devices:
            if recovery:
                train_loss += self.locals[i].train_with_recovery(keeped_masks)
                
            else:
                if keeped_masks is not None:    
                    # get and apply pruned mask from global
                    mask_adder(self.locals[i].model, keeped_masks)
                    
                train_loss += self.locals[i].train()
            
                # merge mask of local (remove masks but pruned weights are still zero)
                mask_merger(self.locals[i].model)    
            
            updated_locals.append(self.locals[i].upload_model())

        train_loss /= len(sampled_devices)
        return train_loss, updated_locals
    
    def distribute_models(self, sampled_devices, model, model_name, model_reference):
        for i in sampled_devices:
            self.locals[i].get_model(copy.deepcopy(model), model_name, model_reference)
        # print(f"Devices will be training: {sampled_devices}")

    def aggregation_models(self, updated_locals):
        self.model.load_state_dict(copy.deepcopy(self.aggregate_model_func(updated_locals)))

    def test(self):
        """ Aggregation 되어있는 Global Model로 Test 진행"""
        self.model.eval()
        test_loss, correct, itr = 0, 0, 0
        for itr, (x, y) in enumerate(self.test_loader):
            logprobs = self.model(x.to(self.args.device))
            test_loss += self.loss_func(logprobs, y.to(self.args.device)).item()
            y_pred = torch.argmax(torch.exp(logprobs), dim=1)
            correct += torch.sum(y_pred.view(-1) == y.to(self.args.device).view(-1)).cpu().item()

        self.model.train()
        return test_loss / (itr + 1), 100 * float(correct) / float(self.dataset_test.__len__())

    def logging(self, train_loss, test_loss, test_acc, r, exp_id=None):
        self.container.append([train_loss, test_loss, test_acc, r, exp_id])
        print(f"Train loss: {train_loss:.3f} Test loss: {test_loss:.3f} | acc: {test_acc:.3f}")

