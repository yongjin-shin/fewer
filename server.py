from torch.utils.data import DataLoader
import numpy as np
import copy
import torch

# Related Classes
from local import Local
from aggregation import get_aggregation_func
from pruning import *
from networks import create_nets
from misc import model_location_switch_downloading, mask_location_switch
from logger import Results
import gc

# from pynvml import *


class Server:
    def __init__(self, args, logger):

        # important variables
        self.args = args
        self.locals = Local(args=args)
        self.logger = logger
        
        # pruning handler
        self.pruning_handler = PruningHandler(args)
        self.sparsity = 0
        
        # dataset
        self.dataset_train, self.dataset_locals = None, None
        self.len_test_data = 0
        self.test_loader = None

        # about model
        self.model = None
        self.make_model()

        # about optimization
        self.loss_func = torch.nn.NLLLoss(reduction='mean')
        self.aggregate_model_func = get_aggregation_func(self.args.aggregation_alg)

        # misc
        self.container = []
        self.nb_unique_label = 0
        self.nb_client_per_round = max(int(self.args.ratio_clients_per_round * self.args.nb_devices), 1)
        self.sampling_clients = lambda nb_samples: np.random.choice(self.args.nb_devices, nb_samples, replace=False)  
        
    def get_data(self, dataset_server, dataset_locals, dataset_test):
        """raw data를 받아와서 server와 local에 데이터를 분배함"""
        self.dataset_train, self.dataset_locals = dataset_server, dataset_locals
        self.test_loader = DataLoader(dataset_test, batch_size=100, shuffle=True)
        self.len_test_data = dataset_test.__len__()
        # for i in range(self.args.nb_devices):
        #     self.locals[i].get_dataset(dataset_locals[i])

    def make_model(self):
        model = create_nets(self.args, 'SERVER')

        if self.args.server_location == 'gpu':
            if self.args.gpu:
                self.model = model.to(self.args.device)
            else:
                raise RuntimeError
        elif self.args.server_location == 'cpu':
            self.model = model
        else:
            raise NotImplementedError

        print(model)

    def train(self, exp_id=None):
        """Distribute, Train, Aggregation and Test"""
        for r in range(self.args.nb_rounds):
            print('==================================================')
            print(f'Epoch [{r+1}/{self.args.nb_rounds}]')

            # global pruning step
            self.model, keeped_masks = self.pruning_handler.pruner(self.model, r)
            current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            print(f'Downloading Sparsity : {current_sparsity:.4f}')

            # Sample Clients
            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            clients_dataset = [self.dataset_locals[i] for i in sampled_devices]

            # distribution step
            # self.distribute_models(sampled_devices, self.model)
        
            # client training
            train_loss, updated_locals = self.clients_training(clients_dataset=clients_dataset,
                                                               keeped_masks=keeped_masks,
                                                               recovery=self.args.recovery)
            
            # # recovery step
            local_sparsity = []
            for i in sampled_devices:
                # _, keeped_local_mask = self.pruning_handler.pruner(self.locals[i].model, r)
                local_sparsity.append(self.pruning_handler.global_sparsity_evaluator(self.locals.model))
            print(f'Avg Uploading Sparsity : {round(sum(local_sparsity)/len(local_sparsity), 4):.4f}')

            # aggregation step
            self.aggregation_models(updated_locals)
            # current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            
            # test & log results
            test_loss, test_acc = self.test()
            self.logger.get_results(Results(train_loss.item(), test_loss, test_acc, current_sparsity*100, r, exp_id))
            print('==================================================')
            
        return self.container, self.model

    def clients_training(self, clients_dataset, keeped_masks=None, recovery=False):
        """Local의 training 하나씩 실행함. multiprocessing은 구현하지 않았음."""

        updated_locals = []
        train_loss, _cnt = 0, 0

        for _cnt, dataset in enumerate(clients_dataset):
            self.locals.get_dataset(client_dataset=dataset)
            self.locals.get_model(server_model=model_location_switch_downloading(model=self.model,
                                                                                 args=self.args))

            if recovery:
                train_loss += self.locals.train_with_recovery(mask_location_switch(keeped_masks, self.args.device))
                
            else:
                if keeped_masks is not None:    
                    # get and apply pruned mask from global
                    mask_adder(self.locals.model, mask_location_switch(keeped_masks, self.args.device))
                    
                train_loss += self.locals.train()
            
                # merge mask of local (remove masks but pruned weights are still zero)
                mask_merger(self.locals.model)
            
            updated_locals.append(self.locals.upload_model())

            self.locals.reset()

        train_loss /= (_cnt+1)
        return train_loss, updated_locals
    
    # def distribute_models(self, sampled_devices, model):
    #     for i in sampled_devices:
    #         self.locals[i].get_model(copy.deepcopy(model))
        # print(f"Devices will be training: {sampled_devices}")

    def aggregation_models(self, updated_locals):
        self.model.load_state_dict(copy.deepcopy(self.aggregate_model_func(updated_locals)))
        gc.collect()
        torch.cuda.empty_cache()

    def test(self):
        """ Aggregation 되어있는 Global Model로 Test 진행"""
        self.model.to(self.args.device).eval()
        test_loss, correct, itr = 0, 0, 0
        for itr, (x, y) in enumerate(self.test_loader):
            logprobs = self.model(x.to(self.args.device))
            test_loss += self.loss_func(logprobs, y.to(self.args.device)).item()
            y_pred = torch.argmax(torch.exp(logprobs), dim=1)
            correct += torch.sum(y_pred.view(-1) == y.to(self.args.device).view(-1)).cpu().item()

        self.model.to(self.args.server_location).train()
        return test_loss / (itr + 1), 100 * float(correct) / float(self.len_test_data)

    def get_global_model(self):
        return self.model


