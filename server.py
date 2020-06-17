from torch.utils.data import DataLoader
import numpy as np
import copy
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# Related Classes
from local import Local
from aggregation import get_aggregation_func
from pruning import *
from networks import create_nets
from misc import model_location_switch_downloading, mask_location_switch, get_size
from logger import Results
import gc
import time


class Server:
    def __init__(self, args, logger):

        # important variables
        self.args = args
        self.locals = Local(args=args)
        self.logger = logger
        
        # pruning handler
        self.pruning_handler = PruningHandler(args)
        self.sparsity = 0
        self.tot_comm_cost = 0
        
        # dataset
        self.dataset_train, self.dataset_locals = None, None
        self.len_test_data = 0
        self.test_loader = None

        # about optimization
        self.loss_func = torch.nn.NLLLoss(reduction='mean')
        self.aggregate_model_func = get_aggregation_func(self.args.aggregation_alg)
        self.server_optim = None
        self.server_lr_scheduler = None

        # about model
        self.model = None
        self.init_cost = 0
        self.make_model()

        # misc
        self.container = []
        self.nb_unique_label = 0
        self.nb_client_per_round = max(int(self.args.ratio_clients_per_round * self.args.nb_devices), 1)
        self.sampling_clients = lambda nb_samples: np.random.choice(self.args.nb_devices, nb_samples, replace=False)  
        
    def get_data(self, dataset_server, dataset_locals, dataset_test):
        self.dataset_train, self.dataset_locals = dataset_server, dataset_locals
        self.test_loader = DataLoader(dataset_test, batch_size=100, shuffle=True)
        self.len_test_data = dataset_test.__len__()

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

        self.init_cost = get_size(self.model.parameters())

        self.server_optim = torch.optim.SGD(self.model.parameters(),
                                            lr=self.args.lr,
                                            momentum=self.args.momentum,
                                            weight_decay=self.args.weight_decay)
        self.server_lr_scheduler = CosineAnnealingLR(self.server_optim,
                                                     self.args.nb_rounds * self.args.local_ep,
                                                     eta_min=5e-6,
                                                     last_epoch=-1)
        print(model)

    def train(self, exp_id=None):

        global_mask = None  # initialize global mask as None

        for r in range(self.args.nb_rounds):
            start_time = time.time()
            print('==================================================')
            print(f'Epoch [{r+1}/{self.args.nb_rounds}]')

            # global pruning step
            if self.args.pruning_type == 'server_pruning':
                self.model, global_mask = self.pruning_handler.pruner(self.model, r)
                mask_merger(self.model)
                
            # distribution step
            current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            print(f'Downloading Sparsity : {current_sparsity:.4f}')
            self.tot_comm_cost += self.init_cost * (1-current_sparsity) * self.nb_client_per_round

            # Sample Clients
            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            clients_dataset = [self.dataset_locals[i] for i in sampled_devices]

            # local pruning step
            # client training & upload models
            train_loss, updated_locals = self.clients_training(clients_dataset=clients_dataset,
                                                               keeped_masks=global_mask,
                                                               recovery=self.args.recovery,
                                                               r=r)

            # aggregation step
            self.aggregation_models(updated_locals)
            
            # global pruning step
            if self.args.pruning_type in ['local_pruning', 'local_pruning_half']:
                self.model, global_mask = self.pruning_handler.pruner(self.model, r)
                mask_merger(self.model)
                current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            
            # test & log results
            test_loss, test_acc = self.test()
            end_time = time.time()
            ellapsed_time = end_time - start_time
            self.logger.get_results(Results(train_loss.item(), test_loss, test_acc, current_sparsity*100, self.tot_comm_cost, r, exp_id,
                                            ellapsed_time, self.server_lr_scheduler.get_last_lr()[0]))
            print('==================================================')
            
        return self.container, self.model

    def clients_training(self, clients_dataset, r, keeped_masks=None, recovery=False):
        updated_locals, local_sparsity = [], []
        train_loss, _cnt = 0, 0

        for _cnt, dataset in enumerate(clients_dataset):
            self.locals.get_dataset(client_dataset=dataset)
            self.locals.get_model(server_model=model_location_switch_downloading(model=self.model,
                                                                                 args=self.args))
            self.locals.get_optim(server_optim=copy.deepcopy(self.server_optim.state_dict()),
                                  server_scheduler=copy.deepcopy(self.server_lr_scheduler.state_dict()))

            if recovery:
                raise RuntimeError("We Dont need recovery step anymore!!!")
                # train_loss += self.locals.train_with_recovery(mask_location_switch(keeped_masks, self.args.device))
                
            else:
                if keeped_masks is not None:    
                    # get and apply pruned mask from global
                    mask_adder(self.locals.model, mask_location_switch(keeped_masks, self.args.device))
                
                if self.args.pruning_type == 'server_pruning':
                    train_loss += self.locals.train()
                    mask_merger(self.locals.model)
                
                elif self.args.pruning_type == 'local_pruning':
                    train_loss += self.locals.train()
                    mask_merger(self.locals.model)
                    _, keeped_local_mask = self.pruning_handler.pruner(self.locals.model, r)
                    mask_merger(self.locals.model)

                elif self.args.pruning_type == 'local_pruning_half':
                    # run half epochs with initial mask
                    self.locals.adapt_half_epochs('half1')
                    train_loss += self.locals.train()
                    
                    # local pruning step 
                    mask_merger(self.locals.model)
                    _, keeped_local_mask = self.pruning_handler.pruner(self.locals.model, r)
                    mask_adder(self.locals.model, mask_location_switch(keeped_local_mask, self.args.device))
                    
                    # run remaining half epochs
                    self.locals.adapt_half_epochs('half2')
                    train_loss += self.locals.train()
                    
                    # merge mask of local (remove masks but pruned weights are still zero)
                    mask_merger(self.locals.model)
                
                else:
                    train_loss += self.locals.train()

            """ Sparsity """
            local_sparsity.append(self.pruning_handler.global_sparsity_evaluator(self.locals.model))

            """ Uploading """
            self.tot_comm_cost += self.init_cost * (1 - local_sparsity[-1])
            updated_locals.append(self.locals.upload_model())

            if _cnt+1 == self.nb_client_per_round:
                local_optim, local_scheduler = self.locals.upload_optim()
                # self.server_optim.load_state_dict(local_optim)
                self.server_lr_scheduler.load_state_dict(local_scheduler)

            self.locals.reset()

        train_loss /= (_cnt+1)
        print(f'Avg Uploading Sparsity : {round(sum(local_sparsity)/len(local_sparsity), 4):.4f}')

        return train_loss, updated_locals

    def aggregation_models(self, updated_locals):
        self.model.load_state_dict(copy.deepcopy(self.aggregate_model_func(updated_locals)))
        gc.collect()
        torch.cuda.empty_cache()

    def test(self):
        self.model.to(self.args.device).eval()
        test_loss, correct, itr = 0, 0, 0
        for itr, (x, y) in enumerate(self.test_loader):
            logprobs = self.model(x.to(self.args.device))
            test_loss += self.loss_func(logprobs, y.to(self.args.device)).item()
            y_pred = torch.argmax(torch.exp(logprobs), dim=1)
            correct += torch.sum(y_pred.view(-1) == y.to(self.args.device).view(-1)).cpu().item()

        self.model.to(self.args.device).train()
        return test_loss / (itr + 1), 100 * float(correct) / float(self.len_test_data)

    def get_global_model(self):
        return self.model



