import numpy as np
import copy, gc, time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# import related custom packages
from local import *
from train_tools import *
from utils import *

__all__ = ['Server']


class Server:
    def __init__(self, args, logger):

        # important variables
        self.args, self.logger = args, logger
        self.locals = Local(args=args)
        
        # sparsity handler
        self.sparsity_handler = SparsityHandler(args)
        self.sparsity = 0
        self.tot_comm_cost = 0
        
        # dataset
        self.dataset_train, self.dataset_locals = None, None
        self.len_test_data = 0
        self.test_loader = None

        # about optimization
        self.criterion = torch.nn.CrossEntropyLoss()
        self.aggregate_model_func = get_aggregation_func(self.args.aggregation_alg)
        self.server_optim = None
        self.server_lr_scheduler = None

        # about model
        self.model = None
        self.init_cost = 0
        self.make_model()
        self.make_opt()

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
        print(model)
        self.model = model.to(self.args.server_location)
        self.init_cost = get_size(self.model.parameters())

    def make_opt(self):
        self.server_optim = torch.optim.SGD(self.model.parameters(),
                                            lr=self.args.lr,
                                            momentum=self.args.momentum,
                                            weight_decay=self.args.weight_decay)
        
        if 'cosine' == self.args.scheduler:
            self.server_lr_scheduler = CosineAnnealingLR(self.server_optim,
                                                         self.args.nb_rounds,
                                                         eta_min=5e-6,
                                                         last_epoch=-1)
        elif 'linear' == self.args.scheduler:
            self.server_lr_scheduler = LinearLR(self.args.lr,
                                                self.args.nb_rounds,
                                                eta_min=5e-6)
        elif 'constant' == self.args.scheduler:
            self.server_lr_scheduler = ConstantLR(self.args.lr)
            
        elif 'step' == self.args.scheduler:
            self.server_lr_scheduler = LinearStepLR(optimizer=self.server_optim,
                                                    init_lr=self.args.lr,
                                                    epoch=self.args.nb_rounds,
                                                    eta_min=5e-6,
                                                    decay_rate=0.5)
        else:
            raise NotImplementedError

    def train(self, exp_id=None):
        
        # initialize global mask & recovery signals as None
        global_mask, recovery_signals = None, []  

        for fed_round in range(self.args.nb_rounds):
            start_time = time.time()
            print('==================================================')
            print(f'Epoch [{exp_id}: {fed_round+1}/{self.args.nb_rounds}]', end='')

            # global pruning step
            if self.args.pruning_type == 'server_pruning':
                self.model, global_mask = self.sparsity_handler.round_sparsifier(self.model, 
                                                                                 fed_round, 
                                                                                 recovery_signals,
                                                                                 global_mask,
                                                                                 merge=True)

            # distribution step
            current_sparsity = sparsity_evaluator(self.model)
            print(f' Down Spars : {current_sparsity:.3f}', end=' ')
            self.tot_comm_cost += self.init_cost * (1-current_sparsity) * self.nb_client_per_round

            # Sample Clients
            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            clients_dataset = [self.dataset_locals[i] for i in sampled_devices]

            # local pruning step (client training & upload models)
            train_loss, updated_locals, len_datasets, recovery_signals = self.clients_training(
                clients_dataset=clients_dataset, keeped_masks=global_mask, use_recovery_signal=self.args.use_recovery_signal)
            
            # model_variance = get_models_variance(self.model.state_dict(), updated_locals, self.args.device)
            model_variance = 0

            self.server_lr_scheduler.step()

            # aggregation step
            self.aggregation_models(updated_locals, len_datasets)
            gc.collect()
            torch.cuda.empty_cache()
            
            # test & log results
            test_loss, test_acc = self.test()
            end_time = time.time()
            ellapsed_time = end_time - start_time
            self.logger.get_results(Results(train_loss, test_loss, test_acc, 
                                            current_sparsity*100, self.tot_comm_cost, 
                                            fed_round, exp_id, ellapsed_time, 
                                            self.server_lr_scheduler.get_last_lr()[0],
                                            model_variance))

        return self.container, self.model

    def clients_training(self, clients_dataset, keeped_masks=None, use_recovery_signal=False):
    
        updated_locals, local_sparsity, recovery_signals, train_acc = [], [], [], []
        train_loss, _cnt = 0, 0
        len_datasets = []

        for _cnt, dataset in enumerate(clients_dataset):
            self.locals.get_dataset(client_dataset=dataset)
            self.locals.get_model(server_model=self.model.state_dict())
            self.locals.get_lr(server_lr=self.server_lr_scheduler.get_last_lr()[0])

            if keeped_masks is not None:    
                # get and apply pruned mask from global
                mask_adder(self.locals.model, keeped_masks)
                local_loss, local_acc = self.locals.train()
                train_loss += local_loss
                train_acc.append(local_acc)
                mask_merger(self.locals.model)                
                
            else:
                local_loss, local_acc = self.locals.train()
                train_loss += local_loss
                train_acc.append(local_acc)
                
            """ Sparsity """
            local_sparsity.append(sparsity_evaluator(self.locals.model))
            
            """ Uploading """
            self.tot_comm_cost += self.init_cost * (1 - local_sparsity[-1])
            updated_locals.append(self.locals.upload_model())
            len_datasets.append(dataset.__len__())
            
            """ Recovery Signal """
            if self.args.use_recovery_signal:
                local_recovery_signal = self.sparsity_handler.get_local_signal(self.locals,
                                                                               keeped_masks,
                                                                               topk=self.args.local_topk,
                                                                               as_mask=self.args.signal_as_mask)
                recovery_signals.append(local_recovery_signal)
            
            # reset local model
            self.locals.reset()
        
        print(f'Avg Up Spars : {round(sum(local_sparsity)/len(local_sparsity), 4):.3f}\n')
        train_loss /= (_cnt+1)
        print(f'Local Train Acc : {train_acc}\n')

        return train_loss, updated_locals, len_datasets, recovery_signals

    def aggregation_models(self, updated_locals, len_datasets):
        self.model.load_state_dict(copy.deepcopy(self.aggregate_model_func(updated_locals, len_datasets)))

    def test(self):
        self.model.to(self.args.device).eval()

        test_loss, correct, itr = 0, 0, 0
        for itr, (data, target) in enumerate(self.test_loader):
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            output = self.model(data)
            test_loss += self.criterion(output, target).item()
            y_pred = torch.max(output, dim=1)[1]
            correct += torch.sum(y_pred.view(-1) == target.to(self.args.device).view(-1)).cpu().item()

        self.model.to(self.args.server_location).train()
        return test_loss / (itr + 1), 100 * float(correct) / float(self.len_test_data)

    def get_global_model(self):
        return self.model
