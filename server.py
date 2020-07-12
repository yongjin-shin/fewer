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

        print(model)

    def train(self, exp_id=None):

        global_mask = None  # initialize global mask as None
        for r in range(self.args.nb_rounds):
            print('==================================================')
            print(f'Epoch [{r+1}/{self.args.nb_rounds}]')

            # global pruning step
            if self.args.pruning_type == 'server_pruning':
                self.model, global_mask = self.pruning_handler.pruner(self.model, r)
            
            # distribution step
            current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            print(f'Downloading Sparsity : {current_sparsity:.4f}')

            # Sample Clients
            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            clients_dataset = [self.dataset_locals[i] for i in sampled_devices]

            """
            모델 parameter는 Local이 미리 가지고 있을 필요가 없습니다.
            Training 할때만 한번에 하나씩 가져갑니다. 그래서 memory 사용량은 일정하게 유지됩니다.
            """
            # distribution step
            # self.distribute_models(sampled_devices, self.model)

            # local pruning step
            # client training & upload models
            train_loss, updated_locals = self.clients_training(clients_dataset=clients_dataset,
                                                               keeped_masks=global_mask,
                                                               recovery=self.args.recovery,
                                                               r=r)

            """
            이 부분은 clients_training으로 이동하였습니다.
            더 이상 sampled_devices는 존재하지 않습니다.
            Local을 선택하는 대신에 dataset을(e.g. 200개 중 10개) 선택합니다.
            이 데이터 셋들은 기존에는 Local에 저장되었지만, 이제는 서버가 다 들고 있습니다. 
            이제 Local은 structure 하나만 유지한채로 parameter, dataset을 받습니다.
            따라서 Local이 훈련을 끝내고 나면, sparsity와 pruning을 한번에 처리해야만 합니다.
            """
            # # recovery step
            # local_sparsity = []
            # for i in sampled_devices:
                # _, keeped_local_mask = self.pruning_handler.pruner(self.locals[i].model, r)
                # local_sparsity.append(self.pruning_handler.global_sparsity_evaluator(self.locals.model))
            # print(f'Avg Uploading Sparsity : {round(sum(local_sparsity)/len(local_sparsity), 4):.4f}')

            # aggregation step
            self.aggregation_models(updated_locals)
            
            # global pruning step
            if self.args.pruning_type == 'local_pruning':
                self.model, global_mask = self.pruning_handler.pruner(self.model, r)
                current_sparsity = self.pruning_handler.global_sparsity_evaluator(self.model)
            
            # test & log results
            test_loss, test_acc = self.test()
            self.logger.get_results(Results(train_loss.item(), test_loss, test_acc, current_sparsity*100, r, exp_id))
            print('==================================================')
            
        return self.container, self.model

    def clients_training(self, clients_dataset, r, keeped_masks=None, recovery=False):
        """Local의 training 하나씩 실행함. multiprocessing은 구현하지 않았음."""

        updated_locals, local_sparsity = [], []
        train_loss, _cnt = 0, 0

        for _cnt, dataset in enumerate(clients_dataset):
            self.locals.get_dataset(client_dataset=dataset)
            self.locals.get_model(server_model=model_location_switch_downloading(model=self.model,
                                                                                 args=self.args))

            if recovery:
                raise RuntimeError("We Dont need recovery step anymore!!!")
                # train_loss += self.locals.train_with_recovery(mask_location_switch(keeped_masks, self.args.device))
                
            else:
                if keeped_masks is not None:    
                    # get and apply pruned mask from global
                    mask_adder(self.locals.model, mask_location_switch(keeped_masks, self.args.device))
                    
                train_loss += self.locals.train()
            
                # merge mask of local (remove masks but pruned weights are still zero)
                mask_merger(self.locals.model)

            """ Pruning """
            _, keeped_local_mask = self.pruning_handler.pruner(self.locals.model, r)

            """ Sparsity """
            local_sparsity.append(self.pruning_handler.global_sparsity_evaluator(self.locals.model))

            """ Uploading """
            updated_locals.append(self.locals.upload_model())
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

        self.model.to(self.args.server_location).train()
        return test_loss / (itr + 1), 100 * float(correct) / float(self.len_test_data)

    def get_global_model(self):
        return self.model


