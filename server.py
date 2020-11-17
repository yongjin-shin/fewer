import numpy as np
import pandas as pd
import copy, gc, time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.tot_comm_cost = 0

        # dataset
        self.dataset_locals = None
        self.len_test_data = 0
        self.test_loader = None
        self.valid_loader = None
        self.true_test_target = []

        # about optimization
        self.criterion = torch.nn.CrossEntropyLoss()
        self.aggregate_model_func = get_aggregation_func(self.args.aggregation_alg)
        self.server_optim = None
        self.server_lr_scheduler = None

        # about model
        self.layers_name = None
        self.model = None
        self.dummy_model = None
        self.init_cost = 0
        self.make_model()
        self.make_opt()

        # misc
        self.container = []
        self.nb_unique_label = 0
        self.nb_client_per_round = max(int(self.args.ratio_clients_per_round * self.args.nb_devices), 1)
        self.sampling_clients = lambda nb_samples: np.random.choice(self.args.nb_devices, nb_samples, replace=False)
        self.betas = np.array([0.1, 0.3, 0.5, 0.7, 0.9] if self.args.beta_validation \
            else [self.args.beta])

    def train(self, exp_id=None):
        self.locals.get_public_data(self.valid_loader)
        
        for fed_round in range(self.args.nb_rounds):
            start_time = time.time()
            print('==================================================')
            print(f'Epoch [{exp_id}: {fed_round+1}/{self.args.nb_rounds}]')

            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            clients_dataset = [self.dataset_locals[i] for i in sampled_devices]

            # Beta를 바꿔가면서 동일한 local에 대해서 training을 진행함
            # Aggregation을 진행한 다음 global로 바로 업데이트 하지않고 일단 HDD에 저장을 함
            # 모델 저장은 beta값마다 하나씩 됨: path/exp_id/model_{beta}.h5 으로 저장됨.
            local_results = None
            for beta in self.betas:
                local_results = self.clients_training(clients_dataset=clients_dataset,
                                                      beta=beta)
                self.aggregation_models(local_results['updated_locals'],
                                        local_results['len_datasets'],
                                        save_model=True,
                                        logger=self.logger,
                                        exp_id=exp_id,
                                        description=str(beta))

            # Layer Variance
            layer_val = get_variance(self.layers_name, self.model.state_dict(),
                                     local_results['updated_locals'], self.args.server_location)
            print(layer_val)

            # 저장된 N개의 Aggregated model은 validation set에서 성능을 비교함
            # beta에 따라 다른 모델들 중 가장 좋은 성능을 보인 모델을 찾아냄
            valid_results, best_beta = self.valid(self.logger,
                                                  exp_id=exp_id)

            # best score를 지닌 모델을 불러옴
            model = self.logger.load_model(exp_id=exp_id,
                                           description=best_beta)

            # global model로 업데이트를 시킨 이후에
            # Test를 진행함!
            self.load_model(model)
            test_results = self.test()
            # test_results = self.test_agg_vs_ensemble(local_results['updated_locals'])

            # Layer Variance
            layer_val = get_variance(self.layers_name, self.model.state_dict(),
                                     local_results['updated_locals'], self.args.server_location)
            print(layer_val)
            print(local_results['grad_norm'])
            print(local_results['weight_norm'])
            
            # Tubulance Motivation Test
            noise_results = None
            if self.args.noise_interval < self.args.nb_rounds and fed_round % self.args.noise_interval == 0:
                noise_results = get_tubulanced_results(self.args, self.server_lr_scheduler.get_last_lr()[0] * 10, 
                                                       self.layers_name, model, self.dummy_model, 
                                                       self.valid_loader, self.criterion, local_results['grad_norm'],
                                                       n_noise=100, dist_list=[0.1, 0.5, 1, 2, 3, 4])
            
            # Swapping Motivation Test
            if self.args.swapping_interval < self.args.nb_rounds and fed_round % self.args.swapping_interval == 0:
                self.swapping_head(fed_round, self.logger, local_results['updated_locals'], model, clients_dataset)

            self.server_lr_scheduler.step()
            end_time = time.time()
            ellapsed_time = end_time - start_time
            self.logger.get_results(Results(local_results['loss'], test_results['loss'], test_results['acc'],
                                            0, self.tot_comm_cost,
                                            fed_round, exp_id, ellapsed_time,
                                            self.server_lr_scheduler.get_last_lr()[0],
                                            best_beta, valid_results['acc'], layer_val,
                                            local_results['grad_norm'], local_results['weight_norm'], noise_results))
            # np.mean(local_results['kld']), test_results['kld'], test_results['ensemble_acc']))

            gc.collect()
            torch.cuda.empty_cache()
            if fed_round == self.args.start_freezing:
                self.locals.add_slower_layer(slow_layer=[4],
                                             slow_ratio=0)

        return self.container, self.model

    def clients_training(self, clients_dataset, beta):
        updated_locals, train_acc, local_kld, local_grad_norm, local_weight_norm = [], [], [], [], []
        train_loss, _cnt = 0, 0
        len_datasets = []

        for _cnt, dataset in enumerate(clients_dataset):
            # distribute local dataset
            self.locals.get_dataset(client_dataset=dataset)
            self.locals.get_model(server_model=self.model.state_dict(),
                                  layers_name=self.layers_name)
            self.locals.get_lr(server_lr=self.server_lr_scheduler.get_last_lr()[0])

            # train local
            local_results = self.locals.train(beta=beta)
            print(local_results)
            train_loss += local_results['loss']
            train_acc.append(local_results['acc'])
            local_kld.append(local_results['kld'])
            local_grad_norm.append(local_results['norm'])

            # uploads local
            updated_locals.append(self.locals.upload_model())
            len_datasets.append(dataset.__len__())
            local_weight_norm.append(calc_l2_norm(self.layers_name, updated_locals[-1],
                                                  self.args.server_location))
            
            # reset local model
            self.locals.reset()

        train_loss /= (_cnt+1)
        with np.printoptions(precision=2, suppress=True):
            print(f'{beta} Local Train Acc :', np.array(train_acc)/100)
            # print(f'Local Train KLD :', np.array(local_kld))

        ret = {
            'loss': train_loss,
            'kld': np.mean(local_kld),
            'len_datasets': len_datasets,
            'updated_locals': updated_locals,
            'grad_norm': dict(pd.DataFrame.from_dict(local_grad_norm).mean().round(3)),
            'weight_norm': dict(pd.DataFrame.from_dict(local_weight_norm).mean().round(3))
        }
        return ret

    def aggregation_models(self, updated_locals, len_datasets, save_model=False, logger=None, exp_id=None,
                           description=None):
        """
        - Aggregate를 하는데, 기존과 다른 점은 global 모델로 바로 반영하지 않고, 저장함!
        - self.load_model에서 global model을 반영하도록 변경
        """
        aggregated_model = self.aggregate_model_func(updated_locals, len_datasets)
        if save_model:
            logger.save_model(param=aggregated_model, exp_id=exp_id, description=description)

        return

    def load_model(self, aggregated_model):
        """
        여기서 global model에 반영함
        """
        self.model.load_state_dict(copy.deepcopy(aggregated_model))

    def valid(self, logger, exp_id):
        """
        1. 각 beta에 대해서 저장된 모델을 load
        2. validation set에 대해서 acc 산출
        3. 가장 큰 스코어에 대한 beta와 acc를 리턴
        """
        acc = []

        for beta in self.betas:
            model = logger.load_model(exp_id=exp_id, description=beta)
            self.load_model(model)
            ret = get_test_results(self.args, self.model, self.valid_loader, self.criterion,
                                   return_loss=False, return_acc=True, return_logit=False)
            acc.append(ret['acc'])
        acc = np.array(acc)
        best_idx = np.argmax(acc)

        best_score = acc[best_idx]
        best_beta = self.betas[best_idx]
        return {'acc': best_score}, best_beta

    def test(self):
        ret = get_test_results(self.args, self.model, self.test_loader, self.criterion,
                               return_loss=True, return_acc=True, return_logit=False)
        return ret

    def test_agg_vs_ensemble(self, locals):
        global_ret = get_test_results(self.args, self.model, self.test_loader, self.criterion,
                                      return_loss=True, return_acc=True, return_logit=True)
        global_logits = global_ret['logits']

        local_logits = []
        for local_model in locals:
            self.dummy_model.load_state_dict(copy.deepcopy(local_model))
            ret = get_test_results(self.args, self.dummy_model, self.test_loader, self.criterion,
                                   return_loss=False, return_acc=False, return_logit=True)
            local_logits.append(ret['logits'])
        local_logits = np.dstack(local_logits)
        major_logits = []
        ensemble_acc = 0
        for i in range(len(global_logits)):
            # vote = np.argmax(local_logits[i], axis=0)
            # major = np.argmax(np.bincount(vote))
            # if major == self.true_test_target[i]:
            #     ensemble_acc += 1
            #
            # major_idx = np.where(vote == major)[0]
            # voted_logits = local_logits[i, :, major_idx]
            # mean_logits = np.mean(voted_logits, axis=0)
            # major_logits.append(mean_logits)

            mean_logits = np.mean(local_logits[i], axis=1)
            major = np.argmax(mean_logits)
            if major == self.true_test_target[i]:
                ensemble_acc += 1
            major_logits.append(mean_logits)

        major_logits = np.vstack(major_logits)
        ensemble_acc = ensemble_acc / len(global_logits) * 100

        #KL(True||Est) = KL(Ensemble||Aggregate)
        kl = compute_js_divergence(major_logits, global_logits)

        ret = {
            'loss': global_ret['loss'],
            'acc': global_ret['acc'],
            'kld': kl,
            'ensemble_acc': ensemble_acc
        }
        return ret

    def swapping_head(self, e, logger, locals_params, global_params, clients_dataset):
        locals_params.append(global_params)
        locals_params.append(torch.load(f"./log/[cifarcnn-cifar10]oracle/0/model.h5"))

        len_locals = len(locals_params)
        client_dataset_results = np.empty(shape=(len_locals, len_locals))
        valid_dataset_results = np.empty_like(client_dataset_results)
        local_unique_data = []
        
        for row, body_params in enumerate(locals_params):
            if row < len_locals-2:
                local_data = clients_dataset[row]
                d_loader = DataLoader(local_data, batch_size=100, shuffle=False)
            else:
                d_loader = self.valid_loader
            
            for col, head_params in enumerate(locals_params):
                params = copy.deepcopy(body_params)
                params[f"{self.layers_name[-1]}.bias"] = head_params[f"{self.layers_name[-1]}.bias"]
                params[f"{self.layers_name[-1]}.weight"] = head_params[f"{self.layers_name[-1]}.weight"]
                self.dummy_model.load_state_dict(params)
                    
                client_acc = get_test_results(args=self.args, model=self.dummy_model, 
                                              dataloader=d_loader, criterion=None,
                                              return_loss=False, return_acc=True, return_logit=False,
                                              return_unique_labels=True)
                client_dataset_results[row, col] = client_acc['acc']
                if col == 0:
                    local_unique_data.append(client_acc['label'])
                
                valid_acc = get_test_results(self.args, self.dummy_model, self.valid_loader, None,
                                            return_loss=False, return_acc=True, return_logit=False)
                valid_dataset_results[row, col] = valid_acc['acc']

        # client_dataset_results = np.abs(client_dataset_results - client_dataset_results.diagonal())
        # valid_dataset_results = np.abs(valid_dataset_results - valid_dataset_results.diagonal())
        
        # draw heatmap
        fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax1.get_shared_y_axes().join(ax2)
        g1 = sns.heatmap(client_dataset_results, annot=True, fmt='.1f', annot_kws={"fontsize":8}, linewidth=0.5, cmap="YlGnBu", cbar=False, ax=ax1)
        g2 = sns.heatmap(valid_dataset_results, annot=True, fmt='.1f', annot_kws={"fontsize":8}, linewidth=0.5, cmap="YlGnBu", cbar=False, ax=ax2)
        
        g1.set_title('Local Dataset')
        # g1.set_xticks([])
        # g1.set_yticks([])
        
        ticks = [f"{unique_labels}" for unique_labels in local_unique_data]
        ticks[-2] = 'Agrgt'
        ticks[-1] = 'Oracle'
        g1.set_yticklabels(tuple(ticks), rotation=0)
        g1.set_xticklabels(tuple(ticks), rotation=45)

        g2.set_title('Valid Dataset')
        # g2.set_xticks([])
        g2.set_yticks([])
        g2.set_xticklabels(tuple(ticks), rotation=45)

        fig.suptitle(f"{e}th Round", fontsize=15)
        fig.text(0.04, 0.5, 'Clients\' Body', va='center', rotation='vertical')
        fig.text(0.5, 0.01, 'Clients\' Head', ha='center')
        # plt.savefig('test.png')
        logger.save_plot(fig, f"{e}th_round")
        plt.close()
        
        return

    def get_data(self, dataset_valid, dataset_locals, dataset_test):
        self.dataset_locals = dataset_locals

        # Validation 데이터 불러오기
        self.valid_loader = DataLoader(dataset_valid, batch_size=100, shuffle=True)
        self.test_loader = DataLoader(dataset_test, batch_size=100, shuffle=True)

        # 각 데이터 target에 대해서 bin count 프린
        aa = []
        for i, (x, y) in enumerate(self.valid_loader):
            aa.append(y)
        print('Validation Set: ', np.bincount(np.concatenate(aa)))

        self.len_test_data = dataset_test.__len__()
        for i, (x, y) in enumerate(self.test_loader):
            self.true_test_target.append(y)
        self.true_test_target = np.concatenate(self.true_test_target)
        print('Test Set: ', np.bincount(self.true_test_target))

        return

    def make_model(self):
        model = create_nets(self.args, 'SERVER')
        dummy_model = create_nets(self.args, 'DUMMY')
        print(model)
        self.model = model.to(self.args.server_location)
        self.dummy_model = dummy_model.to(self.args.server_location)
        self.init_cost = get_size(self.model.parameters())

        self.layers_name = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                self.layers_name.append(name)
        self.layers_name = np.array(self.layers_name)

        # original_params = self.model.state_dict()
        # oracle_params = torch.load(f"./log/[cifarcnn-cifar10]oracle/0/model.h5")
        # original_params['fc3.weight'] = copy.deepcopy(oracle_params['fc3.weight'])
        # original_params['fc3.bias'] = copy.deepcopy(oracle_params['fc3.bias'])
        # self.model.load_state_dict(original_params)
        
        # print("Copied Oracle Head")
        return
        
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
        elif 'step' == self.args.scheduler:
            self.server_lr_scheduler = LinearStepLR(optimizer=self.server_optim,
                                                    init_lr=self.args.lr,
                                                    epoch=self.args.nb_rounds,
                                                    eta_min=5e-6,
                                                    decay_rate=0.5)
        elif 'constant' == self.args.scheduler:
            self.server_lr_scheduler = ConstantLR(self.args.lr)

        elif 'cosine_warmup' == self.args.scheduler:
            self.server_lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=self.server_optim,
                T_0=60,
                T_mult=1,
                eta_min=1e-3
            )
        else:
            raise NotImplementedError

    def get_global_model(self):
        return self.model
