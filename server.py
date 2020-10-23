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
            # layer_val = get_variance(self.layers_name, self.model.state_dict(),
            #                          local_results['updated_locals'], self.args.server_location)
            # print(layer_val)

            # 저장된 N개의 Aggregated model은 validation set에서 성능을 비교함
            # beta에 따라 다른 모델들 중 가장 좋은 성능을 보인 모델을 찾아냄
            valid_results, best_beta = self.valid(data_loader=self.valid_loader,
                                                  logger=self.logger,
                                                  exp_id=exp_id)

            # best score를 지닌 모델을 불러옴
            model = self.logger.load_model(exp_id=exp_id,
                                           description=best_beta)

            # global model로 업데이트를 시킨 이후에
            # Test를 진행함!
            self.load_model(model)
            # test_results = self.test()
            test_results = self.test_agg_vs_ensemble(local_results['updated_locals'])
            self.server_lr_scheduler.step()

            # Layer Variance
            # layer_val = get_variance(self.layers_name, self.model.state_dict(),
            #                          local_results['updated_locals'], self.args.server_location)
            # print(layer_val)

            end_time = time.time()
            ellapsed_time = end_time - start_time
            self.logger.get_results(Results(local_results['loss'], test_results['loss'], test_results['acc'],
                                            0, self.tot_comm_cost,
                                            fed_round, exp_id, ellapsed_time,
                                            self.server_lr_scheduler.get_last_lr()[0],
                                            0, test_results['ensemble_acc'], 0))
                                            # best_beta, valid_results['acc'], layer_val))
            # np.mean(local_results['kld']), test_results['kld'], test_results['ensemble_acc']))

            gc.collect()
            torch.cuda.empty_cache()

        return self.container, self.model

    def clients_training(self, clients_dataset, beta):
        updated_locals, train_acc, local_kld = [], [], []
        train_loss, _cnt = 0, 0
        len_datasets = []

        for _cnt, dataset in enumerate(clients_dataset):
            # distribute local dataset
            self.locals.get_dataset(client_dataset=dataset)
            self.locals.get_model(server_model=self.model.state_dict())
            self.locals.get_lr(server_lr=self.server_lr_scheduler.get_last_lr()[0])

            # train local
            local_results = self.locals.train(beta=beta)
            train_loss += local_results['loss']
            train_acc.append(local_results['acc'])
            local_kld.append(local_results['kld'])

            # uploads local
            updated_locals.append(self.locals.upload_model())
            len_datasets.append(dataset.__len__())
            
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
            'updated_locals': updated_locals
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

    def valid(self, data_loader, logger, exp_id):
        """
        1. 각 beta에 대해서 저장된 모델을 load
        2. validation set에 대해서 acc 산출
        3. 가장 큰 스코어에 대한 beta와 acc를 리턴
        """
        acc = []

        for beta in self.betas:
            model = logger.load_model(exp_id=exp_id, description=beta)
            self.load_model(model)
            ret = get_test_results(self.args, self.model, data_loader, self.criterion,
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
        for l, local_model in enumerate(locals):
            self.dummy_model.load_state_dict(copy.deepcopy(local_model))
            ret = get_test_results(self.args, self.dummy_model, self.test_loader, self.criterion,
                                   return_loss=False, return_acc=True, return_logit=True)
            print(f"{l}th local ACC: {ret['acc']}")
            local_logits.append(ret['logits'])
        local_logits = np.dstack(local_logits)
        major_logits = []
        ensemble_acc_vote = 0
        ensemble_acc_mean = 0
        for i in range(len(global_logits)):
            vote = np.argmax(local_logits[i], axis=0)
            major = np.argmax(np.bincount(vote))
            if major == self.true_test_target[i]:
                ensemble_acc_vote += 1
            #
            # major_idx = np.where(vote == major)[0]
            # voted_logits = local_logits[i, :, major_idx]
            # mean_logits = np.mean(voted_logits, axis=0)
            # major_logits.append(mean_logits)

            mean_logits = np.mean(local_logits[i], axis=1)
            major = np.argmax(mean_logits)
            if major == self.true_test_target[i]:
                ensemble_acc_mean += 1
            # major_logits.append(mean_logits)

        ensemble_acc_vote = round(ensemble_acc_vote / len(global_logits) * 100, 2)
        ensemble_acc_mean = round(ensemble_acc_mean / len(global_logits) * 100, 2)

        #KL(True||Est) = KL(Ensemble||Aggregate)
        # major_logits = np.vstack(major_logits)
        # kl = compute_js_divergence(major_logits, global_logits)
        kl = 0

        ret = {
            'loss': global_ret['loss'],
            'acc': global_ret['acc'],
            'kld': kl,
            'ensemble_acc': {'vote': ensemble_acc_vote,
                             'mean': ensemble_acc_mean}
        }
        return ret

    def get_data(self, dataset_valid, dataset_locals, dataset_test):
        self.dataset_locals = dataset_locals

        # Validation 데이터 불러오기
        self.valid_loader = DataLoader(dataset_valid, batch_size=100, shuffle=True)
        self.test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False)

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

        params = [k for k in self.model.state_dict()]
        self.layers_name = []
        for layer in params:
            split_name = layer.split('.')[0]
            if 'weight' in split_name or 'bias' in split_name:
                raise RuntimeError

            self.layers_name.append(split_name)
        self.layers_name = np.sort(np.unique(self.layers_name))

    def make_opt(self):
        if self.args.optimizer.lower() == str('SGD').lower():
            self.server_optim = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                                momentum=self.args.momentum,
                                                weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == str('ADAM').lower():
            self.server_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                                 weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError

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
        else:
            raise NotImplementedError

    def get_global_model(self):
        return self.model
