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
        self.dataset_train, self.dataset_locals = None, None
        self.len_test_data = 0
        self.test_loader = None
        self.true_test_target = []

        # about optimization
        self.criterion = torch.nn.CrossEntropyLoss()
        self.aggregate_model_func = get_aggregation_func(self.args.aggregation_alg)
        self.server_optim = None
        self.server_lr_scheduler = None

        # about model
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
        self.betas = np.arange(0.1, 1, 0.2) if self.args.beta_validation else [self.args.beta]

    def train(self, exp_id=None):
        for fed_round in range(self.args.nb_rounds):
            start_time = time.time()
            print('==================================================')
            print(f'Epoch [{exp_id}: {fed_round+1}/{self.args.nb_rounds}]')

            sampled_devices = self.sampling_clients(self.nb_client_per_round)
            clients_dataset = [self.dataset_locals[i] for i in sampled_devices]

            for beta in self.betas:
                local_results = self.clients_training(clients_dataset=clients_dataset,
                                                      beta=beta)
                self.aggregation_models(local_results['updated_locals'], local_results['len_datasets'])

            test_results = self.test()
            # test_results = self.test_agg_vs_ensemble(local_results['updated_locals'])
            self.server_lr_scheduler.step()

            end_time = time.time()
            ellapsed_time = end_time - start_time
            self.logger.get_results(Results(local_results['loss'], test_results['loss'], test_results['acc'],
                                            0, self.tot_comm_cost,
                                            fed_round, exp_id, ellapsed_time,
                                            self.server_lr_scheduler.get_last_lr()[0],
                                            0, 0, 0))
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
            print('Local Train Acc :', np.array(train_acc)/100)
            print('Local Train KLD :', np.array(local_kld))

        ret = {
            'loss': train_loss,
            'kld': np.mean(local_kld),
            'len_datasets': len_datasets,
            'updated_locals': updated_locals
        }
        return ret

    def aggregation_models(self, updated_locals, len_datasets):
        self.model.load_state_dict(copy.deepcopy(self.aggregate_model_func(updated_locals, len_datasets)))

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

    def get_data(self, dataset_server, dataset_locals, dataset_test):
        self.dataset_train, self.dataset_locals = dataset_server, dataset_locals
        self.test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False)
        self.len_test_data = dataset_test.__len__()
        for i, (x, y) in enumerate(self.test_loader):
            self.true_test_target.append(y)
        self.true_test_target = np.concatenate(self.true_test_target)

    def make_model(self):
        model = create_nets(self.args, 'SERVER')
        dummy_model = create_nets(self.args, 'DUMMY')
        print(model)
        self.model = model.to(self.args.server_location)
        self.dummy_model = dummy_model.to(self.args.server_location)
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
