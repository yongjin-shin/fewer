import torch, json, yaml, argparse
import seaborn as sns; sns.set()
from argparse import Namespace
from pathlib import Path
from collections import namedtuple
from train_tools.utils import *

__all__ = ['Results', 'Logger', 'read_argv', 'make_exp_name']

tup = ['train_loss', 'test_loss', 'test_acc', 'sparsity',
       'cost', 'round', 'exp_id', 'ellapsed_time', 'lr',
       'beta', 'ensemble_acc', 'layer_var', 'layer_norm']
Results = namedtuple('Results', tup,)


class Logger:
    def __init__(self):
        self.exp_results = {}
        self.path = f"./log"

        self.args = None
        self.tot_sparsity = None

    def get_args(self, args):
        self.args = args

        # Create saving folder
        self.path = f"{self.path}/[{args.model}-{args.dataset}]{args.exp_name}"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        for k in sorted(vars(args).keys()):
            print("{}: {}".format(k, vars(args)[k]))
        print(f"\033[91mPath: {self.path}\033[00m")

    def save_model(self, param, exp_id, description='final'):
        Path(f"{self.path}/{exp_id}").mkdir(parents=True, exist_ok=True)
        torch.save(param, f"{self.path}/{exp_id}/model_{str(description)}.h5")

    def load_model(self, exp_id, description='final'):
        return torch.load(f"{self.path}/{exp_id}/model_{str(description)}.h5")

    def get_results(self, results):
        self.print_data(results)
        self.add_results(results)

    def print_data(self, results):
        print(f"Train loss: {results.train_loss:.3f} "
              f"Test loss: {results.test_loss:.3f} | "
              f"Acc: {results.test_acc:.3f}/{results.ensemble_acc:.3f} | "
              f"Beta: {results.beta:.2f} | "
              # f"D(G2L): {results.d_g2l:.3f} | "
              # f"D(E2G): {results.d_e2g:.3f} | "
              f"Time: {results.ellapsed_time:.2f}s | "
              f"lr: {results.lr:.5f}"
              )

    def save_data(self):
        with open(f"{self.path}/results.json", 'w') as fp:
            json.dump(self.exp_results, fp)

    def add_results(self, results):
        if results.exp_id not in self.exp_results.keys():
            self.exp_results[results.exp_id] = self.make_basic_dict()

        for _item in results._fields:
            self.exp_results[results.exp_id][_item].append(getattr(results, _item))

        # self.exp_results[results.exp_id]['round'].append(results.round)
        # self.exp_results[results.exp_id]['cost'].append(results.cost)
        # self.exp_results[results.exp_id]['train_loss'].append(results.train_loss)
        # self.exp_results[results.exp_id]['test_loss'].append(results.test_loss)
        # self.exp_results[results.exp_id]['test_acc'].append(results.test_acc)
        # self.exp_results[results.exp_id]['ensemble_acc'].append(results.ensemble_acc)
        # self.exp_results[results.exp_id]['sparsity'].append(results.sparsity)
        # self.exp_results[results.exp_id]['lr'].append(results.lr)
        # # self.exp_results[results.exp_id]['d_e2g'].append(results.d_e2g)
        # # self.exp_results[results.exp_id]['d_g2l'].append(results.d_g2l)
        # self.exp_results[results.exp_id]['beta'].append(results.beta)

    def make_basic_dict(self):
        ret = {}
        for _item in tup:
            ret[_item] = []
        return ret

    def save_yaml(self):
        f_name = f'{self.path}/exp_config.yaml'
        with open(f_name, 'w') as outfile:
            yaml.dump(self.args, outfile, default_flow_style=False)


def read_argv():
    parser = argparse.ArgumentParser(description='For Multiple experiments')
    parser.add_argument('--config_file', default='config.yaml', type=str)
    parser.add_argument('--nb_exp_reps', type=int)
    parser.add_argument('--nb_devices', type=int)
    parser.add_argument('--nb_rounds', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--smoothing', type=float)
    parser.add_argument('--temp', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--use_beta_scheduler', type=str)
    parser.add_argument('--beta_schedule_type', type=str)
    parser.add_argument('--beta_validation', type=str)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--oracle', type=str)
    parser.add_argument('--oracle_path', type=str)
    parser.add_argument('--pruning', type=str)
    parser.add_argument('--pruning_type', type=str)
    parser.add_argument('--plan_type', type=str)
    parser.add_argument('--plan', nargs='+', type=int)
    parser.add_argument('--decay_type', type=str)
    parser.add_argument('--use_recovery_signal', type=str)
    parser.add_argument('--local_topk', type=float)
    parser.add_argument('--signal_as_mask', type=str)
    parser.add_argument('--global_loss_type', type=str)
    parser.add_argument('--global_alpha', type=float)
    parser.add_argument('--no_reg_to_recover', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--target_sparsity', type=float)
    parser.add_argument('--base_sparsity', type=float)
    parser.add_argument('--local_ep', type=int)
    parser.add_argument('--cuda_type', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--server_location', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nb_server_data', type=int)
    parser.add_argument('--iid', type=str)
    parser.add_argument('--data_hetero_alg', type=str)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--ratio_clients_per_round', type=float)

    parser.add_argument('--slow_layer', nargs='+', default=None, type=int)
    parser.add_argument('--slow_ratio', type=float)
    parser.add_argument('--dir_alpha', type=float)

    additional_args = parser.parse_args()

    yaml_file = additional_args.config_file
    
    try:
        args = yaml.load(stream=open(f"config/{yaml_file}"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open(f"config/{yaml_file}", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = Namespace(**args)
    # motivation settings
    args.slow_layer = additional_args.slow_layer if additional_args.slow_layer is not None else args.slow_layer
    args.slow_ratio = additional_args.slow_ratio if additional_args.slow_ratio is not None else args.slow_ratio

    # general settings
    args.model = additional_args.model if additional_args.model is not None else args.model
    args.device = additional_args.device if additional_args.device is not None else get_device(args)
    args.cuda_type = additional_args.cuda_type if additional_args.cuda_type is not None else args.cuda_type
    
    # FL settings
    args.server_location = additional_args.server_location if additional_args.server_location is not None else get_server_location(args)
    args.nb_devices = additional_args.nb_devices if additional_args.nb_devices is not None else args.nb_devices
    args.local_ep = additional_args.local_ep if additional_args.local_ep is not None else args.local_ep
    args.nb_exp_reps = additional_args.nb_exp_reps if additional_args.nb_exp_reps is not None else args.nb_exp_reps
    args.nb_rounds = additional_args.nb_rounds if additional_args.nb_rounds is not None else args.nb_rounds
    args.ratio_clients_per_round = additional_args.ratio_clients_per_round if additional_args.ratio_clients_per_round is not None else args.ratio_clients_per_round

    # dataset settings
    args.dataset = additional_args.dataset if additional_args.dataset is not None else args.dataset
    args.nb_server_data = additional_args.nb_server_data if additional_args.nb_server_data is not None else args.nb_server_data
    args.iid = str2bool(additional_args.iid) if additional_args.iid is not None else args.iid
    args.data_hetero_alg = additional_args.data_hetero_alg if additional_args.data_hetero_alg is not None else args.data_hetero_alg
    args.dir_alpha = additional_args.dir_alpha if additional_args.dir_alpha is not None else args.dir_alpha

    # learning settings
    args.lr = additional_args.lr if additional_args.lr is not None else args.lr
    args.scheduler = additional_args.scheduler if additional_args.scheduler is not None else args.scheduler
    args.weight_decay = additional_args.weight_decay if additional_args.weight_decay is not None else args.weight_decay
    
    # distill settings
    args.mode = additional_args.mode if additional_args.mode is not None else args.mode
    args.smoothing = additional_args.smoothing if additional_args.smoothing is not None else args.smoothing
    args.temp = additional_args.temp if additional_args.temp is not None else args.temp
    args.beta = additional_args.beta if additional_args.beta is not None else args.beta
    args.use_beta_scheduler = str2bool(
        additional_args.use_beta_scheduler) if additional_args.use_beta_scheduler is not None else args.use_beta_scheduler
    args.beta_schedule_type = additional_args.beta_schedule_type if additional_args.beta_schedule_type is not None else args.beta_schedule_type
    args.beta_validation = str2bool(
        additional_args.beta_validation) if additional_args.beta_validation is not None else args.beta_validation
    args.oracle = str2bool(additional_args.oracle if additional_args.oracle is not None else args.oracle)
    args.oracle_path = additional_args.oracle_path if additional_args.oracle_path is not None else args.oracle_path

    # FedProx settings
    args.global_loss_type = additional_args.global_loss_type if additional_args.global_loss_type is not None else args.global_loss_type
    args.global_alpha = additional_args.global_alpha if additional_args.global_alpha is not None else args.global_alpha
    args.no_reg_to_recover = str2bool(additional_args.no_reg_to_recover) if additional_args.no_reg_to_recover is not None else args.no_reg_to_recover

    # other settings
    args.exp_name = additional_args.exp_name if additional_args.exp_name is not None else make_exp_name(args)
    args.model = args.model.lower()
    args.dataset = args.dataset.lower()
    return args


def make_exp_name(args):
    if args.exp_name is None:
        title = ''
    else:
        title = args.exp_name + '_'

    if args.global_loss_type == 'l2':
        title += f'fedprox_{args.global_alpha}_'

    return title + f"_iid_{args.iid}_lr_{args.scheduler}_{args.lr}_localep_{args.local_ep}"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


