import torch, json, yaml, argparse
import seaborn as sns; sns.set()
from argparse import Namespace
from pathlib import Path
from collections import namedtuple
from train_tools.utils import *

__all__ = ['Results', 'Logger', 'read_argv', 'make_exp_name']


Results = namedtuple('Results', ['train_loss', 'test_loss', 'test_acc', 'sparsity',
                                 'cost', 'round', 'exp_id', 'ellapsed_time', 'lr', 'var'])
rets = ['ACC', 'Loss']
xs = ['sparsity', 'round', 'Cost']


class Logger:
    def __init__(self):
        self.exp_results = {}
        self.path = f"./log"

        self.log = None
        self.args = None
        self.tot_sparsity = None

    def get_args(self, args):
        self.args = args
        self.tot_sparsity = args.target_sparsity

        # Create saving folder
        self.path = f"{self.path}/[{args.model}-{args.dataset}]{args.exp_name}"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        Path(f"{self.path}/heatmap").mkdir(parents=True, exist_ok=True)
        for k in sorted(vars(args).keys()):
            print("{}: {}".format(k, vars(args)[k]))
        print(f"\033[91mPath: {self.path}\033[00m")
        self.log = open(f"{self.path}/logfile.log", "a")

    def save_model(self, param, exp_id):
        Path(f"{self.path}/{exp_id}").mkdir(parents=True, exist_ok=True)
        torch.save(param, f"{self.path}/{exp_id}/model.h5")

    def get_results(self, results):
        self.print_data(results)
        self.add_results(results)

    def print_data(self, results):
        f = f"Train loss: {results.train_loss:.3f} "\
            f"Test loss: {results.test_loss:.3f} | "\
            f"Acc: {results.test_acc:.3f} | "\
            f"Time: {results.ellapsed_time:.2f}s | "\
            f"lr: {results.lr:.5f} | "\
            f"var: {results.var:.5f} | "
        print(f)
        self.log.write(f)

    def save_data(self):
        with open(f"{self.path}/results.json", 'w') as fp:
            json.dump(self.exp_results, fp)

    def add_results(self, results):
        if results.exp_id not in self.exp_results.keys():
            self.exp_results[results.exp_id] = self.make_basic_dict()

        self.exp_results[results.exp_id]['round'].append(results.round)
        self.exp_results[results.exp_id]['cost'].append(results.cost)
        self.exp_results[results.exp_id]['train_loss'].append(results.train_loss)
        self.exp_results[results.exp_id]['test_loss'].append(results.test_loss)
        self.exp_results[results.exp_id]['test_acc'].append(results.test_acc)
        self.exp_results[results.exp_id]['sparsity'].append(results.sparsity)
        self.exp_results[results.exp_id]['lr'].append(results.lr)
        self.exp_results[results.exp_id]['var'].append(results.var)

    def make_basic_dict(self):
        return {'round': [],
                'cost': [],
                'train_loss': [],
                'test_loss': [],
                'test_acc': [],
                'sparsity': [],
                'lr': [],
                'var': []
                }

    def save_yaml(self):
        f_name = f'{self.path}/exp_config.yaml'
        with open(f_name, 'w') as outfile:
            yaml.dump(self.args, outfile, default_flow_style=False)

    def plot_heatmap(self, plt, title):
        plt.savefig(f"{self.path}/heatmap/{title}.png")
        plt.show()
        plt.close()


def read_argv():
    parser = argparse.ArgumentParser(description='For Multiple experiments')
    parser.add_argument('--config_file', default='config.yaml', type=str)
    parser.add_argument('--nb_exp_reps', type=int)
    parser.add_argument('--nb_devices', type=int)
    parser.add_argument('--nb_rounds', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--pruning', type=str)
    parser.add_argument('--pruning_type', type=str)
    parser.add_argument('--plan_type', type=str)
    parser.add_argument('--plan', nargs='+', type=int)
    parser.add_argument('--decay_type', type=str)
    parser.add_argument('--recovery_type', type=str)
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
    parser.add_argument('--clip_type', type=str)
    parser.add_argument('--clip_dir', type=str)
    parser.add_argument('--ratio_clients_per_round', type=float)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--freezing_interval', type=int)
    additional_args = parser.parse_args()

    yaml_file = additional_args.config_file
    
    try:
        args = yaml.load(stream=open(f"config/{yaml_file}"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open(f"config/{yaml_file}", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = Namespace(**args)
    # general settings
    args.model = additional_args.model if additional_args.model is not None else args.model
    args.dataset = additional_args.dataset if additional_args.dataset is not None else args.dataset
    args.device = additional_args.device if additional_args.device is not None else get_device(args)
    args.cuda_type = additional_args.cuda_type if additional_args.cuda_type is not None else args.cuda_type
    
    # FL settings
    args.server_location = additional_args.server_location if additional_args.server_location is not None else get_server_location(args)
    args.nb_devices = additional_args.nb_devices if additional_args.nb_devices is not None else args.nb_devices
    args.local_ep = additional_args.local_ep if additional_args.local_ep is not None else args.local_ep
    args.nb_exp_reps = additional_args.nb_exp_reps if additional_args.nb_exp_reps is not None else args.nb_exp_reps
    args.nb_rounds = additional_args.nb_rounds if additional_args.nb_rounds is not None else args.nb_rounds
    
    # learning settings
    args.lr = additional_args.lr if additional_args.lr is not None else args.lr
    args.scheduler = additional_args.scheduler if additional_args.scheduler is not None else args.scheduler
    args.weight_decay = additional_args.weight_decay if additional_args.weight_decay is not None else args.weight_decay
    
    # pruning settings
    args.pruning = str2bool(additional_args.pruning if additional_args.pruning is not None else args.pruning)
    args.plan = additional_args.plan if additional_args.plan is not None else args.plan
    args.pruning_type = additional_args.pruning_type if additional_args.pruning_type is not None else args.pruning_type
    args.plan_type = additional_args.plan_type if additional_args.plan_type is not None else args.plan_type
    args.base_sparsity = additional_args.base_sparsity if additional_args.base_sparsity is not None else args.base_sparsity
    args.target_sparsity = additional_args.target_sparsity if additional_args.target_sparsity is not None else args.target_sparsity

    args.decay_type = additional_args.decay_type if additional_args.decay_type is not None else args.decay_type
    args.recovery_type = additional_args.recovery_type if additional_args.recovery_type is not None else args.recovery_type
    args.use_recovery_signal = str2bool(additional_args.use_recovery_signal \
                                        if additional_args.use_recovery_signal is not None else args.use_recovery_signal)
    args.signal_as_mask = str2bool(additional_args.signal_as_mask \
                                   if additional_args.signal_as_mask is not None else args.signal_as_mask)
    args.local_topk = additional_args.local_topk if additional_args.local_topk is not None else args.local_topk
    
    args.global_loss_type = additional_args.global_loss_type if additional_args.global_loss_type is not None else args.global_loss_type
    args.global_alpha = additional_args.global_alpha if additional_args.global_alpha is not None else args.global_alpha
    args.no_reg_to_recover = str2bool(additional_args.no_reg_to_recover) if additional_args.no_reg_to_recover is not None else args.no_reg_to_recover

    args.clip_type = additional_args.clip_type if additional_args.clip_type is not None else args.clip_type
    args.clip_dir = additional_args.clip_dir if additional_args.clip_dir is not None else args.clip_dir
    args.ratio_clients_per_round = additional_args.ratio_clients_per_round if additional_args.ratio_clients_per_round is not None else args.ratio_clients_per_round
    args.freezing_interval = additional_args.freezing_interval if additional_args.freezing_interval is not None else args.freezing_interval

    if args.exp_name is None:
        args.exp_name = additional_args.exp_name if additional_args.exp_name is not None else make_exp_name(args)

    args.model = args.model.lower()
    args.dataset = args.dataset.lower()
    return args


def make_exp_name(args):
    if args.pruning:
        title = f"{args.pruning_type}_{args.plan_type}_{args.decay_type}_{args.target_sparsity}_lr_"
    else:
        title = "vanilla_lr_"

    if args.clip_type != 'None':
        title += f"clip_{args.clip_type}_{args.clip_dir}_"

    return title + f"{args.scheduler}_{args.lr}_localep_{args.local_ep}"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


