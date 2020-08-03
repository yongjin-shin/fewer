import seaborn as sns; sns.set()
from torch.optim.lr_scheduler import StepLR
from argparse import Namespace
import copy
import yaml
import argparse
from math import log


def read_argv():
    parser = argparse.ArgumentParser(description='For Multiple experiments')
    parser.add_argument('--config_file', default='config.yaml', type=str)
    parser.add_argument('--nb_exp_reps', type=int)
    parser.add_argument('--nb_devices', type=int)
    parser.add_argument('--nb_rounds', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--pruning_type', type=str)
    parser.add_argument('--plan_type', type=str)
    parser.add_argument('--decay_type', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--target_sparsity', type=float)
    parser.add_argument('--local_ep', type=int)
    parser.add_argument('--cuda_type', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--dataset', type=str)
    additional_args = parser.parse_args()

    yaml_file = additional_args.config_file
    try:
        args = yaml.load(stream=open(f"config/{yaml_file}"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open(f"config/{yaml_file}", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = Namespace(**args)
    args.nb_devices = additional_args.nb_devices if additional_args.nb_devices is not None else args.nb_devices
    args.nb_exp_reps = additional_args.nb_exp_reps if additional_args.nb_exp_reps is not None else args.nb_exp_reps
    args.nb_rounds = additional_args.nb_rounds if additional_args.nb_rounds is not None else args.nb_rounds
    args.pruning_type = additional_args.pruning_type if additional_args.pruning_type is not None else args.pruning_type
    args.plan_type = additional_args.plan_type if additional_args.plan_type is not None else args.plan_type
    args.decay_type = additional_args.decay_type if additional_args.decay_type is not None else args.decay_type
    args.scheduler = additional_args.scheduler if additional_args.scheduler is not None else args.scheduler
    args.target_sparsity = additional_args.target_sparsity if additional_args.target_sparsity is not None else args.target_sparsity
    args.lr = additional_args.lr if additional_args.lr is not None else args.lr
    args.device = additional_args.device if additional_args.device is not None else get_device(args)
    args.local_ep = additional_args.local_ep if additional_args.local_ep is not None else args.local_ep
    args.cuda_type = additional_args.cuda_type if additional_args.cuda_type is not None else args.cuda_type
    args.weight_decay = additional_args.weight_decay if additional_args.weight_decay is not None else args.weight_decay
    args.dataset = additional_args.dataset if additional_args.dataset is not None else args.dataset
    args.model = additional_args.model if additional_args.model is not None else args.model

    args.experiment_name = make_exp_name(args)
    return args


def get_device(args):
    if args.gpu:
        if args.cuda_type:
            return 'cuda:1'
        else:
            return 'cuda:0'
    else:
        return 'cpu'


def make_exp_name(args):
    if args.pruning:
        title = f"{args.pruning_type}_{args.plan_type}_{args.target_sparsity}_lr_"
    else:
        title = "vanilla_lr_"

    return title + f"{args.scheduler}_{args.lr}_localep_{args.local_ep}_wd_{args.weight_decay}"


def model_location_switch_downloading(model, args):
    if args.gpu:
        if 'gpu' == args.server_location:
            return copy.deepcopy(model.state_dict())
        else:
            _state_dict = copy.deepcopy(model.state_dict())
            for i in _state_dict:
                _state_dict[i] = _state_dict[i].to(args.device)
            return _state_dict
    else:
        if 'gpu' == args.server_location:
            raise RuntimeError("This cannot be happened!")
        else:
            return copy.deepcopy(model.state_dict())


def model_location_switch_uploading(model, args):
    if args.gpu:
        if 'gpu' == args.server_location:
            return copy.deepcopy(model.state_dict())
        else:
            _state_dict = copy.deepcopy(model.state_dict())
            for i in _state_dict:
                _state_dict[i] = _state_dict[i].to('cpu')
            return _state_dict
    else:
        if 'gpu' == args.server_location:
            raise RuntimeError("This cannot be happened!")
        else:
            return copy.deepcopy(model.state_dict())


def mask_location_switch(keeped_masks, _device):
    for i in keeped_masks:
        keeped_masks[i] = keeped_masks[i].to(_device)
    return keeped_masks


def get_size(param):
    size = 0

    for p in param:
        tmp = p.detach().to('cpu').numpy()
        size += tmp.nbytes

    return round(size/1024/1024, 2)


class ConstantLR:
    def __init__(self, init_lr):
        self.init_lr = init_lr
        self.crnt_lr = self.init_lr

    def get_lr(self):
        return [self.crnt_lr]
    
    def get_last_lr(self):
        return [self.crnt_lr]

    def step(self):
        pass


class LinearLR:
    def __init__(self, init_lr, epoch, eta_min):
        self.init_lr = init_lr
        self.crnt_lr = init_lr

        tot_diff = init_lr - eta_min
        self.diff = tot_diff / (epoch-1)

    def get_last_lr(self):
        return [self.crnt_lr]

    def step(self):
        self.crnt_lr -= self.diff


class LinearStepLR:
    def __init__(self, optimizer, init_lr, epoch, eta_min, decay_rate):
        n = int((log(eta_min) - log(init_lr))/log(decay_rate)) + 1
        step_size = int(epoch/n)
        self.scheduler = StepLR(optimizer=optimizer, gamma=decay_rate,
                                step_size=step_size)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def step(self):
        self.scheduler.step()
