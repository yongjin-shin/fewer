import seaborn as sns; sns.set()
import torch
from argparse import Namespace
import copy
import os
import math
from pathlib import Path
from sys import getsizeof


def read_argv(*argv):
    argv = argv[0]
    if len(argv) < 2:
        raise RuntimeError("Please write proper config file name! e.g. python main.py config.yaml")
    else:
        if '.yaml' in argv[-1]:
            return argv[-1]
        else:
            raise RuntimeError("Please write proper config file name! e.g. python main.py config.yaml")

    # if not 'main.py' in _file:
    #     files = []
    #     for _, _, folder in os.walk(f'./config/{_file}'):
    #         for f in folder:
    #             if '.yaml' in f and not 'check' in folder:
    #                 files.append(f'{_file}/{f}')
    # else:
    #     files = ['config.yaml']
    #
    # folders = {}
    # for file in files:
    #     tmp = file[9:12]
    #     if tmp in folders:
    #         pass
    #     else:
    #         folders[tmp] = [f for f in files if tmp in f]

    # if len(folders) > 0:
    #     for _f in folders:
    #         Path(f'./log/{time}/{_f}').mkdir(parents=True, exist_ok=True)

    # return files, folders


def fix_arguments(args):
    args = Namespace(**args)

    if args.gpu:
        # args.device = torch.device("cuda:1")
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


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

    def step(self):
        pass


class LinearLR:
    def __init__(self, init_lr, epoch, eta_min):
        self.init_lr = init_lr
        self.crnt_lr = init_lr

        tot_diff = init_lr - eta_min
        self.diff = tot_diff / (epoch-1)

    def get_lr(self):
        return [self.crnt_lr]

    def step(self):
        self.crnt_lr -= self.diff

