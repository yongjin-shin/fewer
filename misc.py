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


class Scheduler:
    def __init__(self, max_iter, init_lr, end_lr):
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.lr = init_lr

    def get_lr(self, _iter):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

# 반복실험 결과 저장
# def save_results(path, args, all_exps):
#     all_exps = np.concatenate(all_exps)
#     np.save(f'{path}/data_{args.dataset}_allexps.npy', all_exps)
#     plot_graph(args, all_exps, path)


# 반복실험 결과 plot
# def plot_graph(args, data, path):
#     cols = ['loss_train', 'loss_test', 'acc_test', 'round', 'exp']
#     df = pd.DataFrame(data, columns=cols)
#     for c in cols[:-2]:
#         plt.plot(df['round'], df[c])
#         if 'acc' in c:
#             plt.ylim(60,100)
#         elif 'loss' in c:
#             plt.ylim(0,0.3)
#         plt.title(f'Dataset: {args.dataset} | {c}')
#         plt.savefig(f'{path}/data_{args.dataset}_{c}.png')
#         print(f"saved {path}/data_{args.dataset}_{c}.png")
#         plt.show()
#         plt.close()
