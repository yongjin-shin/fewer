import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
import argparse, json, os
import matplotlib as mpl
import pandas as pd
from scipy.signal import savgol_filter
mpl.use('Agg')


class FolderOrganizer:
    def __init__(self, _dir):
        self.subdirs = np.sort(next(os.walk(_dir))[1])
        self.exps = []

    def parser(self):
        datas = []
        for subdir in self.subdirs:
            if '.DS_Store' == subdir:
                continue
            model_data, exp_info = subdir.split(']')
            model, data = model_data.split('-')
            model = model[1:]

            self.exps.append(SimpleName(model=model, data=data, dir=subdir))
            datas.append(data)
            self.exps[-1].add(exp_info)
            print("")

        if len(np.unique(datas)) > 1:
            raise RuntimeError("Different datasets")

    def get_exp_dirs(self):
        exp_dirs = []
        # exp.dir for exp in self.exps
        _order = ['vanilla', 'base', 'reverse']
        for target in _order:
            for exp in self.exps:
                if target in exp.exp_type:
                    exp_dirs.append(exp.dir)

        print(exp_dirs)
        return exp_dirs

    def get_legends(self):
        infos = pd.DataFrame([exp.to_dict() for exp in self.exps])
        col_for_legend = []
        for col in infos.columns:
            if len(infos[col].unique()) > 1:
                col_for_legend.append(col)

        for i in range(len(infos)):
            legend = []
            for col in col_for_legend:
                if col == 'spars':
                    continue
                legend.append(infos.loc[i, col])

            infos.loc[i, 'legend'] = '_'.join(legend)

        return infos['legend'].to_list()


class SimpleName:
    def __init__(self, model, data, dir):
        self.dir = dir
        self.model = model
        self.data = data
        self.exp_type = None
        self.target_spars = 0
        self.lr_type = None
        self.lr = 0
        self.localep = 0

    def add(self, exp_info):
        details = exp_info.split('_')
        self.parser(details)

    def parser(self, details):
        pos = 0
        tot_len = len(details)
        while pos < tot_len:
            if details[pos] == 'vanilla':
                self.exp_type = 'vanilla'

            if details[pos] == 'pruning':
                self.exp_type = details[pos+1]
                pos += 1

            if details[pos] == 'lr':
                if self.exp_type is not 'vanilla':
                    self.target_spars = details[pos-1]

                self.lr_type = details[pos+1]
                pos += 1

                self.lr = details[pos+1]
                pos += 1

            if details[pos] == 'localep':
                self.localep = details[pos+1]

            pos += 1

        print(self.__str__())

    def to_dict(self):
        ret = {
            "model": self.model,
            "data": self.data,
            "exp": self.exp_type,
            "spars": self.target_spars,
            "lr": self.lr,
            "lr_scheduler": self.lr_type,
            "local_ep": self.localep
        }
        return ret

    def __str__(self):
        ret = str(f"Model: {self.model} | "
                  f"Data: {self.data} | "
                  f"Exp: {self.exp_type} | "
                  f"Spars: {self.target_spars} | "
                  f"LR: {self.lr} | "
                  f"Scheduler: {self.lr_type} | "
                  f"LocalEp: {self.localep} |"
                  f"Dir: {self.dir}")
        return ret


def get_avg(data, xs, ys):
    raw = defaultdict(list)
    avg = defaultdict(dict)
    avg['loss'] = defaultdict(dict)
    cols = ys + xs

    for exp in data.keys():
        if np.mean(data[exp]['test_acc']) < 20:
            continue
        d = data[exp]
        for col in cols:
            raw[col].append(d[col])

    for col in cols:
        raw_vec = np.array(raw[col])
        mean_vec = np.nanmean(raw_vec, axis=0)
        std_vec = np.nanstd(raw_vec, axis=0)

        if col in xs:
            avg[col] = {'raw': mean_vec}

        else:
            if 'loss' in col:
                if 'train' in col:
                    avg['loss']['train'] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}
                else:
                    avg['loss']['test'] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}
            else:
                if 'acc' in col:
                    avg['acc'] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}
                else:
                    avg[col] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}

    return avg


def read_all(root, folders, args):
    default_ys = ['train_loss', 'test_loss', 'test_acc', 'var', 'lr', 'cost']
    ys = []
    for _y in args.ys:
        for default_y in default_ys:
            if _y in default_y:
                ys.append(default_y)

    data = []
    for folder in folders:
        path = root + folder
        with open(f'{path}/results.json') as f:
            d = json.load(f)
            dd = get_avg(d, args.xs, ys)
            dd['path'] = path
            data.append(dd)

    order_check(data, args.legend)
    return data


def order_check(data, legend):
    for d, l in zip(data, legend):
        if l not in d['path'].split('/')[-1]:
            try:
                if 'reverse' in d['path']:
                    pass
            except:
                raise RuntimeError(f"legend: {l} Real: {d['path']}")


def smoothing(args, y):
    if not args.no_smoothing:
        try:
            y_hat = savgol_filter(y, args.window_size, args.poly)
        except:
            y_hat = y
    else:
        y_hat = y

    return y_hat


def line_plot(args, x, y, y_type, label, color):
    if 'loss' == y_type:
        if args.no_test_loss:
            y_hat = smoothing(args, y['train']['mean'])
            plt.plot(x['raw'], y_hat, label=label, color=color, alpha=0.5, linestyle='--')
        else:
            y_hat = smoothing(args, y['test']['mean'])
            plt.plot(x['raw'], y_hat, label=label, lw=1, color=color, alpha=1)
            plt.fill_between(x['raw'], y['test']['lower'], y['test']['upper'], color=color, alpha=0.2)

            y_hat = smoothing(args, y['train']['mean'])
            plt.plot(x['raw'], y_hat, color=color, alpha=0.5, linestyle='--')
    else:
        if y_type == 'lr':
            y_hat = y['mean']
        else:
            y_hat = smoothing(args, y['mean'])

        plt.plot(x['raw'], y_hat, label=label, lw=2, color=color, alpha=1)
        plt.fill_between(x['raw'], y['lower'], y['upper'], color=color, alpha=0.2)


def plot(args, x, y, data):
    fig = plt.figure(figsize=(8, 6))
    colors = cm.Set1
    for idx, d in enumerate(data):
        line_plot(args, x=d[x], y=d[y], y_type=y, label=args.legend[idx], color=colors(idx))

    plt.xlabel('Round', fontsize=20) if 'round' in x else plt.xlabel('Cost', fontsize=20)
    if 'acc' in y:
        plt.ylabel('Accuracy (%)', fontsize=20)
        plt.ylim(args.ylim['acc'][0], args.ylim['acc'][1]) if args.ylim is not None else None
    elif 'loss' in y:
        plt.ylabel('Loss', fontsize=20)
        plt.ylim(args.ylim['loss'][0], args.ylim['loss'][1]) if args.ylim is not None else None
    else:
        plt.ylabel(y, fontsize=20)

    if 'round' in x:
        plt.xlim(args.xlim[0], args.xlim[1]) if args.xlim is not None else None

    plt.legend(fontsize=18, loc='best')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.savefig(f"{args.root}{args.title}_{x}_{y}.png")
    print(f"Save {args.root}{args.title}_{x}_{y}.png")


def main(args):
    data = read_all(args.root, args.exp_name, args)
    for x in args.xs:
        for y in args.ys:
            plot(args, x, y, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot parser')
    parser.add_argument('--root', default='./log/', type=str)
    parser.add_argument('--xs', default=['round'], nargs='+')
    parser.add_argument('--xlim', default=[0, 300], nargs='+', type=int)
    parser.add_argument('--ys', default=['loss', 'acc'], nargs='+', type=str)
    parser.add_argument('--ylim_acc', default=[0, 80], nargs='+', type=int)
    parser.add_argument('--ylim_loss', default=[0, 2], nargs='+', type=int)
    parser.add_argument('--ylim', default=None, type=int)
    parser.add_argument('--exp_name', default=False, nargs='+')
    parser.add_argument('--legend', nargs='+')
    parser.add_argument('--title', type=str)
    parser.add_argument('--no_test_loss', default=False, action='store_true')
    parser.add_argument('--no_smoothing', default=False, action='store_true')
    parser.add_argument('--window_size', default=51, type=int)
    parser.add_argument('--poly', default=3, type=int)
    args = parser.parse_args()

    if not args.exp_name:
        exps = FolderOrganizer(args.root)
        exps.parser()
        args.exp_name = exps.get_exp_dirs()
        if args.legend is None:
            args.legend = exps.get_legends()

    if args.title is None:
        args.title = args.root.split('/')[-2]

    if args.legend is None:
        raise RuntimeError("Please Enter legends")

    if 'cost' in args.xs:
        args.xlim = None

    args.ylim = {'acc': args.ylim_acc, 'loss': args.ylim_loss}
    main(args)
