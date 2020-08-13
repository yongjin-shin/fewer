import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
import argparse, json
import matplotlib as mpl
mpl.use('Agg')

ys = ['train_loss', 'test_loss', 'test_acc']


def get_avg(data, xs, ys):
    raw = defaultdict(list)
    avg = defaultdict(dict)
    avg['loss'] = defaultdict(dict)
    cols = ys + xs

    for exp in data.keys():
        d = data[exp]
        for col in cols:
            raw[col].append(d[col])

    for col in cols:
        raw_vec = np.array(raw[col])
        mean_vec = np.mean(raw_vec, axis=0)
        std_vec = np.std(raw_vec, axis=0)

        if col in xs:
        #    if any(list(std_vec)) > 0:
        #        raise RuntimeError("xs should be same for all the exp")
        #    else:
            avg[col] = {'raw': mean_vec}
        
        else:
            if 'loss' in col:
                if 'train' in col:
                    avg['loss']['train'] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}
                else:
                    avg['loss']['test'] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}
            else:
                avg['acc'] = {'mean': mean_vec, 'upper': mean_vec + std_vec, 'lower': mean_vec - std_vec}

    return avg


def read_all(root, folders, args):
    data = []
    for folder in folders:
        path = root + folder
        with open(f'{path}/results.json') as f:
            d = json.load(f)
            data.append(get_avg(d, args.xs, ys))
    return data


def line_plot(x, y, y_type, label, color):
    if 'loss' == y_type:
        plt.plot(x['raw'], y['test']['mean'], label=label, lw=1, color=color, alpha=1)
        plt.fill_between(x['raw'], y['test']['lower'], y['test']['upper'], color=color, alpha=0.2)
        plt.plot(x['raw'], y['train']['mean'], color=color, alpha=0.5, linestyle='--')
    else:
        plt.plot(x['raw'], y['mean'], label=label, lw=1, color=color, alpha=1)
        plt.fill_between(x['raw'], y['lower'], y['upper'], color=color, alpha=0.2)


def plot(x, y, data, title, xlim=None, ylim=None):
    fig = plt.figure(figsize=(8, 6))
    colors = cm.Set1
    for idx, d in enumerate(data):
        line_plot(x=d[x], y=d[y], y_type=y, label=args.legend[idx], color=colors(idx))

    plt.xlabel('Round', fontsize=20) if 'round' in x else plt.xlabel('Cost', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20) if 'acc' in y else plt.ylabel('Loss', fontsize=20)
    plt.xlim(xlim[0], xlim[1]) if xlim is not None else None
    plt.ylim(ylim[0], ylim[1]) if ylim is not None else None
    plt.legend(fontsize=18, loc='best')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.savefig(f"./log/{x}_{y}_{title}.png")


def main(args):
    data = read_all(args.root, args.exp_name, args)
    for x in args.xs:
        for y in ['loss', 'acc']:
            plot(x, y, data, args.title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot parser')
    parser.add_argument('--root', default='./log/', type=str)
    parser.add_argument('--xs', nargs='+')
    parser.add_argument('--exp_name', nargs='+')
    parser.add_argument('--legend', default=False, nargs='+')
    parser.add_argument('--title', type=str)
    args = parser.parse_args()
    if not args.legend:
        args.legend = args.exp_name
    main(args)
