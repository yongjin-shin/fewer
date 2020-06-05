
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import torch
from argparse import Namespace


def fix_arguments(args):
    args = Namespace(**args)

    if args.gpu:
        # args.device = torch.device("cuda:1")
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    for k in sorted(vars(args).keys()):
        print("{}: {}".format(k, vars(args)[k]))

    return args


# 반복실험 결과 저장
def save_results(path, args, all_exps):
    all_exps = np.concatenate(all_exps)
    np.save(f'{path}/data_{args.dataset}_allexps.npy', all_exps)
    plot_graph(args, all_exps, path)


# 반복실험 결과 plot
def plot_graph(args, data, path):
    cols = ['loss_train', 'loss_test', 'acc_test', 'round', 'exp']
    df = pd.DataFrame(data, columns=cols)
    for c in cols[:-2]:
        plt.plot(df['round'], df[c])
        if 'acc' in c:
            plt.ylim(60,100)
        elif 'loss' in c:
            plt.ylim(0,0.3)
        plt.title(f'Dataset: {args.dataset} | {c}')
        plt.savefig(f'{path}/data_{args.dataset}_{c}.png')
        print(f"saved {path}/data_{args.dataset}_{c}.png")
        plt.show()
        plt.close()
