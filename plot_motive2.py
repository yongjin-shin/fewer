import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
import argparse, json, os
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
from scipy.signal import savgol_filter


def plot(mode):
    data_mean = {}
    data_std = {}
    for layer in ts[0].keys():
        layer_mean, layer_std = [], []
        for t in ts:
            layer_mean.append(t[layer][mode]['mean'])
            layer_std.append(t[layer][mode]['std'])
        
        data_mean[layer] = layer_mean
        data_std[layer] = layer_std

    fig, ax = plt.subplots()
    X = np.arange(len(data_std[list(ts[0].keys())[0]])) * 100
    cmap = cm.Set1
    cnt = -1
    for layer in ts[0].keys():
        cnt += 1
        y = np.array(data_mean[layer])
        error = np.array(data_std[layer])
        plt.plot(X, error, color=cmap(cnt), alpha=0.2)
        plt.plot(X, savgol_filter(error, 11, 3), label=layer, color=cmap(cnt))
        # plt.plot(X, y, label=layer)
        # plt.fill_between(X, y - error, y + error, alpha=0.2)
        
    plt.legend(loc='best')
    plt.ylabel('std', fontsize=15)
    plt.xlabel('Comm Rounds', fontsize=15)
    plt.title(f'{mode}', fontsize=20)
    ax.figure.savefig(f'plot_{mode}.png')


path = '[cifarcnn-cifar10]maximum_comm5000_swap_noise'
with open(f'./log/{path}/results.json') as f:
    d = json.load(f)

ts = np.array(d['0']['tubulance_info'])[np.array(d['0']['tubulance_info']) != None]

plot(mode='acc')
plot(mode='loss')

