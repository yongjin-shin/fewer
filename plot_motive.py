import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
import argparse, json, os
import matplotlib as mpl
import pandas as pd
from scipy.signal import savgol_filter

# ce_lp05_layer$layer""_lr$lr

_layers = ['input', 'conv1', 'conv2', 'fc1', 'fc2']  # ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
_lrs =  [1, 5, 10, 20 , 50, 100] # [0, 0.001, 0.01, 1, 2, 5]
df = pd.DataFrame(data=np.zeros(shape=(len(_layers), len(_lrs))),
                  columns=np.arange(len(_lrs)),
                  index=_layers)


for i in range(len(_layers)):
# for i in range(1, len(_layers)+1):
    for j in df.columns.values:
        lr = _lrs[j]
        if lr % 1:
            # path = f'./log/[cifarcnn-cifar10]real_ce_lp05_layer{int(i)}_lr{lr}'
            path = f'./log/[cifarcnn-cifar10]mixup{int(lr)}_layer{int(i)}'
        else:
            # path = f'./log/[cifarcnn-cifar10]real_ce_lp05_layer{int(i)}_lr{int(lr)}'
            path = f'./log/[cifarcnn-cifar10]mixup{int(lr)}_layer{int(i)}'

        with open(f'{path}/results.json') as f:
            d = json.load(f)
            max_acc = np.mean(d['0']['test_acc'][-10:])  # max_acc = np.max(d['0']['test_acc'])
            # df.loc[df.index.values[i], j] = max_acc
            df.loc[df.index.values[i], j] = max_acc

print(df)

df = df.T
df.plot(marker='o')
plt.xlabel('N x learning rate')
plt.xticks(ticks=df.index.values, labels=np.array(_lrs, dtype=str))
plt.ylabel('Last 5 Avg Acc')
plt.title('CE + lep05')
plt.ylim((50, 70))

with open(f'./log/[cifarcnn-cifar10]lep05/results.json') as f:
    d = json.load(f)
    baseline_score = np.mean(d['0']['test_acc'][-10:])  # max_acc = np.max(d['0']['test_acc'])
    print(baseline_score)
    plt.hlines(baseline_score, min(df.index.values), max(df.index.values))

plt.savefig('test.png')

