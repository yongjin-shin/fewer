import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
import argparse, json, os
import matplotlib as mpl
import pandas as pd
from scipy.signal import savgol_filter

# ce_lp05_layer$layer""_lr$lr

def plot(mode, lp, decaying):
    root = './log/[cifarcnn-cifar10]'
    name = ''
    if decaying == 'linear':
        root += 'linear_'
        name += 'linear_'
    
    if mode == 'CE':
        root += 'real_'
        name += 'CE'
    else:
        name += 'KD'
    
    root += 'ce_'
    
    if lp == 5:
        root += 'lp05'
        name += '+lep05'
    else:
        root += 'lp20'
        name += '+lep20'
    
    print(root)
    print(name)
    
    _cols = [0, 0.001, 0.01, 1, 2, 5, 10, 20]
    df = pd.DataFrame(data=np.zeros(shape=(5, len(_cols))), columns=_cols,
                    index=['conv1', 'conv2', 'fc1', 'fc2', 'fc3'])

    for i in range(5):
        for j in df.columns.values:
            # lr = lrs[j]
            if j % 1:
                path = f'{root}_layer{int(i)}_lr{j}'
                print(path)
            else:
                path = f'{root}_layer{int(i)}_lr{int(j)}'
                print(path)

            try:
                with open(f'{path}/results.json') as f:
                    d = json.load(f)
                    # max_acc = np.max(d['0']['test_acc'])
                    max_acc = np.mean(d['0']['test_acc'][-3:])
                    df.loc[df.index.values[i], j] = max_acc
            except:
                df.loc[df.index.values[i], j] = None
                
    print(df)
    x = np.arange(len(df.columns.values))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for row in df.iterrows():
        ax.plot(x, row[1], label=row[0], marker='o')
        
    plt.xlabel('N x learning rate')
    plt.xticks(ticks=x, labels=np.array(df.columns.values, dtype=str))
    plt.ylabel('Last 3 Avg Acc')
    plt.title(name)
    plt.ylim((50, 75))
    plt.legend(loc='best')
    plt.savefig(f"{name}.png")

plot(mode='CE', lp=5, decaying='constant')
plot(mode='CE', lp=20, decaying='constant')
plot(mode='CE', lp=5, decaying='linear')
plot(mode='CE', lp=20, decaying='linear')

plot(mode='KD', lp=5, decaying='constant')
plot(mode='KD', lp=20, decaying='constant')
plot(mode='KD', lp=5, decaying='linear')
plot(mode='KD', lp=20, decaying='linear')
