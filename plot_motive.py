import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
import argparse, json, os
import matplotlib as mpl
import pandas as pd
from scipy.signal import savgol_filter

# ce_lp05_layer$layer""_lr$lr

lrs = [0, 0.001, 0.01, 1, 2, 5]
df = pd.DataFrame(data=np.zeros(shape=(5, 6)), columns=[0, 1, 2, 3, 4, 5],
                  index=['conv1', 'conv2', 'fc1', 'fc2', 'fc3'])

for i in range(5):
    for j in df.columns.values:
        lr = lrs[j]
        if lr % 1:
            path = f'./log/[cifarcnn-cifar10]real_ce_lp05_layer{int(i)}_lr{lr}'
        else:
            path = f'./log/[cifarcnn-cifar10]real_ce_lp05_layer{int(i)}_lr{int(lr)}'

        with open(f'{path}/results.json') as f:
            d = json.load(f)
            # max_acc = np.max(d['0']['test_acc'])
            max_acc = np.mean(d['0']['test_acc'][-3:])
            df.loc[df.index.values[i], j] = max_acc

print(df)

df = df.T
df.plot(marker='o')
plt.xlabel('N x learning rate')
plt.xticks(ticks=df.index.values, labels=np.array(lrs, dtype=str))
plt.ylabel('Last 5 Avg Acc')
plt.title('CE + lep05')
plt.ylim((50, 70))
plt.show()
print("")
