import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


def main():
    p = sys.argv[-1]
    if 'plot_figure.py' not in p:
        path = f'./log/{p}'
    else:
        path = f'./log'

    cifar_folders = [f.path for f in os.scandir(path) if f.is_dir() if 'cifar' in f.path]
    mnist_folders = [f.path for f in os.scandir(path) if f.is_dir() if 'mnist' in f.path]
    dirs = {'cifar': cifar_folders, 'mnist': mnist_folders}

    for data_type in dirs:
        dfs = []
        _dir = dirs[data_type]
        print(f"Read {_dir}...")

        for _subdir in _dir:
            if 'local_pruning_reverse' in _subdir:
                continue
            df = pd.read_csv(f"{_subdir}/results.csv")
            tmp_cost = 0
            for _id in df['exp_id'].unique():
                cost = df[df['exp_id'] == _id]['Cost'].values
                tmp_cost += cost

            tmp_cost /= len(df['exp_id'].unique())
            tmp_cost = tmp_cost.tolist()
            tmp_cost = tmp_cost * len(df['exp_id'].unique())
            df['Cost'] = np.array(tmp_cost)

            flag = []
            for _id in df['exp_id'].unique():
                if np.mean(df[df['exp_id'] == _id]['ACC'].dropna().values) < 20:
                    flag.append(_id)

            for _f in flag:
                df = df[~(df['exp_id'] == _f)]

            dfs.append(df)

        df = pd.concat(dfs)
        df["Experiments"] = "[" + df['exp_name'] + "] " + df['Legend']

        acc_df = df[df['ACC'].notna()]
        loss_df = df[df['Loss'].notna()]
        loss_train = loss_df[loss_df['Legend'] == 'Train']
        loss_test = loss_df[loss_df['Legend'] == 'Test']

        xs = ['Cost', 'round']
        rets = ['Loss', 'ACC']

        for _x in xs:
            for ret in rets:
                for legend in ['train', 'test']:
                    if 'Loss' == ret:
                        if 'mnist' == data_type:
                            plt.ylim(0, 1)
                        else:
                            plt.ylim(0, 2)
                        if 'train' == legend:
                            data = loss_train
                        else:
                            data = loss_test
                    else:
                        if 'train' == legend:
                            continue
                        else:
                            data = acc_df
                            if 'mnist' == data_type:
                                plt.ylim(80, 100)
                            else:
                                plt.ylim(0, 100)

                    print("Make plot...")
                    hue_order = np.sort(data['Experiments'].unique())
                    sns.lineplot(x=_x, y=ret, hue='Experiments', hue_order=hue_order, data=data)

                    if 'Cost' == _x:
                        plt.xlabel(f"{_x} (Mbytes)")
                    else:
                        plt.xlabel(f"{_x}")

                    plt.savefig(f"{path}/{data_type}_{_x}_{ret}_{legend}.png")
                    print("Plot saved!")
                    # plt.show()
                    plt.close()


if __name__ == '__main__':
    main()
