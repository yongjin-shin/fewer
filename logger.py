import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

Results = namedtuple('Results', ['train_loss', 'test_loss', 'test_acc', 'sparcity', 'cost', 'round', 'exp_id', 'ellapsed_time'])

rets = ['ACC', 'Loss']
xs = ['sparcity', 'round', 'Cost']
Table_items = ['exp_name'] + xs + ['exp_id'] + rets + ['Legend']
Tabletup = namedtuple('Tabletup', Table_items)


class Logger:
    def __init__(self, _time, argv, folder):
        self.df = pd.DataFrame(columns=Table_items)
        self.root = f"./log/{_time}_{argv}/{folder}"

        self.args = None
        self.tot_sparsity = None
        self.path = None

    def get_args(self, args):
        self.args = args
        self.tot_sparsity = np.matmul(args.pruning_pack, args.pruning_plan)

        # Create saving folder
        self.path = f"{self.root}/[{args.model}-{args.dataset}]{args.experiment_name}"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        for k in sorted(vars(args).keys()):
            print("{}: {}".format(k, vars(args)[k]))
        print(f"\033[91mPath: {self.path}\033[00m")

    def save_model(self, param, exp_id):
        Path(f"{self.path}/{exp_id}").mkdir(parents=True, exist_ok=True)
        torch.save(param, f"{self.path}/{exp_id}/model.h5")

    def get_results(self, results):
        train_loss = Tabletup(self.args.experiment_name, results.sparcity, results.round, results.cost, results.exp_id, float('NaN'), results.train_loss, 'Train')
        test_loss = Tabletup(self.args.experiment_name, results.sparcity, results.round, results.cost, results.exp_id, float('NaN'), results.test_loss, 'Test')
        test_acc = Tabletup(self.args.experiment_name, results.sparcity, results.round, results.cost, results.exp_id, results.test_acc, float('NaN'), 'Test')

        self.df = self.df.append(train_loss._asdict(), ignore_index=True)
        self.df = self.df.append(test_loss._asdict(), ignore_index=True)
        self.df = self.df.append(test_acc._asdict(), ignore_index=True)
        self.print_data(results)

    def print_data(self, results):
        print(f"Train loss: {results.train_loss:.3f} "
              f"Test loss: {results.test_loss:.3f} | "
              f"Acc: {results.test_acc:.3f} | "
              f"Time: {results.ellapsed_time:.2f}s ")

    def save_data(self):
        self.df.to_csv(f"{self.root}/results.csv")
        self.plot(print_avg=True)

    def plot(self, print_avg=False, exp_id=None):
        if print_avg:
            tmp = self.df
            path = self.path
        else:
            if exp_id is None:
                raise RuntimeError
            else:
                tmp = self.df.loc[self.df['exp_id'] == exp_id]
                path = f"{self.path}/{exp_id}"

        for _x in xs:

            if self.args.experiment_name == 'baseline' and _x == 'sparcity':
                continue

            for ret in rets:
                sns.lineplot(x=_x, y=ret,  hue='Legend', style='Legend', markers=True, data=tmp[tmp[ret].notna()])

                if 'sparcity' == _x:
                    plt.xlabel(f"{_x} (%)")
                elif 'Cost' == _x:
                    plt.xlabel(f"{_x} (Mbytes)")
                else:
                    plt.xlabel(f"{_x}")

                if 'ACC' in ret:
                    plt.ylabel(f"{ret}(%)")
                    plt.ylim(40, 100)
                elif 'Loss' in ret:
                    plt.ylabel(f"{ret}")
                    plt.ylim(0, 1)

                plt.title(f"Dataset: {self.args.dataset} | Model: {self.args.model}")
                plt.tight_layout()
                plt.savefig(f'{path}/{_x}_{ret}.png')
                print(f"\033[91mSaved {path}/{_x}_{ret}.png\033[00m")
                plt.show()
                plt.close()

    def global_plot(self, files):
        file = f"{self.root}/results.csv"
        df = pd.read_csv(file)
        df["legend"] = "[" + df['exp_name'] + "] " + df['Legend']

        acc_df = df[df['ACC'].notna()]
        loss_df = df[df['Loss'].notna()]

        for _x in xs:
            for ret in rets:

                if 'sparcity' == _x:
                    plt.xlabel(f"{_x} (%)")
                elif 'Cost' == _x:
                    plt.xlabel(f"{_x} (Mbytes)")
                else:
                    plt.xlabel(f"{_x}")

                if 'Loss' == ret:
                    data = loss_df
                else:
                    data = acc_df

                if _x == 'sparcity':
                    data = data[~(data['exp_name'] == 'baseline')]

                sns.lineplot(x=_x, y=ret, hue='legend', style='legend', markers=True, data=data)
                plt.savefig(f"{self.root}/{_x}_{ret}.png")
                plt.close()

        print("")
