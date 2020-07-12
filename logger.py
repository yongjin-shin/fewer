import json
import yaml
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from collections import namedtuple
from copy import deepcopy
import numpy as np

Results = namedtuple('Results', ['train_loss', 'test_loss', 'test_acc', 'sparsity', 'cost', 'round', 'exp_id', 'ellapsed_time', 'lr'])
rets = ['ACC', 'Loss']
xs = ['sparsity', 'round', 'Cost']


class Logger:
    def __init__(self):
        # self.df = pd.DataFrame(columns=Table_items)
        self.exp_results = {}
        self.path = f"./log"

        self.args = None
        self.tot_sparsity = None

        self.cmap = cm.Set2

    def get_args(self, args):
        self.args = args
        self.tot_sparsity = args.target_sparsity

        # Create saving folder
        self.path = f"{self.path}/[{args.model}-{args.dataset}]{args.experiment_name}"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        for k in sorted(vars(args).keys()):
            print("{}: {}".format(k, vars(args)[k]))
        print(f"\033[91mPath: {self.path}\033[00m")

    def save_model(self, param, exp_id):
        Path(f"{self.path}/{exp_id}").mkdir(parents=True, exist_ok=True)
        torch.save(param, f"{self.path}/{exp_id}/model.h5")

    def get_results(self, results):
        self.print_data(results)
        self.add_results(results)

        # train_loss = Tabletup(self.args.experiment_name, results.sparsity, results.round, results.cost, results.exp_id, float('NaN'), results.train_loss, 'Train')
        # test_loss = Tabletup(self.args.experiment_name, results.sparsity, results.round, results.cost, results.exp_id, float('NaN'), results.test_loss, 'Test')
        # test_acc = Tabletup(self.args.experiment_name, results.sparsity, results.round, results.cost, results.exp_id, results.test_acc, float('NaN'), 'Test')
        #
        # self.df = self.df.append(train_loss._asdict(), ignore_index=True)
        # self.df = self.df.append(test_loss._asdict(), ignore_index=True)
        # self.df = self.df.append(test_acc._asdict(), ignore_index=True)

    def print_data(self, results):
        print(f"Train loss: {results.train_loss:.3f} "
              f"Test loss: {results.test_loss:.3f} | "
              f"Acc: {results.test_acc:.3f} | "
              f"Time: {results.ellapsed_time:.2f}s | "
              f"lr: {results.lr:.5f}")

    def save_data(self):
        with open(f"{self.path}/results.json", 'w') as fp:
            json.dump(self.exp_results, fp)
        # self.plot(print_avg=True)

    def plot(self, print_avg=False, exp_id=None):
        """Todo: Need to fix this for plotting!"""
        if print_avg:
            tmp = deepcopy(self.exp_results)
            path = self.path
        else:
            if exp_id is None:
                raise RuntimeError
            else:
                tmp = deepcopy(self.exp_results[exp_id])
                path = f"{self.path}/{exp_id}"

        for _x in xs:

            if _x == 'sparsity' and len(np.unique(tmp[_x])) < 2:
                continue

            for ret in rets:
                ax = plt.plot(tmp[_x], tmp[ret], color=self.cmap[0])
                ax = plt.plot(tmp[_x], tmp[ret], color=self.cmap[0])
                # ax = sns.lineplot(x=_x, y=ret,  hue='Legend', style='Legend', markers=True, data=tmp[tmp[ret].notna()])
                # if 'reverse' in self.args.experiment_name and _x == 'sparsity':
                #     ax.invert_xaxis()
                #
                # if 'sparsity' == _x:
                #     plt.xlabel(f"{_x} (%)")
                # elif 'Cost' == _x:
                #     plt.xlabel(f"{_x} (Mbytes)")
                # else:
                #     plt.xlabel(f"{_x}")
                #
                # if 'ACC' in ret:
                #     plt.ylabel(f"{ret}(%)")
                #     plt.ylim(40, 100)
                # elif 'Loss' in ret:
                #     plt.ylabel(f"{ret}")
                #     plt.ylim(0, 1)

                plt.title(f"Dataset: {self.args.dataset} | Model: {self.args.model}")
                plt.tight_layout()
                plt.savefig(f'{path}/{_x}_{ret}.png')
                print(f"\033[91mSaved {path}/{_x}_{ret}.png\033[00m")
                # plt.show()
                plt.close()

    # def global_plot(self):
    #     raise RuntimeError
    #     file = f"{self.root}/results.csv"
    #     df = pd.read_csv(file)
    #     df["legend"] = "[" + df['exp_name'] + "] " + df['Legend']
    #
    #     acc_df = df[df['ACC'].notna()]
    #     loss_df = df[df['Loss'].notna()]
    #
    #     for _x in xs:
    #         for ret in rets:
    #
    #             if 'sparsity' == _x:
    #                 plt.xlabel(f"{_x} (%)")
    #             elif 'Cost' == _x:
    #                 plt.xlabel(f"{_x} (Mbytes)")
    #             else:
    #                 plt.xlabel(f"{_x}")
    #
    #             if 'Loss' == ret:
    #                 data = loss_df
    #             else:
    #                 data = acc_df
    #
    #             if _x == 'sparsity':
    #                 data = data[~(data['exp_name'] == 'baseline')]
    #
    #             sns.lineplot(x=_x, y=ret, hue='legend', style='legend', markers=True, data=data)
    #             plt.savefig(f"{self.root}/{_x}_{ret}.png")
    #             plt.close()
    #
    #     print("")

    def add_results(self, results):
        if results.exp_id not in self.exp_results.keys():
            self.exp_results[results.exp_id] = self.make_basic_dict()

        self.exp_results[results.exp_id]['round'].append(results.round)
        self.exp_results[results.exp_id]['cost'].append(results.cost)
        self.exp_results[results.exp_id]['train_loss'].append(results.train_loss)
        self.exp_results[results.exp_id]['test_loss'].append(results.test_loss)
        self.exp_results[results.exp_id]['test_acc'].append(results.test_acc)
        self.exp_results[results.exp_id]['sparsity'].append(results.sparsity)

    def make_basic_dict(self):
        return {'round': [],
                'cost': [],
                'train_loss': [],
                'test_loss': [],
                'test_acc': [],
                'sparsity': [],
                }

    def save_yaml(self):
        f_name = f'{self.path}/exp_config.yaml'
        with open(f_name, 'w') as outfile:
            yaml.dump(self.args, outfile, default_flow_style=False)
