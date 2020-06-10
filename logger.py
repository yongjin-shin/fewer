import numpy as np
import datetime
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

rets = ['train_loss', 'test_loss', 'test_acc']
xs = ['sparcity', 'round']
_items = rets + xs + ['exp_id']
Results = namedtuple('Results', _items)


class Logger:
    def __init__(self, args):
        self.args = args
        self.tot_sparsity = np.matmul(args.pruning_pack, args.pruning_plan)
        self.df = pd.DataFrame(columns=_items)
        self._pointer = 0

        # Create saving folder
        self.path = f"./log/{args.dataset}/[Fed]model_{args.model}_round_{args.nb_rounds}" \
                    f"_client{args.nb_devices}_localstep{args.local_ep}" \
                    f"_localbs_{args.local_bs}_sparsity{self.tot_sparsity:.3f}_iid_{args.iid}" \
                    f"_{datetime.datetime.now()}"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        for k in sorted(vars(args).keys()):
            print("{}: {}".format(k, vars(args)[k]))
        print(f"\033[91mPath: {self.path}\033[00m")

    def save_model(self, param, exp_id):
        Path(f"{self.path}/{exp_id}").mkdir(parents=True, exist_ok=True)
        torch.save(param, f"{self.path}/{exp_id}/model.h5")

    def get_results(self, results):
        self.df = self.df.append(results._asdict(), ignore_index=True)
        self.print_data(results)

    def print_data(self, results):
        print(f"Train loss: {results.train_loss:.3f} "
              f"Test loss: {results.test_loss:.3f} | "
              f"acc: {results.test_acc:.3f}")

    def save_data(self):
        self.df.to_csv(f"{self.path}/results.csv")
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
            for ret in rets:
                sns.lineplot(x=_x, y=ret, data=tmp)

                if 'sparcity' == _x:
                    plt.xlabel(f"{_x}(%)")
                else:
                    plt.xlabel(f"{_x}")

                if 'acc' in ret:
                    plt.ylabel(f"{ret}(%)")
                    plt.ylim(50, 100)
                elif 'loss' in ret:
                    plt.ylabel(f"{ret}")
                    plt.ylim(0, 0.5)

                plt.title(f"Dataset: {self.args.dataset} | Model: {self.args.model}")
                plt.tight_layout()
                plt.savefig(f'{path}/{_x}_{ret}.png')
                print(f"\033[91mSaved {path}/{_x}_{ret}.png\033[00m")
                plt.show()
                plt.close()


if __name__ == '__main__':
    import yaml
    from misc import fix_arguments

    try:
        args = yaml.load(stream=open("config/config.yaml"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open("config/config.yaml", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = fix_arguments(args)
    logger = Logger(args)
    print("debugging")
