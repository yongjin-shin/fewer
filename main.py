import os
import numpy as np
import random
import torch
import yaml
import sys
import os
import datetime

# Related Class and functions
from server import Server
from data import Preprocessor
from misc import fix_arguments
from logger import Logger

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def main():
    _file = sys.argv[-1]
    if not 'main.py' in _file:
        files = []
        for _, _, folder in os.walk('./config/settings'):
            for f in folder:
                if '.yaml' in f:
                    files.append(f'settings/{f}')
    else:
        files = ['config.yaml']

    _time = datetime.datetime.now()
    for _file in np.sort(files):
        try:
            args = yaml.load(stream=open(f"config/{_file}"), Loader=yaml.FullLoader)
        except:
            args = yaml.load(stream=open(f"config/{_file}", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

        args = fix_arguments(args)
        logger = Logger(args, _time=_time)

        # 반복실험을 합니다
        for i in range(args.nb_exp_reps):
            model = single_experiment(args, i, logger)
            logger.save_model(param=model.state_dict(), exp_id=i)
            logger.plot(exp_id=i)

        # 결과를 저장합니다
        logger.save_data()


def single_experiment(args, i, logger):
    print(f'\033[91m======================{args.dataset} exp: {i + 1}====================\033[00m')
    np.random.seed(int(args.seed + i))  # for the reproducibility
    random.seed(int(args.seed + i))
    torch.manual_seed(int(args.seed + i))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.seed += 1

    server = Server(args, logger)  # 서버와 로컬이 만들어짐
    data = Preprocessor(args)  # Data 불러오는 곳
    data.distribute_data(server)  # 서버에 데이터를 전달함
    server.train(exp_id=i)  # 서버 Training 진행함
    print("\033[91m=\033[00m" * 50 + "\n")

    return server.get_global_model()


if __name__ == '__main__':
    main()
