import os
import numpy as np
import random
import torch
import yaml

# Related Class and functions
from server import Server
from data import Preprocessor
from misc import save_results, fix_arguments
from logger import Logger

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


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
    print("\033[91m=\033[00m"*50 + "\n")

    return server.get_global_model()


def main():
    try:
        args = yaml.load(stream=open("config/config.yaml"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open("config/config.yaml", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = fix_arguments(args)
    # path = f'./log/{args.dataset}/{args.experiment_name}'
    # Path(path).mkdir(parents=True, exist_ok=True)
    logger = Logger(args)
    # models = args.models

    # 반복실험을 합니다
    for i in range(args.nb_exp_reps):
        model = single_experiment(args, i, logger)
        logger.save_model(param=model.state_dict(), exp_id=i)
        logger.plot(exp_id=i)
        # torch.save(model.state_dict(), os.path.join(path, 'model%d.h5'%i))

    # 결과를 저장합니다
    logger.save_data()


if __name__ == '__main__':
    main()
