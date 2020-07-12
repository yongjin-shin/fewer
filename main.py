import numpy as np
import random
import torch

# Related Class and functions
from server import Server
from data import Preprocessor
from misc import read_argv
from logger import Logger
import warnings
warnings.filterwarnings('ignore')


def main():
    args = read_argv()
    logger = Logger()
    logger.get_args(args)

    for i in range(args.nb_exp_reps):
        model = single_experiment(args, i, logger)
        logger.save_model(param=model.state_dict(), exp_id=i)
        # logger.plot(exp_id=i)

    logger.save_data()
    logger.save_yaml()


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
