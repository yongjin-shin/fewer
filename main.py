import numpy as np
from server import Server
from data import Preprocessor
from utils import *
import torch, gc, warnings, random

warnings.filterwarnings('ignore')


def main():
    args = read_argv()
    logger = Logger()
    logger.get_args(args)
    logger.save_yaml()

    for i in range(args.nb_exp_reps):
        model = single_experiment(args, i, logger)
        logger.save_model(param=model.state_dict(), exp_id=i)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    logger.save_data()


def single_experiment(args, i, logger):
    print(f'\033[91m======================{args.dataset} exp: {i}====================\033[00m')
    np.random.seed(int(args.seed * (i+1)))  # for the reproducibility
    random.seed(int(args.seed * (i+1)))
    torch.manual_seed(int(args.seed * (i+1)))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.seed += 1

    server = Server(args, logger)  # 서버와 로컬이 만들어짐
    data = Preprocessor(args)  # Data 불러오는 곳
    data.distribute_data(server)  # 서버에 데이터를 전달함
    server.train(exp_id=i)  # 서버 Training 진행함
    print("\033[91m=\033[00m" * 50 + "\n")
    ret_model = server.get_global_model()
    del data
    del server
    gc.collect()
    torch.cuda.empty_cache()

    logger.save_data()
    return ret_model


if __name__ == '__main__':
    main()