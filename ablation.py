import numpy as np
import random
import torch
import yaml
import sys
import datetime

# Related Class and functions
from server import Server
from data import Preprocessor
from misc import fix_arguments, read_argv
from logger import Logger
from argparse import Namespace
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='For Multiple experiments')
parser.add_argument('--config_file', default='config.yaml', type=str)
parser.add_argument('--nb_exp_reps', default=3, type=int)
parser.add_argument('--nb_devices', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--model', default='mnistcnn', type=str)
parser.add_argument('--pruning_type', default='baseline', type=str)
parser.add_argument('--plan_type', default='base', type=str)
parser.add_argument('--decay_type', default='gradual', type=str)
parser.add_argument('--device', default='cuda', type=str)
additional_args = parser.parse_args()


def main():
    # yaml_file = read_argv(sys.argv)
    yaml_file = additional_args.config_file
    try:
        args = yaml.load(stream=open(f"config/{yaml_file}"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open(f"config/{yaml_file}", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = Namespace(**args)
    # args = fix_arguments(args)
    args.model = additional_args.model
    args.nb_devices = additional_args.nb_devices
    args.nb_exp_reps = additional_args.nb_exp_reps
    args.pruning_type = additional_args.pruning_type
    args.plan_type = additional_args.plan_type
    args.decay_type = additional_args.decay_type
    args.device = additional_args.device
        
    #how to change other things for fair comparison??
    args.lr = additional_args.lr

    logger = Logger()
    logger.get_args(args)

    # 반복실험을 합니다
    for i in range(args.nb_exp_reps):
        model = single_experiment(args, i, logger)
        logger.save_model(param=model.state_dict(), exp_id=i)
        logger.plot(exp_id=i)

    # 결과를 저장합니다
    logger.save_data()
    logger.save_yaml()
    # logger.global_plot(files)


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
