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
import argparse


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='For Multiple experiments')

parser.add_argument('--config_file', default='config.yaml', type=str)
parser.add_argument('--nb_exp_reps', default=1, type=int)
parser.add_argument('--nb_devices', default=100, type=int)
parser.add_argument('--lr', default=0.15, type=float)
parser.add_argument('--model', default='mnistcnn', type=str)
parser.add_argument('--pruning_type', default='baseline', type=str)
parser.add_argument('--plan_type',default='base',type=str)

additional_args = parser.parse_args()


def main():
    # yaml_file = read_argv(sys.argv)
    yaml_file = additional_args.config_file
    try:
        args = yaml.load(stream=open(f"config/{yaml_file}"), Loader=yaml.FullLoader)
    except:
        args = yaml.load(stream=open(f"config/{yaml_file}", 'rt', encoding='utf8'), Loader=yaml.FullLoader)

    args = fix_arguments(args)
    args.model = additional_args.model

    if 'mnist' in args.model:
        args.dataset = 'mnist'
        args.nb_rounds = 150
        args.iid = False
        args.plan = [0,130,20]
    elif 'cifar' in args.model:
        args.dataset = 'cifar10'
        args.nb_rounds = 300
        args.iid = True
        args.plan = [0, 280, 20]

    args.nb_devices = additional_args.nb_devices
    args.nb_exp_reps = additional_args.nb_exp_reps
    #pruning_type = 'server_pruning','local_pruning', 'local_pruning_half'
    args.pruning_type = additional_args.pruning_type
    args.experiment_name= args.pruning_type
    args.plan_type = additional_args.plan_type

    if args.pruning_type == 'server_pruning' and args.plan_type == 'reverse':
        if args.dataset == 'mnist':
            args.plan = [20,110,20]
        elif args.dataset == 'cifar10':
            args.plan = [20,260,20]
        args.experiment_name = 'server_pruning_reverse'
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
