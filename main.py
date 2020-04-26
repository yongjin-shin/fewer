from pathlib import Path
import numpy as np
import random
import torch
import yaml

# Related Class and functions
from server import Server
from data import Preprocessor
from misc import save_results, fix_arguments


def single_experiment(args, i):
    print(f'======================{args.dataset} exp: {i + 1}====================')
    np.random.seed(int(args.seed + i))  # for the reproducibility
    torch.manual_seed(int(args.seed + i))
    random.seed(int(args.seed + i))
    args.seed += 1

    data = Preprocessor(args)  # Data 불러오는 곳
    server = Server(args)  # 서버와 로컬이 만들어짐
    data.distribute_data(server)  # 서버에 데이터를 전달함. 동시에 로컬에 데이터 분배됨.
    server.make_model()  # Base Model을 만듦.
    results = server.train(exp_id=i)  # 서버 Training 진행한 후, Global Model의 테스트 결과 받아옴
    print("\n")

    return results


def main():
    args = yaml.load(stream=open("config/config.yaml"), Loader=yaml.FullLoader)
    args = fix_arguments(args)
    path = f'./log/{args.dataset}/ep{args.local_ep}'
    Path(path).mkdir(parents=True, exist_ok=True)

    # 반복실험을 합니다
    all_exps, idx = [], 0
    for i in range(args.nb_exp_reps):
        all_exps.append(single_experiment(args, i))

    # 결과를 저장합니다
    save_results(path, args, all_exps)


if __name__ == '__main__':
    main()