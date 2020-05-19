from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import deque
import numpy as np


class Mydataset(Dataset):
    def __init__(self, dataset, args):
        self.args = args
        self.x = dataset['x']
        self.y = dataset['y']

        if 'cnn' in self.args.model:
            if 'mnist' in self.args.dataset:
                self.x = self.x.reshape((-1, 1, 28, 28))
            elif 'cifar' in self.args.dataset:
                self.x = self.x.permute(0, 3, 1, 2)
            else:
                raise NotImplementedError

        self.std, self.mean = torch.std_mean(self.x.float())

    def unique(self):
        return len(torch.unique(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx] - self.mean)/self.std, self.y[idx]


def get_simple_dataset():
    """기본 데이터셋 구조"""
    return {'x': [], 'y': []}


class Preprocessor:
    def __init__(self, args):
        self.args = args

    def distribute_data(self, server):
        """ 서버에게 Train, Test 데이터 전달함 """

        dataset_train, dataset_test = self.download_dataset()
        if 'mnist' in self.args.dataset:
            train = {'x': dataset_train.train_data.to(self.args.device),
                     'y': dataset_train.train_labels.to(self.args.device)}
            test = {'x': dataset_test.test_data.to(self.args.device),
                    'y': dataset_test.test_labels.to(self.args.device)}
        elif 'cifar' in self.args.dataset:
            train = {'x': dataset_train.data.to(self.args.device),
                     'y': dataset_train.targets.to(self.args.device)}
            test = {'x': dataset_test.data.to(self.args.device),
                    'y': dataset_test.targets.to(self.args.device)}
        else:
            raise NotImplementedError

        dataset_server, dataset_locals = self.make_data_for_local_and_server(train)
        server.get_data(dataset_server, dataset_locals, test)

    def make_data_for_local_and_server(self, train):
        """서버와 데이터에게 얼마나 데이터 분배할지 결정. Train 데이터만 사용한다"""

        # 서버의 데이터
        len_data_server = self.args.nb_server_data
        data_server = {'x': train['x'][:len_data_server],
                       'y': train['y'][:len_data_server]} if len_data_server > 0 else get_simple_dataset()

        # 로컬의 데이터
        tot_len_data_local = (len(train['x']) - len_data_server)
        data_locals = self.split_data_for_locals(tot_len_data_local, train)

        print(f"\nDataset Length\n"
              f" Center length: {len(data_server['y'])}\n"
              f" Local length: {len(data_locals[0]['y'])} x {len(data_locals)}\n")
        return data_server, data_locals

    def split_data_for_locals(self, tot_len_data_local, train):
        data_locals = []
        len_data_local = int(tot_len_data_local / self.args.nb_devices)  # e.g. 600 = 60,000 / 100

        # non iid 데이터를 만들어준다
        distributed_idx = self.make_non_iid(train['y'].cpu().numpy(), len_data_local)
        for i in range(self.args.nb_devices):
            local_idx = distributed_idx.pop()
            data_locals.append({'x': train['x'][local_idx], 'y': train['y'][local_idx]})
            # print(torch.unique(train['y'][local_idx]))
        return data_locals

    def make_non_iid(self, label, length):
        idx = []

        # non-iid로 만들 필요가 없는 경우
        if self.args.iid or self.args.nb_devices == 1:
            tot_idx = np.arange(len(label))
            for _ in range(self.args.nb_devices):
                idx.append(tot_idx[:length])
                tot_idx = tot_idx[length:]

        else:  # non-iid으로 만들어야 하는 경우
            # Todo: Unbalanced dataset을 만들기 위해서는 아래 num_shards를 수정해야 함
            # 하나의 local은 nb_max_classes만큼 unique한 class를 가져갈 거임
            # 다만, 각 class의 개수는 num_shards만큼 동일함.
            shard_size = int(length / self.args.nb_max_classes)  # e.g. 300 = 600 / 2
            unique_classes = np.unique(label)

            tot_idx_by_label = []  # shape: class x num_shards x shard_size
            for i in unique_classes:
                idx_by_label = np.where(label == i)[0]
                tmp = []
                while len(idx_by_label) > 0:
                    tmp.append(idx_by_label[:shard_size])
                    idx_by_label = idx_by_label[shard_size:]
                tot_idx_by_label.append(tmp)

            # 각 client 별로 randomly 각기 다른 class를 뽑음
            for _ in range(self.args.nb_devices):
                idx_by_devices = []
                while len(idx_by_devices) < self.args.nb_max_classes:
                    chosen_label = np.random.choice(unique_classes, 1, replace=False)[0]  # 임의의 Label을 하나 뽑음
                    if len(tot_idx_by_label[chosen_label]) > 0:  # 만약 해당 Label의 shard가 하나라도 남아있다면,
                        l_idx = np.random.choice(len(tot_idx_by_label[chosen_label]), 1, replace=False)[0]  # shard 중 일부를 하나 뽑고
                        idx_by_devices.append(tot_idx_by_label[chosen_label][l_idx].tolist())  # 클라이언트에 넣어준다.
                        del tot_idx_by_label[chosen_label][l_idx]  # 뽑힌 shard의 원본은 제거!
                idx.append(np.concatenate(idx_by_devices))

        return deque(idx)

    def download_dataset(self):
        print(f"Get data: {self.args.dataset}...")
        path = f'./dataset/{self.args.dataset}/'
        dataset_train, dataset_test = None, None

        if self.args.dataset == 'mnist':
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset_train = datasets.MNIST(path, train=True, download=True)
            dataset_test = datasets.MNIST(path, train=False, download=True)
        elif self.args.dataset == 'fmnist':
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset_train = datasets.FashionMNIST(path, train=True, download=True)
            dataset_test = datasets.FashionMNIST(path, train=False, download=True)
        elif self.args.dataset == 'cifar10':
            train_transform, test_transform = self._data_argumentation()
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset_train = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
            dataset_test = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
        else:
            exit('Error: unrecognized dataset')

        return dataset_train, dataset_test

    def _data_argumentation(self):
        mean = [0.4914, 0.4822, 0.4465]
        stdv = [0.2023, 0.1994, 0.2010]
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ]

        train_transforms = transforms.Compose(train_transform_list)
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        return train_transforms, test_transforms

