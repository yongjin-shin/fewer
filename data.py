from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import deque
import numpy as np
from collections import defaultdict

__all__ = ['Preprocessor']


class Preprocessor:
    def __init__(self, args):
        self.args = args

    def distribute_data(self, server):
        """ 서버에게 Train, Test 데이터 전달함 """

        dataset_train, dataset_test = self.download_dataset()
        dataset_real_test, dataset_valid, dataset_locals = self.make_data_for_local_and_server(dataset_train,
                                                                                               dataset_test,
                                                                                               self.args.data_hetero_alg)
        server.get_data(dataset_valid, dataset_locals, dataset_real_test)

    def download_dataset(self):
        print(f"Get data: {self.args.dataset}...")
        path = f'./dataset/{self.args.dataset}/'
        dataset_train, dataset_test = None, None

        if self.args.dataset == 'mnist':
            train_transform, test_transform = self.mnist_data_augmentation()
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset_train = datasets.MNIST(path, train=True, transform=train_transform, download=True)
            dataset_test = datasets.MNIST(path, train=False, transform=test_transform, download=True)
            
        elif self.args.dataset == 'fmnist':
            train_transform, test_transform = self.fashion_mnist_data_augmentation()
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset_train = datasets.FashionMNIST(path, train=True, transform=train_transform, download=True)
            dataset_test = datasets.FashionMNIST(path, train=False, transform=test_transform, download=True)
            
        elif self.args.dataset == 'cifar10':
            train_transform, test_transform = self.cifar_data_augmentation()
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset_train = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
            dataset_test = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
            
        else:
            exit('Error: unrecognized dataset')

        return dataset_train, dataset_test

    def make_data_for_local_and_server(self, dataset_train, dataset_test, hetero_alg):
        """서버와 데이터에게 얼마나 데이터 분배할지 결정. Train 데이터만 사용한다"""

        # non iid 데이터를 만들어준다
        locals_idx = self.make_non_iid(dataset_train.targets, hetero_alg)

        # 서버의 데이터
        # 일단 Test 데이터에서 각 class 별 index를 뽑아냄
        all_test_targets = dataset_test.targets
        label_by_idx = {}
        unique_target = np.unique(all_test_targets)
        for ut in unique_target:
            label_by_idx[ut] = np.where(all_test_targets == ut)[0]

        # 앞에서 뽑아낸 class 별 index에서 각 class마다 nb_server_data만큼 validation set으로 뽑아냄
        server_idx = []
        if self.args.nb_server_data > 0:
            for ut in unique_target:
                ut_idx = np.random.choice(label_by_idx[ut], self.args.nb_server_data, replace=False)
                server_idx.extend(list(ut_idx))

        # Valdiation을 제외한 나머지는 Test로 들어감!
        test_idx = list(set(np.arange(len(all_test_targets))) - set(server_idx))

        # Valdiation은 혹시 모르니 한번 섞어줬음.
        np.random.shuffle(server_idx)
        dataset_valid = Subset(dataset_test, server_idx)
        dataset_real_test = Subset(dataset_test, test_idx)

        # 로컬의 데이터
        datasets_local = []
        for i in range(self.args.nb_devices):
            local_idx = locals_idx.pop()
            datasets_local.append(Subset(dataset_train, local_idx))

        print(f"\nDataset Length\n"
              f" Center length: {dataset_valid.__len__()}\n"
              f" Test length: {dataset_real_test.__len__()}\n"
              f" Local length: {len(datasets_local)} x {datasets_local[0].__len__()}\n")
        
        return dataset_real_test, dataset_valid, datasets_local

    def make_non_iid(self, labels, alg):
        length = int(len(labels) / self.args.nb_devices)
        # length = int((len(labels) - self.args.nb_server_data) / self.args.nb_devices)
        idx = []
        
        # non-iid로 만들 필요가 없는 경우
        if self.args.iid or self.args.nb_devices == 1:
            tot_idx = np.arange(len(labels))
            for _ in range(self.args.nb_devices):
                idx.append(tot_idx[:length])
                tot_idx = tot_idx[length:]
        
        # non-iid으로 만들어야 하는 경우
        else:  
            # Todo: Unbalanced dataset을 만들기 위해서는 아래 num_shards를 수정해야 함
            # 하나의 local은 nb_max_classes만큼 unique한 class를 가져갈 거임
            # 다만, 각 class의 개수는 num_shards만큼 동일함.
            shard_size = int(length / self.args.nb_max_classes)  # e.g. 300 = 600 / 2
            unique_classes = np.unique(labels)

            tot_idx_by_label = []  # shape: class x num_shards x shard_size
            for i in unique_classes:
                idx_by_label = np.where(labels == i)[0]
                tmp = []
                while 1:
                    tmp.append(idx_by_label[:shard_size])
                    idx_by_label = idx_by_label[shard_size:]
                    if len(idx_by_label) < shard_size/2:
                        break
                tot_idx_by_label.append(tmp)

            if 'fedavg' == alg:
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

            elif 'fedma' == alg:
                idx_batch = [[] for _ in range(self.args.nb_devices)]
                idx = [defaultdict(list) for _ in range(self.args.nb_devices)]
                for it, k in enumerate(unique_classes):
                    this_labels = np.concatenate(tot_idx_by_label[it])
                    prop = np.random.dirichlet([0.2 for _ in range(self.args.nb_devices)])
                    prop = np.array([p * (len(idx_j) < length)
                                     for p, idx_j in zip(prop, idx_batch)])
                    prop = prop / prop.sum()
                    prop = (prop * len(this_labels)).astype(int).cumsum()[:-1]
                    label_by_device = np.split(this_labels, prop)
                    for device_id, lb in enumerate(label_by_device):
                        idx_batch[device_id] += lb.copy().tolist()
                        idx[device_id][k] = lb.copy().tolist()

                from termcolor import colored
                print(colored(f'{"Tot":5s}', 'red'), end='')
                for i in range(10):
                    print(f"{i:5d}", end='')
                print('\n')

                for i in range(self.args.nb_devices):
                    print(colored(f"{len(idx_batch[i]):5d}", 'red'), end='')
                    for k in idx[i].keys():
                        print(f"{len(idx[i][k]):5d}", end='')
                    print('\n')

                idx = idx_batch

            else:
                raise RuntimeError

        # remained_idx = set(np.arange(len(labels))) - set(np.concatenate(idx))
        # server_idx = np.random.choice(list(remained_idx), size=self.args.nb_server_data)
        return deque(idx)

    def mnist_data_augmentation(self):
        mean = [0.1307]
        stdv = [0.3081]
        train_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ]

        train_transforms = transforms.Compose(train_transform_list)
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        return train_transforms, test_transforms

    def fashion_mnist_data_augmentation(self):
        mean = [0.5]
        stdv = [0.5]
        train_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ]

        train_transforms = transforms.Compose(train_transform_list)
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        return train_transforms, test_transforms

    def cifar_data_augmentation(self):
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
