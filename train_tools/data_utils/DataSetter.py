from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict, deque
from termcolor import colored
import numpy as np
import os

__all__ = ['DataSetter']


DATASIZE = {
    'mnist' : 28, 'fashion-mnist': 28, 'cifar10': 32, 'cifar100' : 32, 'tiny-imagenet': 64
}
DATASTAT = {
    'mnist' : {'mean': [0.1307], 'std': [0.3081]},
    'fashion-mnist': {'mean': [0.5], 'std': [0.5]},
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'cifar100' : {'mean': [0.5071, 0.4865, 0.4409], 'std': [0.1980, 0.2010, 0.1970]},
    'tiny-imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
}

CLASSNUM = {    
    'mnist' : 10, 'fashion-mnist': 10, 'cifar10': 10, 'cifar100' : 100, 'tiny-imagenet': 200
}


class DataSetter():
    """
    Assigns data to local clients.
    """
    def __init__(self, root='./data', dataset='cifar10'):
        self.root = os.path.join(root, dataset)
        self.dataset = dataset
        self.num_classes = CLASSNUM[dataset]
        self.default_transform = {'train': 
                                  transforms.Compose([
                                      transforms.RandomCrop(DATASIZE[dataset], padding=DATASIZE[dataset]//8),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=DATASTAT[dataset]['mean'], 
                                                           std=DATASTAT[dataset]['std'])]),
                                  'test': 
                                  transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=DATASTAT[dataset]['mean'], 
                                                           std=DATASTAT[dataset]['std'])])
                                 }


    def data_distributer(self, n_clients=100, alg='fedavg', max_class_num=2, dir_alpha=0.5, 
                         train_transform=None, test_transform=None, as_dict=True):
        """서버와 데이터에게 얼마나 데이터 분배할지 결정. Train 데이터만 사용한다"""
        
        trainset, testset = self._dataset_getter(train_transform, test_transform)
            
        # non iid 데이터를 만들어준다
        locals_idx = self._data_processor(trainset, n_clients, alg, max_class_num, dir_alpha)

        # 서버의 데이터
        # 일단 Test 데이터에서 각 class 별 index를 뽑아냄
        all_test_targets = testset.targets
        label_by_idx = {}
        unique_target = np.unique(all_test_targets)
        for ut in unique_target:
            label_by_idx[ut] = np.where(all_test_targets == ut)[0]

        # 앞에서 뽑아낸 class 별 index에서 각 class마다 nb_server_data만큼 validation set으로 뽑아냄
        server_idx = []
        if n_clients > 0:
            for ut in unique_target:
                ut_idx = np.random.choice(label_by_idx[ut], n_clients, replace=False)
                server_idx.extend(list(ut_idx))

        # Valdiation을 제외한 나머지는 Test로 들어감!
        test_idx = list(set(np.arange(len(all_test_targets))) - set(server_idx))

        # Valdiation은 혹시 모르니 한번 섞어줬음.
        np.random.shuffle(server_idx)
        validset = Subset(testset, server_idx)
        testset = Subset(testset, test_idx)

        # 로컬의 데이터
        local_trainset = []
        for i in range(n_clients):
            local_idx = locals_idx.pop()
            local_trainset.append(Subset(trainset, local_idx))

        print(f"\nDataset Length\n"
              f" Center length: {validset.__len__()}\n"
              f" Test length: {testset.__len__()}\n"
              f" Local length: {len(local_trainset)} x {local_trainset[0].__len__()}\n")
        
        if as_dict:
            return {'valid': validset, 'local': local_trainset, 'test': testset}
        
        return validset, local_trainset, testset
                
            
    def _dataset_getter(self, train_transform=None, test_transform=None):
        """
        Make datasets to build 
        """
        if train_transform is None:
            train_transform = self.default_transform['train']
            
        if test_transform is None:
            test_transform = self.default_transform['test']
        
        if self.dataset == 'mnist':
            trainset = datasets.MNIST(self.root, train=True, transform=train_transform, download=True)
            testset = datasets.MNIST(self.root, train=False, transform=test_transform, download=True)
            
        if self.dataset == 'fashion-mnist':
            trainset = datasets.FashionMNIST(self.root, train=True, transform=train_transform, download=True)
            testset = datasets.FashionMNIST(self.root, train=False, transform=test_transform, download=True)            
            
        if self.dataset == 'cifar10':
            trainset = datasets.CIFAR10(self.root, train=True, transform=train_transform, download=True)
            testset = datasets.CIFAR10(self.root, train=False, transform=test_transform, download=True)            
            
        if self.dataset == 'cifar100':
            trainset = datasets.CIFAR100(self.root, train=True, transform=train_transform, download=True)
            testset = datasets.CIFAR100(self.root, train=False, transform=test_transform, download=True)            
        
        # To be implemented
        #if self.dataset == 'tiny-imagenet':
        #    trainset = TinyImageNet(self.root, train=True, transform=train_transform, download=True)
        #    testset = TinyImageNet(self.root, train=False, transform=test_transform, download=True)               
        
        return trainset, testset
        
        
    def _data_processor(self, trainset, n_clients=100, alg='fedavg', max_class_num=2, dir_alpha=0.5):
        labels = trainset.targets
        length = int(len(labels) / n_clients)
        idx = []
        
        # non-iid로 만들 필요가 없는 경우
        if alg == 'iid' or n_clients == 1:
            tot_idx = np.arange(len(labels))
            for _ in range(n_clients):
                idx.append(tot_idx[:length])
                tot_idx = tot_idx[length:]
            
            return deque(idx)
        
        else:
            # Todo: Unbalanced dataset을 만들기 위해서는 아래 num_shards를 수정해야 함
            # 하나의 local은 nb_max_classes만큼 unique한 class를 가져갈 거임
            # 다만, 각 class의 개수는 num_shards만큼 동일함.
            shard_size = int(length / max_class_num)  # e.g. 300 = 600 / 2
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
                for _ in range(n_clients):
                    idx_by_devices = []
                    while len(idx_by_devices) < max_class_num:
                        chosen_label = np.random.choice(unique_classes, 1, replace=False)[0]  # 임의의 Label을 하나 뽑음
                        if len(tot_idx_by_label[chosen_label]) > 0:  # 만약 해당 Label의 shard가 하나라도 남아있다면,
                            l_idx = np.random.choice(len(tot_idx_by_label[chosen_label]), 1, replace=False)[0]  # shard 중 일부를 하나 뽑고
                            idx_by_devices.append(tot_idx_by_label[chosen_label][l_idx].tolist())  # 클라이언트에 넣어준다.
                            del tot_idx_by_label[chosen_label][l_idx]  # 뽑힌 shard의 원본은 제거!
                    idx.append(np.concatenate(idx_by_devices))
                    
            elif 'fedma' == alg:
                idx_batch = [[] for _ in range(n_clients)]
                idx = [defaultdict(list) for _ in range(n_clients)]
                for it, k in enumerate(unique_classes):
                    this_labels = np.concatenate(tot_idx_by_label[it])
                    prop = np.random.dirichlet([dir_alpha for _ in range(n_clients)])
                    prop = np.array([p * (len(idx_j) < length)
                                     for p, idx_j in zip(prop, idx_batch)])
                    prop = prop / prop.sum()
                    prop = (prop * len(this_labels)).astype(int).cumsum()[:-1]
                    label_by_device = np.split(this_labels, prop)
                    for device_id, lb in enumerate(label_by_device):
                        idx_batch[device_id] += lb.copy().tolist()
                        idx[device_id][k] = lb.copy().tolist()

                print(colored(f'{"Tot":5s}', 'red'), end='')
                for i in range(10):
                    print(f"{i:5d}", end='')
                print('\n')

                for i in range(n_clients):
                    print(colored(f"{len(idx_batch[i]):5d}", 'red'), end='')
                    for k in idx[i].keys():
                        print(f"{len(idx[i][k]):5d}", end='')
                    print('\n')

                idx = idx_batch

            else:
                raise RuntimeError

        return deque(idx)                     
