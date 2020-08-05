import torch.nn as nn
import torch

from resnet_cifar import *
from vgg_cifar import *


def create_nets(args, location):
    print(f"{location}: ", end="", flush=True)

    if 'mnist' in args.dataset:
        _in_dim = 1
    elif 'cifar' in args.dataset:
        _in_dim = 3
    else:
        raise NotImplementedError

    if args.model == 'mlp':
        model = MLP(784, args.hidden, 10)
    elif args.model == 'deep_mlp':
        model = DeepMLP(3072, args.hidden, 10)
    elif args.model == 'mnistcnn':
        model = MnistCNN(1, 10)
    elif args.model == 'cifarcnn':
        model = CifarCnn(3, 10)
    elif args.model == 'cifarcnnlarge':
        model = CifarCnnLarge(3, 10)
    elif args.model == 'cifarcnnslim':
        model = CifarCnnSlim(3, 10)
    elif args.model == 'testcnn':
        model = TestCNN(_in_dim, 10)
    elif args.model == 'vgg11':
        model = vgg11()
    elif args.model == 'vgg11_original':
        model = vgg11(m_type='original')
    elif args.model == 'vgg11_slim':
        model = vgg11(m_type='slim')
    elif args.model == 'res8':
        model = resnet8(num_classes=10)
    else:
        raise NotImplementedError

    return model


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.elu = nn.ELU()
        # self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        print("MLP was made")

    def forward(self, x):
        x = x.view((x.size()[0], -1))
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x


class DeepMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, dim_hidden)
        self.fc4 = nn.Linear(dim_hidden, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, dim_out)
        self.elu = nn.ELU()
        print("DeepMLP was made")

    def forward(self, x):
        x = x.view((x.size()[0], -1))
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        x = self.elu(self.fc6(x))
        x = self.elu(self.fc7(x))
        x = self.fc8(x)
        return x


# Exactly same model for FedAvg
class MnistCNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MnistCNN, self).__init__()
        self.dim_in = dim_in
        self.conv1 = nn.Conv2d(dim_in, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, dim_out)
        self.relu = nn.ReLU()
        print("MnistCNN was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CifarCnn(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, dim_out)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        print("CifarCnn was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CifarCnnSlim(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CifarCnnSlim, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*5*5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, dim_out)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        print("CifarCnn was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CifarCnnLarge(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CifarCnnLarge, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, dim_out)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        print("CifarCnn was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Testing model for pruning operations
class TestCNN(nn.Module):
    def __init__(self, dim_in=3, dim_out=10):
        super(TestCNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv1 = nn.Conv2d(dim_in, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.elu = nn.ELU()
        self.fc = nn.Linear(64, dim_out)
        print("TestCNN was made")
        
    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
