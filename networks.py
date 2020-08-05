import torch.nn as nn
import torch


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
        model = DeepMLP(784, args.hidden, 10)
    elif args.model == 'mnistcnn':
        model = MnistCNN(1, 10)
    elif args.model == 'cifarcnn':
        model = CifarCnn(3, 10)
    elif args.model == 'testcnn':
        model = TestCNN(_in_dim, 10)
    elif args.model == 'vgg':
        model = VGG(_in_dim, 10)
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
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        print("MLP was made")

    def forward(self, x):
        x = x.view((x.size()[0], -1))
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.elu(x)
        x = self.LogSoftmax(self.fc2(x))
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
        self.LogSoftmax = nn.LogSoftmax(dim=1)
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
        x = self.LogSoftmax(self.fc8(x))
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
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        print("MnistCNN was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.LogSoftmax(self.fc2(x))
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
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        print("CifarCnn was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.LogSoftmax(self.fc3(x))
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
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        print("TestCNN was made")
        
    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.LogSoftmax(self.fc(x))
        return x


# VGG 11-layer model
class VGG(nn.Module):
    def __init__(self, dim_in=3, dim_out=10, init_weights=True):
        if dim_in == 1:
            raise NotImplemented

        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=dim_out, bias=True)
        self.dp = nn.Dropout(p=0.5, inplace=False)

        if init_weights:
            self._initialize_weights()

        print("VGG was made")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #       (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (1): ReLU(inplace=True)
        #       (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.maxpool(self.relu(self.conv1(x)))

        #       (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (4): ReLU(inplace=True)
        #       (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.maxpool(self.relu(self.conv2(x)))

        #       (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (7): ReLU(inplace=True)
        x = self.relu(self.conv3(x))

        #       (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (9): ReLU(inplace=True)
        #       (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.maxpool(self.relu(self.conv4(x)))

        #       (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (12): ReLU(inplace=True)
        x = self.relu(self.conv5(x))

        #       (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (14): ReLU(inplace=True)
        #       (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.maxpool(self.relu(self.conv6(x)))

        #       (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (17): ReLU(inplace=True)
        x = self.relu(self.conv7(x))

        #       (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (19): ReLU(inplace=True)
        #       (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.maxpool(self.relu(self.conv8(x)))

        #     (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        #       (0): Linear(in_features=25088, out_features=4096, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): Dropout(p=0.5, inplace=False)
        x = self.dp(self.relu(self.fc1(x)))

        #       (3): Linear(in_features=4096, out_features=4096, bias=True)
        #       (4): ReLU(inplace=True)
        #       (5): Dropout(p=0.5, inplace=False)
        x = self.dp(self.relu(self.fc2(x)))

        #       (6): Linear(in_features=4096, out_features=10, bias=True)
        x = self.LogSoftmax(self.fc3(x))
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
