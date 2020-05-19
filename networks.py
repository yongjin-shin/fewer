
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        print("\nMLP was made")

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-1])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.elu(x)
        x = self.fc2(x)
        return self.LogSoftmax(x)


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
        print("\nMnistCNN was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.LogSoftmax(self.fc2(x))
        return self.LogSoftmax(x)


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
        print("\nCifarCnn was made")

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
        print("\nTestCNN was made")
        
    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.LogSoftmax(x)
