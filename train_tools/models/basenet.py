import torch
import torch.nn as nn

__all__ = ['MLP', 'DeepMLP', 'TestCNN']


class MLP(nn.Module):
    def __init__(self, dim_in=3*32*32, dim_hidden=100, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(dim_hidden, num_classes)
        print("MLP was made")

    def forward(self, x):
        x = x.view((x.size()[0], -1))
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x


class DeepMLP(nn.Module):
    def __init__(self, dim_in=3*32*32, num_classes=10):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, num_classes)
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


class TestCNN(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(TestCNN, self).__init__()
        self.dim_in = dim_in
        self.conv1 = nn.Conv2d(dim_in, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.elu = nn.ELU()
        self.fc = nn.Linear(64, num_classes)
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
