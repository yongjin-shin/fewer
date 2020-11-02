import torch.nn as nn
import torch

__all__ = ['ExpNet0', 'ExpNet1', 'ExpNet2', 'ExpNet3', 'ExpNet4', 'ExpNet5']

class ExpNet0(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(ExpNet0, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.fc = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ExpNet1(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(ExpNet1, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ExpNet2(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(ExpNet2, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
class ExpNet3(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(ExpNet3, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
class ExpNet4(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(ExpNet4, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.conv5 = nn.Conv2d(256, 256, 5)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
class ExpNet5(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(ExpNet5, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.conv5 = nn.Conv2d(256, 256, 5)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
