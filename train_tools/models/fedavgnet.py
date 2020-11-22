import torch.nn as nn
import torch


__all__ = ['MnistCNN', 'CifarCNN']

class MnistCNN(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(MnistCNN, self).__init__()
        self.dim_in = dim_in
        self.conv1 = nn.Conv2d(dim_in, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        print("MnistCNN was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CifarCNN(nn.Module):
    def __init__(self, dim_in=3, num_classes=10):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        print("CifarCNN was made")

    def forward(self, x, y=None, mix_layer=None, alpha=None, mixup_mode='add'):
        if mix_layer == 0:
            x, y = simple_mixup(x, y, alpha, mixup_mode)
        x = self.mp(self.relu(self.conv1(x)))
        
        if mix_layer == 1:
            x, y = simple_mixup(x, y, alpha, mixup_mode)
        x = self.mp(self.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        if mix_layer == 2:
            x, y = simple_mixup(x, y, alpha, mixup_mode)
        x = self.relu(self.fc1(x))
        
        if mix_layer == 3:
            x, y = simple_mixup(x, y, alpha, mixup_mode)
        x = self.relu(self.fc2(x))
        
        if mix_layer == 4:
            x, y = simple_mixup(x, y, alpha, mixup_mode)
        x = self.fc3(x)
        
        if y is None and mix_layer is None:
            return x
        elif y is not None and mix_layer is not None:
            return x, y
        else:
            raise RuntimeError("Here")


def simple_mixup(x, y, alpha, mixup_mode):
    if 'add' == mixup_mode:
        idx = torch.randperm(len(y))
        randx = x.clone()[idx]
        randy = y.clone()[idx]
    elif 'split' == mixup_mode:
        idx = int(len(y)/2)
        randx, randy = x[idx:].clone(), y[idx:].clone()
        x, y = x[:idx], y[:idx]
    else:
        raise NotImplementedError

    # lam = torch.rand(len(idx)).reshape(-1, 1, 1, 1).to(x.device)
    beta = torch.distributions.beta.Beta(alpha, alpha)
    lam = torch.Tensor([beta.sample() for _ in range(len(randy))]).to(x.device)
    lam_weight = torch.cat([b.expand_as(x[0]).unsqueeze(0) for b in lam], dim=0)

    assert torch.all(lam >= 0.0)
    assert torch.all(lam <= 1.0)

    mixed_x = lam_weight * x + (1 - lam_weight) * randx
    mixed_y = torch.stack((y.to(lam.dtype), randy.to(lam.dtype), lam)).T
    ret = (mixed_x, mixed_y)
    return ret

