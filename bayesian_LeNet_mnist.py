import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
import time
import random
import numpy as np

np.random.seed(123)  # for the reproducibility
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# train_dataset = dsets.MNIST(root="./data",
#                              train=True,
#                              transform=transforms.ToTensor(),
#                              download=True
#                             )
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=64,
#                                            shuffle=True)
#
# test_dataset = dsets.MNIST(root="./data",
#                              train=False,
#                              transform=transforms.ToTensor(),
#                              download=True
#                             )
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                            batch_size=64,
#                                            shuffle=True)


def cifar_data_augmentation():
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


train_transform, test_transform = cifar_data_augmentation()
dataset_train = dsets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=128, shuffle=True)
dataset_test = dsets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=128, shuffle=True)


@variational_estimator
class BayesianCifarCNN(nn.Module):
    def __init__(self):
        super(BayesianCifarCNN, self).__init__()
        self.conv1 = BayesianConv2d(3, 32, (5, 5))
        self.conv2 = BayesianConv2d(32, 64, (5, 5))
        self.fc1 = BayesianLinear(64*5*5, 512)
        self.fc2 = BayesianLinear(512, 128)
        self.fc3 = BayesianLinear(128, 10)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        print("BayesianCifarCNN was made")

    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@variational_estimator
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1 = BayesianLinear(256, 120)
        self.fc2 = BayesianLinear(120, 84)
        self.fc3 = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCifarCNN().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# classifier.freeze_()
for epoch in range(50):
    classifier.unfreeze_()
    epoch_loss = 0
    tic = time.time()
    for i, (datapoints, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                      labels=labels.to(device),
                                      criterion=criterion,
                                      sample_nbr=5,
                                      complexity_cost_weight=1/50000)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # classifier.freeze_()
            outputs = classifier(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    print(f'Freeze Epoch {epoch} | {str(100 * correct / total)}% | Elpased: {time.time() - tic:.1f}s')

    # total = 0
    # correct = 0
    # classifier.unfreeze_()
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         outputs = classifier(images.to(device))
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels.to(device)).sum().item()
    # print(f'UnFreeze Epoch {epoch} | {str(100 * correct / total)}% | Elpased: {time.time() - tic:.1f}s')
