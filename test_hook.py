import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.s1 = nn.Sigmoid()
        self.fc2 = nn.Linear(2, 2)
        self.s2 = nn.Sigmoid()
        self.fc1.weight = torch.nn.Parameter(torch.Tensor([[0.15, 0.2], [0.250, 0.30]]))
        self.fc1.bias = torch.nn.Parameter(torch.Tensor([0.35]))
        self.fc2.weight = torch.nn.Parameter(torch.Tensor([[0.4, 0.45], [0.5, 0.55]]))
        self.fc2.bias = torch.nn.Parameter(torch.Tensor([0.6]))

    def forward(self, x):
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        x = self.s2(x)
        return x

net = Net()
print(net)

# parameters: weight and bias
print(list(net.parameters()))
# input data
weight2 = list(net.parameters())[2]
data = torch.Tensor([0.05,0.1])

# output of last layer
out = net(data)
target = torch.Tensor([0.01,0.99])  # a dummy target, for example
opt = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
loss = criterion(out, target)

opt.zero_grad()
loss.backward()

print(loss)


class Hook:
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


hookF, hookB = [], []
for layer in list(net._modules.items()):
    hookF.append(Hook(layer[1]))
    hookB.append(Hook(layer[1],backward=True))

# run a data batch
out=net(data)
# backprop once to get the backward hook results
out.backward(torch.tensor([1,1],dtype=torch.float),retain_graph=True)

print(")")