import torch, copy
from .models import *

__all__ = ['create_nets', 'get_size', 'get_server_location', 'get_device']


MODELS = {'mlp': MLP, 'deep_mlp': DeepMLP, 'testcnn': TestCNN,'mnistcnn': MnistCNN, 'cifarcnn': CifarCNN,
         'vgg11': vgg11, 'vgg11_slim': vgg11_slim, 'res8': resnet8, 'res14': resnet14, 'res20': resnet20}


def create_nets(args, location, num_classes=10):
    print(f"{location}: ", end="", flush=True)
    
    if 'mnist' in args.dataset:
        dim_in = 1
        img_size = 1*28*28
        
    elif 'cifar' in args.dataset:
        dim_in = 3
        img_size = 3*32*32
    else:
        raise NotImplementedError

    if args.model in ['mlp', 'deep_mlp']:
        model = MODELS[args.model](img_size, args.hidden, num_classes=num_classes)
    else:
        model = MODELS[args.model](dim_in=dim_in, num_classes=num_classes)

    return model


def get_size(param):
    size = 0

    for p in param:
        tmp = p.detach().to('cpu').numpy()
        size += tmp.nbytes

    return round(size/1024/1024, 2)


def get_server_location(args):
    location = args.device if args.gpu else 'cpu'
    
    return location


def get_device(args):
    if args.gpu:
        device = 'cuda:1' if args.cuda_type else 'cuda:0'
        
    else:
        device = 'cpu'
        
    return device
