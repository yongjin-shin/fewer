import torch, copy
import numpy as np
from .models import *
import seaborn as sns
import matplotlib.pylab as plt

__all__ = ['create_nets', 'get_size', 'get_server_location', 'get_device', 'get_models_variance', 'local_clippers', 'get_models_covariance']


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
        if 'cifar100' == args.dataset:
            num_classes = 100
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


def calc_delta(local, server):
    """local - server"""
    delta = {}
    for k, v in server:
        delta[k] = local[k] - v

    return delta


def get_delta_params(server, locals):
    delta_locals = []
    for local in locals:
        delta_locals.append(calc_delta(local, server))
    return delta_locals


def calc_var(server, local):
    _var = {}
    for k in server:
        _var[k] = torch.matmul(server[k].reshape((-1, 1)).T, local[k].reshape((-1, 1))).item()
    return _var


def get_vec(order, vecs, device):
    ret = torch.empty((0, 1)).to(device)
    for k in order:
        ret = torch.cat((ret, vecs[k].reshape((-1, 1))), axis=0)
    return ret


def get_local_vec(order, locals, device):
    ret = []
    for local in locals:
        ret.append(get_vec(order, local, device))
    return torch.cat(ret, dim=1)


def get_variance(server, locals, device):
    _vars = []
    calc_order = [k for k in server]
    server_vecs = get_vec(calc_order, server, device)
    local_vecs = get_local_vec(calc_order, locals, device)

    cos = torch.nn.CosineSimilarity(dim=0)
    ret = torch.log(torch.var(cos(server_vecs.repeat(1, local_vecs.shape[-1]),
                                  local_vecs)))

    return ret.item()


def get_models_variance(server, locals, device):
    return get_variance(server, locals, device)


def sim_matrix(a, b, eps=1e-6):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=0)[None, :], b.norm(dim=0)[None, :]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

    sim_mt = torch.empty(a_norm.size()[-1], a_norm.size()[-1])
    for i in range(a_norm.size()[-1]):
        for j in range(a_norm.size()[-1]):
            sim_mt[i, j] = torch.dot(a_norm[:, i], b_norm[:, j])
    return sim_mt


def get_models_covariance(locals, device, title):
    calc_order = [k for k in locals[0]]
    local_vecs = get_local_vec(calc_order, locals, device)
    ret = sim_matrix(local_vecs, local_vecs)
    sns.heatmap(ret, linewidth=0.5)
    plt.title(f"Round: {title}")
    return plt


def local_clippers(args, loss, weights):
    orig_idx = set(np.arange(len(loss)))
    if args.clip_type == 'half':
        idx = np.argsort(loss)
        idx = idx[:int(len(idx)/2)] if args.clip_dir == 'good' else idx[int(len(idx) / 2):]
        _remained = list(orig_idx - set(idx))
        return np.array(loss)[idx], np.array(loss)[_remained], np.array(weights)[idx]

    elif args.clip_type == 'std':
        mean = np.mean(loss)
        std = np.std(loss)
        idx = np.arange(len(loss))
        idx = idx[np.array(loss) <= mean-std] if args.clip_dir == 'good' else idx[np.array(loss) >= mean+std]
        _remained = list(orig_idx - set(idx))
        return np.array(loss)[idx], np.array(loss)[_remained], np.array(weights)[idx]

    else:
        return loss, [], weights
