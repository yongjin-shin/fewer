import torch, copy
from .models import *

__all__ = ['create_nets', 'get_size', 'get_server_location', 'get_device', 'get_models_variance', 'tensor_concater']


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

    # server_norm = torch.norm(server_vecs)
    # local_norms = torch.norm(local_vecs, dim=0)
    #
    # inner_prod = server_vecs.T @ local_vecs
    # normalizer = server_norm * local_norms
    # ret = torch.var(inner_prod / normalizer)
    cos = torch.nn.CosineSimilarity(dim=0)
    ret = torch.log(torch.var(cos(server_vecs.repeat(1, local_vecs.shape[-1]),
                                  local_vecs)))

    return ret.item()


def get_models_variance(server, locals, device):
    return get_variance(server, locals, device)


def tensor_concater(tensor1, tensor2):
    if tensor1 is None:
        tensor1 = tensor2
    else:
        tensor1 = torch.cat((tensor1, tensor2), dim=0)
        
    return tensor1