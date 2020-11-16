import torch, copy
import numpy as np
from scipy.special import softmax
from .models import *

__all__ = ['create_nets', 'get_size', 'compute_kl_divergence', 'compute_js_divergence',
           'model_evaluator', 'get_server_location', 'get_device', 'calc_l2_norm',
           'get_variance']


MODELS = {'mnistcnn': MnistCNN, 'cifarcnn': CifarCNN,
          'vgg11': vgg11, 'vgg11_slim': vgg11_slim, 
          'res8': resnet8, 'res14': resnet14, 'res20': resnet20}


def create_nets(model, dataset='cifar10', location='cpu', num_classes=10):
    print(f"{location}: ", end="", flush=True)
    
    if 'mnist' in dataset:
        dim_in = 1
        img_size = 1*28*28
        
    elif 'cifar' in dataset:
        dim_in = 3
        img_size = 3*32*32
    else:
        raise NotImplementedError

    if model in ['mlp', 'deep_mlp']:
        model = MODELS[model](img_size, hidden, num_classes=num_classes)
    else:
        model = MODELS[model](dim_in=dim_in, num_classes=num_classes)

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


def compute_kl_divergence(p_logits, q_logits):
    """"KL (p || q)"""
    p_probs = softmax(p_logits, axis=1)
    q_probs = softmax(q_logits, axis=1)

    kl_div = p_probs * np.log(p_probs / q_probs + 1e-12)
    return np.mean(np.sum(kl_div, axis=1), axis=0)


def compute_js_divergence(p_logits, q_logits):
    p_probs = softmax(p_logits, axis=1)
    q_probs = softmax(q_logits, axis=1)
    m = 0.5 * (p_probs + q_probs)

    kld_p_m = np.sum(p_probs * np.log(p_probs / m + 1e-12), axis=1)
    kld_q_m = np.sum(q_probs * np.log(q_probs / m + 1e-12), axis=1)
    js = np.sqrt(0.5 * (kld_p_m + kld_q_m))
    return float(np.mean(js, axis=0))


@torch.no_grad()
def model_evaluator(model, dataloader, criterion, device):
    running_loss, running_correct, data_num = 0, 0, 0

    model.to(device).eval()
    for itr, (data, target) in enumerate(dataloader):
        data_num += data.size(0)
        data, target = data.to(device), target.to(device)
        
        logits = model(data)
        pred = torch.max(logits, dim=1)[1]
        
        running_correct += (pred == target).sum().item()
        running_loss += criterion(logits, target).item()
        
    eval_loss = round(running_loss / data_num, 4)
    eval_acc = round((running_correct / data_num) * 100, 2)
    
    return eval_loss, eval_acc


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
    ret = {}
    for k in order:
        serial_vec = torch.empty((1, 0)).to(device)

        weight = vecs[f'{k}.weight'].reshape((1, -1))
        serial_vec = torch.cat((serial_vec, weight), axis=-1)

        if f'{k}.bias' in vecs.keys():
            bias = vecs[f'{k}.bias'].reshape((1, -1))
            serial_vec = torch.cat((serial_vec, bias), axis=-1)

        ret[k] = serial_vec
    return ret


def get_local_vec(order, locals, device):
    ret = []
    for local in locals:
        ret.append(get_vec(order, local, device))
    return ret


def get_variance(calc_order, server_state_dict, locals, device):
    _vars = []
    server_vecs = get_vec(calc_order, server_state_dict, device)
    local_vecs = get_local_vec(calc_order, locals, device)

    cos = torch.nn.CosineSimilarity(dim=1)
    ret_cos = {}
    
    for layer in calc_order:
        val_cos = torch.tensor([0], dtype=torch.float, device=device)
        for i, local_vec in enumerate(local_vecs):
            local_cos = torch.clamp(cos(server_vecs[layer], local_vec[layer]),
                                    max=1,
                                    min=-1)
            val_cos += torch.abs(torch.acos(local_cos))
        val_cos /= len(local_vecs)
        ret_cos[layer] = round(val_cos.item(), 3)
        
    return ret_cos


def calc_l2_norm(order, state_dict, device):
    ret = get_vec(order, state_dict, device)

    for k in ret.keys():
        ret[k] = torch.norm(ret[k]).item()

    return ret

