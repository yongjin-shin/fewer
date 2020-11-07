import torch, copy
import numpy as np
from scipy.special import softmax
from .models import *

__all__ = ['create_nets', 'get_size', 'compute_kl_divergence', 'compute_js_divergence',
           'get_test_results', 'get_server_location', 'get_device', 'calc_l2_norm',
           'get_variance']


MODELS = {'mlp': MLP, 'deep_mlp': DeepMLP, 'testcnn': TestCNN,'mnistcnn': MnistCNN, 'cifarcnn': CifarCNN,
          'vgg11': vgg11, 'vgg11_slim': vgg11_slim, 'res8': resnet8, 'res14': resnet14, 'res20': resnet20,
          'exp0': ExpNet0, 'exp1': ExpNet1, 'exp2': ExpNet2, 'exp3': ExpNet3 
         }


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


def get_test_results(args, model, dataloader, criterion, return_loss, return_acc, return_logit, return_unique_labels=False):
    ret = {}
    ret_logit, test_loss, correct, itr = [], 0, 0, 0
    len_data = 0
    
    y_containers = []

    model.to(args.device).eval()
    for itr, (data, target) in enumerate(dataloader):
        len_data += data.size(0)
        data = data.to(args.device)
        target = target.to(args.device)
        y_containers.append(target)
        logits = model(data)

        if return_loss:
            test_loss += criterion(logits, target).item()

        if return_acc:
            y_pred = torch.max(logits, dim=1)[1]
            correct += torch.sum(y_pred.view(-1) == target.to(args.device).view(-1)).cpu().item()

        if return_logit:
            ret_logit.append(logits.cpu().detach().numpy())

    if return_loss:
        ret['loss'] = test_loss / (itr + 1)

    if return_acc:
        ret['acc'] = 100 * float(correct) / float(len_data)

    if return_logit:
        ret['logits'] = np.concatenate(ret_logit)
        
    if return_unique_labels:
        ret['label'] = torch.unique(torch.cat(y_containers)).tolist()

    return ret


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

    # server_norm = torch.norm(server_vecs)
    # local_norms = torch.norm(local_vecs, dim=0)
    #
    # inner_prod = server_vecs.T @ local_vecs
    # normalizer = server_norm * local_norms
    # ret = torch.var(inner_prod / normalizer)
    cos = torch.nn.CosineSimilarity(dim=1)
    # ret = torch.log(torch.var(cos(server_vecs.repeat(1, local_vecs.shape[-1]),
    #                               local_vecs)))
    ret_cos = {}
    for layer in calc_order:
        val_cos = torch.tensor([0], dtype=torch.float, device=device)
        for i, local_vec in enumerate(local_vecs):
            local_cos = torch.clamp(cos(server_vecs[layer], local_vec[layer]),
                                    max=1,
                                    min=-1)
            # print(f"{layer}/{i}: {local_cos}")
            val_cos += torch.abs(torch.acos(local_cos))
        val_cos /= len(local_vecs)
        ret_cos[layer] = round(val_cos.item(), 3)
    return ret_cos


def calc_l2_norm(order, state_dict, device):
    ret = get_vec(order, state_dict, device)

    for k in ret.keys():
        ret[k] = torch.norm(ret[k]).item()

    return ret

