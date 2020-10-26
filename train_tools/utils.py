import torch, copy
import numpy as np
from scipy.special import softmax
from .models import *
from .criterion import *

__all__ = ['create_nets', 'get_size', 'compute_kl_divergence', 'compute_js_divergence',
           'get_test_results', 'ensemble_calc', 'get_server_location', 'get_device',
           'get_variance', 'knowledge_distillation']


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


def get_test_results(args, model, dataloader, return_loss, return_acc, return_logit, criterion=None):
    ret = {}
    ret_logit, test_loss, correct, itr = [], 0, 0, 0
    len_data = 0

    model.to(args.device).eval()
    for itr, (data, target) in enumerate(dataloader):
        len_data += data.size(0)
        data = data.to(args.device)
        target = target.to(args.device)
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

    return ret


def knowledge_distillation(args, dataset, mean_logits, student):

    kd_dataset = {
        'mean_logits': torch.tensor(np.vstack(mean_logits), dtype=torch.float32, device=args.device),
        'x': None,
        'y': None
    }

    x_container, y_container = [], []
    for x, y in dataset:
        x_container.append(x)
        y_container.append(y)

    kd_dataset['x'] = torch.cat(x_container, dim=0)
    kd_dataset['y'] = torch.cat(y_container)

    idx = np.arange(0, len(kd_dataset['y']))
    student.to(args.device)

    optim = torch.optim.Adam(student.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim,
                                                           T_max=args.fedDF_epoch,
                                                           eta_min=1e-5)

    for ep in range(args.fedDF_epoch):
        block_idx = np.array_split(np.random.permutation(idx),
                                   round(len(kd_dataset['y'])/128))
        avg_loss, i = 0, 0
        for enum, i in enumerate(block_idx):
            t_logits = kd_dataset['mean_logits'][i]
            output = student(kd_dataset['x'][i].to(dtype=torch.float32,
                                                   device=args.device))

            t_prob = torch.softmax(t_logits, dim=1)
            s_prob = torch.softmax(output, dim=1)
            loss = torch.nn.functional.kl_div(t_prob, s_prob, reduction='batchmean')

            optim.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optim.step()
        student.to(args.device).eval()
        training_results = get_test_results(args, student, dataset,
                                            return_loss=False, return_acc=True, return_logit=False)
        print(f"ep {ep}: {training_results}")
        scheduler.step()

    student.to(args.device).eval()
    return student


def ensemble_calc(args, data_loader, dummy_model, local_models):
    true_target = get_true_targets(data_loader)
    local_logits = []

    for l, local_model in enumerate(local_models):
        dummy_model.load_state_dict(copy.deepcopy(local_model))
        ret = get_test_results(args, dummy_model, data_loader,
                               return_loss=False, return_acc=True, return_logit=True)
        print(f"{l}th local ACC: {ret['acc']}")
        local_logits.append(ret['logits'])

    local_logits = np.dstack(local_logits)
    major_logits = []

    ensemble_acc_vote = 0
    ensemble_acc_mean = 0
    for i in range(len(local_logits)):
        vote = np.argmax(local_logits[i], axis=0)
        major = np.argmax(np.bincount(vote))
        if major == true_target[i]:
            ensemble_acc_vote += 1
        #
        # major_idx = np.where(vote == major)[0]
        # voted_logits = local_logits[i, :, major_idx]
        # mean_logits = np.mean(voted_logits, axis=0)
        # major_logits.append(mean_logits)

        mean_logits = np.mean(local_logits[i], axis=1)
        major = np.argmax(mean_logits)
        major_logits.append(mean_logits)
        if major == true_target[i]:
            ensemble_acc_mean += 1

    ensemble_acc_vote = round(ensemble_acc_vote / len(local_logits) * 100, 2)
    ensemble_acc_mean = round(ensemble_acc_mean / len(local_logits) * 100, 2)
    return ensemble_acc_mean, ensemble_acc_vote, major_logits


def get_true_targets(data_loader):
    ret = []
    for x, y in data_loader:
        ret.append(y)

    ret = torch.cat(ret)
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

