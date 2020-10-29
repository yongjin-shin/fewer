import torch, copy, random
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from train_tools.utils import create_nets
from train_tools.criterion import OverhaulLoss
from train_tools.utils import get_test_results, compute_kl_divergence, compute_js_divergence


__all__ = ['Local']


# class Testobject:
#     def __init__(self, layer_list):
#         self.container = {}
#         for layer in layer_list:
#             self.container[layer] = []
#
#     def __call__(self, name, norm):
#         print(name, norm)
#         self.container[name].append(norm)
#
#     def __str__(self):
#         return self.container
# for n, p in self.model.named_parameters():
#     p.register_hook(obj(n, torch.norm(p.grad)[0]))


class Local:
    def __init__(self, args):
        self.args = args
        self.data_loader = None

        # model & optimizer
        self.model = create_nets(self.args, 'LOCAL').to(self.args.server_location)
        self.oracle = self.read_oracle() if self.args.oracle else None
        self.round_global = None
        self.optim, self.lr_scheduler = None, None
        self.criterion = OverhaulLoss(self.args)
        self.epochs = self.args.local_ep
        self.layers_name = None
        self.slow_list = None
        self.fast_list = None

    def train(self, beta=None):
        local_acc = None
        if self.args.global_loss_type != 'none':
            self.keep_global()
            
        if (self.args.mode == 'KD') or (self.args.mode == 'FedLSD'):
            self.keep_global()
            if self.args.use_beta_scheduler:
                local_ret = get_test_results(self.args, self.round_global, self.data_loader, None,
                                             return_loss=False, return_acc=True, return_logit=True)
                local_acc = local_ret['acc']

        t_logits = None
        fake_loader = cycle([(None, None)])
        ret_norm = dict(zip(self.layers_name, [[] for _ in range(len(self.layers_name))]))

        train_loss, train_acc, itr, ep = 0, 0, 0, 0
        self.model.to(self.args.device)

        # obj = Testobject(self.layers_name)

        for ep in range(self.epochs):
            for itr, ((data, target), (fake_data, fake_target)) in enumerate(zip(self.data_loader, fake_loader)):
                # forward pass
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                if fake_data is not None:
                    x = torch.cat((data, fake_data), dim=0)
                    y = torch.cat((target, fake_target), dim=0)
                    idx = torch.randperm(len(y))
                    data, target = x[idx], y[idx]

                output = self.model(data)
                if (self.args.mode == 'KD') or (self.args.mode == 'FedLSD'):
                    with torch.no_grad():
                        if self.args.oracle:
                            t_logits = self.oracle(data)
                        else:
                            t_logits = self.round_global(data)
                        
                loss = self.criterion(output, target, t_logits, acc=local_acc, beta=beta)
                # print(loss)

                if self.args.global_loss_type != 'none' and self.args.global_alpha > 0:
                    loss += (self.args.global_alpha * self.loss_to_round_global())
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optim.step()
                ret_norm = self.norm_info_stack(dict_params=dict(self.model.named_parameters()),
                                                containers=ret_norm)

                train_loss += loss.detach().item()

        local_loss = train_loss / (self.args.local_ep * (itr + 1))
        local_acc, kld = self.test()

        for layer in self.layers_name:
            ret_norm[layer] = np.mean(ret_norm[layer])

        ret = {
            'loss': local_loss,
            'acc' : local_acc,
            'kld'  : kld,
            'norm' : ret_norm
        }

        self.model.to(self.args.server_location)
        return ret

    def norm_info_stack(self, dict_params, containers):

        for layer in self.layers_name:
            _weight = dict_params[f"{layer}.weight"].grad.reshape((-1, 1))

            if f"{layer}.bias" in dict_params.keys():
                _bias = dict_params[f"{layer}.bias"].grad.reshape((-1, 1))
                _weight = torch.cat((_weight, _bias))

            _norm = torch.norm(_weight).item()
            containers[layer].append(_norm)
            # print(f"{layer}: {_norm}")
        return containers

    def test(self):
        with torch.no_grad():
            local_ret = get_test_results(self.args, self.model, self.data_loader, None,
                                         return_loss=False, return_acc=True, return_logit=True)
            local_acc = local_ret['acc']
            # global_ret = get_test_results(self.args, self.round_global, self.data_loader, None,
            #                               return_loss=False, return_acc=True, return_logit=True)
            # kld = compute_js_divergence(local_ret['logits'], global_ret['logits'])
            kld = 0

        return local_acc, kld

    def get_dataset(self, client_dataset):
        if client_dataset.__len__() <= 0:
            raise RuntimeError
        else:
            self.data_loader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True)
            
    def get_model(self, server_model, layers_name):
        self.model.load_state_dict(server_model)
        self.layers_name = layers_name

        # for p in self.model.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        
    def get_lr(self, server_lr):
        if server_lr < 0:
            raise RuntimeError("Less than 0")

        if self.args.slow_layer is None and self.args.fast_layer is None:
            self.optim = torch.optim.SGD(self.model.parameters(),
                                         lr=server_lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)

        if self.args.slow_layer is not None:
            self.slow_list = np.concatenate(
                [[f"{self.layers_name[i]}.weight", f"{self.layers_name[i]}.bias"] for i in self.args.slow_layer])

            slow_params = list(
                map(lambda x: x[1], list(filter(lambda kv: kv[0] in self.slow_list, self.model.named_parameters()))))
            base_params = list(
                map(lambda x: x[1], list(filter(lambda kv: kv[0] not in self.slow_list, self.model.named_parameters()))))

            self.optim = torch.optim.SGD(
                [{'params': base_params},
                 {'params': slow_params, 'lr': server_lr * self.args.slow_ratio}],
                lr=server_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )

        if self.args.fast_layer is not None:
            self.fast_list = np.concatenate(
                [[f"{self.layers_name[i]}.weight", f"{self.layers_name[i]}.bias"] for i in self.args.fast_layer])

            fast_params = list(
                map(lambda x: x[1], list(filter(lambda kv: kv[0] in self.slow_list, self.model.named_parameters()))))
            base_params = list(
                map(lambda x: x[1],
                    list(filter(lambda kv: kv[0] not in self.slow_list, self.model.named_parameters()))))

            self.optim = torch.optim.SGD(
                [{'params': base_params},
                 {'params': fast_params, 'lr': server_lr * self.args.fast_ratio}],
                lr=server_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )

        return
        
    def upload_model(self):
        return copy.deepcopy(self.model.state_dict())
    
    def reset(self):
        self.data_loader = None
        self.lr_scheduler = None
        self.round_global = None
        self.slow_list = None
        self.fast_list = None
        self.optim = None

    def keep_global(self):
        self.round_global = copy.deepcopy(self.model)
        for params in self.round_global.parameters():
            params.requires_grad = False
        
    def loss_to_round_global(self):
        vec = []
        if self.args.global_loss_type == 'l2':
            for i, ((name1, param1), (name2, param2)) in enumerate(zip(self.model.named_parameters(),
                                                                       self.round_global.named_parameters())):
                if name1 != name2:
                    raise RuntimeError

                if self.args.no_reg_to_recover:
                    raise NotImplemented
                else:
                    if self.args.slow_layer is not None and name1 in self.slow_list:
                        vec.append((param1 - param2).view(-1, 1))
                    else:
                        vec.append((param1-param2).view(-1, 1))

            all_vec = torch.cat(vec)
            loss = torch.norm(all_vec)  # (all_vec** 2).sum().sqrt()
        else:
            raise NotImplemented
        
        return loss * 0.5

    def read_oracle(self):
        model = create_nets(self.args, 'Oracle').to(self.args.server_location)
        model.load_state_dict(torch.load(f"./log/{self.args.oracle_path}/model.h5"))
        return model

    def sneaky_adversarial(self):
        x = []
        y = []

        self.round_global.to(self.args.device).eval()
        for itr, (data, true_target) in enumerate(self.data_loader):
            for j in range(2):
                target = torch.tensor(
                    [random.choice([x for x in range(10) if x not in torch.unique(true_target)]) for _ in range(len(true_target))],
                    device=self.args.device)
                data = data.to(self.args.device)

                noise1 = torch.rand(size=data.size(), requires_grad=True, device=self.args.device)
                # noise2 = torch.rand(size=data.size(), requires_grad=True, device=self.args.device)
                opt = torch.optim.Adam([noise1], lr=0.01)
                acc = 0
                cnt = 0
                idx = None
                while acc < 1 and cnt < 50:
                    # print(noise2[0, 0, 0])
                    # forward pass
                    output = self.model(data + noise1)
                    loss = F.cross_entropy(output, target)

                    # backward pass
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    idx = torch.argmax(output, dim=1) == target
                    acc = torch.sum(idx).item()/len(target)
                    # print(acc)
                    cnt += 1
                # print(f"{cnt}: {acc}")
                x.append((data + noise1.detach())[idx])
                y.append(target[idx])

        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        if len(y) > 0:
            ret_dataset = TensorDataset(x, y)
            ret_loader = DataLoader(ret_dataset, batch_size=self.args.local_bs * 2, shuffle=True)
            return ret_loader
        else:
            return cycle([(None, None)])
