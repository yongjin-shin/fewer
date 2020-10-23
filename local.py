import torch, copy, random
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from train_tools.utils import create_nets
from train_tools.criterion import OverhaulLoss
from train_tools.utils import get_test_results, compute_kl_divergence, compute_js_divergence


__all__ = ['Local']


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
        # fake_loader = self.sneaky_adversarial()
        fake_loader = cycle([(None, None)])

        train_loss, train_acc, itr, ep = 0, 0, 0, 0
        self.model.to(self.args.device)

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
                try:
                    loss = self.criterion(output, target, t_logits, acc=local_acc, beta=beta)
                except:
                    print("here")
                
                if self.args.global_loss_type != 'none' and self.args.global_alpha != 0:
                    loss += (self.args.global_alpha * self.loss_to_round_global())
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_loss += loss.detach().item()

        local_loss = train_loss / (self.args.local_ep * (itr + 1))
        local_acc, kld = self.test()

        ret = {
            'loss': local_loss,
            'acc' : local_acc,
            'kld'  : kld
        }

        self.model.to(self.args.server_location)
        print(ret)
        return ret

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
            
    def get_model(self, server_model):
        self.model.load_state_dict(server_model)
        
    def get_lr(self, server_lr):
        if server_lr < 0:
            raise RuntimeError("Less than 0")

        if self.args.optimizer.lower() == str('SGD').lower():
            self.optim = torch.optim.SGD(self.model.parameters(), lr=server_lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == str('ADAM').lower():
            self.optim = torch.optim.Adam(self.model.parameters(), lr=server_lr,
                                          weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError
        
    def upload_model(self):
        return copy.deepcopy(self.model.state_dict())
    
    def reset(self):
        self.data_loader = None
        self.optim = None
        self.lr_scheduler = None
        self.round_global = None

    def keep_global(self):
        self.round_global = copy.deepcopy(self.model)
        for params in self.round_global.parameters():
            params.requires_grad = False
        
    def loss_to_round_global(self):
        vec = []
        if self.args.global_loss_type == 'l2':
            for i, (param1, param2) in enumerate(zip(self.model.parameters(), self.round_global.parameters())):
                if self.args.no_reg_to_recover:
                    raise NotImplemented
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
