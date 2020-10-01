from torch.optim.lr_scheduler import StepLR
from math import log

__all__ = ['ConstantLR', 'LinearLR', 'LinearStepLR']


class CustomLR:
    def __init__(self, init_lr, epoch=None, eta_min=None):
        self.init_lr = init_lr
        self.crnt_lr = init_lr
        self.diff = 0

    def get_last_lr(self):
        return [self.crnt_lr]

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        return {
            'init': self.init_lr,
            'crnt': self.crnt_lr,
            'diff': self.diff
        }

    def load_state_dict(self, _dict):
        self.init_lr = _dict['init']
        self.crnt_lr = _dict['crnt']
        self.diff = _dict['diff']


class ConstantLR(CustomLR):
    def __init__(self, init_lr):
        super().__init__(init_lr)

    def step(self):
        pass


class LinearLR(CustomLR):
    def __init__(self, init_lr, epoch, eta_min):
        super().__init__(init_lr, epoch, eta_min)
        tot_diff = init_lr - eta_min
        self.diff = tot_diff / (epoch-1)

    def step(self):
        self.crnt_lr -= self.diff


class LinearStepLR:
    def __init__(self, optimizer, init_lr, epoch, eta_min, decay_rate):
        n = int((log(eta_min) - log(init_lr))/log(decay_rate)) + 1
        step_size = int(epoch/n)
        self.scheduler = StepLR(optimizer=optimizer, gamma=decay_rate,
                                step_size=step_size)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def step(self):
        self.scheduler.step()