import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['OverhaulLoss']


class OverhaulLoss(nn.Module):
    def __init__(self, args):
        super(OverhaulLoss, self).__init__()
        self.num_classes = args.num_classes
        self.mode = args.mode
        self.temp = args.temp
        self.smoothing = args.smoothing
        self.beta = args.beta

    def forward(self, outputs, target, t_logits=None):
        logits = outputs
        loss = torch.zeros(logits.size(0)).to(str(target.device)) # initialize loss

        # one-hot + cross-entropy loss
        if self.mode == 'CE':
            hard_target = onehot(target, N=self.num_classes).float()
            loss += cross_entropy(logits, hard_target, reduction='none')
        ############### Soft Target Methods ##############################################

        # label smoothing
        elif self.mode == 'LS':
            with torch.no_grad():
                hard_target = onehot(target, N=self.num_classes).float()

            loss += cross_entropy(logits, hard_target, smooth_eps=self.smoothing, reduction='none')

        # knowledge distillation
        elif self.mode == 'KD':
            with torch.no_grad():
                t_distill = torch.softmax(t_logits/self.temp, dim=1)
                #print(t_distill)
            
            ce_loss = cross_entropy(logits, target, reduction='none')
            kd_loss = ((self.temp)**2) * cross_entropy(logits/self.temp, t_distill, reduction='none')
            loss +=  ((1-self.beta)*ce_loss + self.beta*kd_loss)
            #print((target == t_distill.max(dim=1)[1]).sum().item())

        # FedLSD
        elif self.mode == 'FedLSD':
            with torch.no_grad():
                t_logits[range(t_logits.shape[0]), target] = -10000 # very small number to true label logits
                hard_target = onehot(target, N=self.num_classes).float()
                t_distill = torch.softmax(t_logits/self.temp, dim=1)
                new_target = ((1-self.smoothing) * hard_target) + (self.smoothing * t_distill)
            loss += cross_entropy(logits, new_target, reduction='none')
        
        # Random Soft
        elif self.mode == 'RandKD':
            with torch.no_grad():
                t_distill = torch.softmax(t_logits.sort()[0]/self.temp, dim=1)
                #print(t_distill)
            
            ce_loss = cross_entropy(logits, target, reduction='none')
            kd_loss = ((self.temp)**2) * cross_entropy(logits/self.temp, t_distill, reduction='none')
            loss +=  ((1-self.beta)*ce_loss + self.beta*kd_loss)
            #print((target == t_distill.max(dim=1)[1]).sum().item())
                
        
        loss = loss.mean()  # Average Batch Loss

        return loss


###########################################################################################################################    
def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)
