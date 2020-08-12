import torch, math
import torch.nn as nn

__all__ = ['get_top_percentile', 'module_signal', 'signal_getter', 'signal_aggregator']


def get_top_percentile(tensor, topk=0.05, as_mask=True, min_zero=False):
    thres_idx = int(tensor.nelement() * topk)
    
    threshold = tensor.reshape(-1).topk(thres_idx)[0].min()
    threshold = max(1e-7, threshold) if min_zero else threshold
    
    elem_mask = tensor > threshold
    signal = elem_mask.int() if as_mask else tensor[elem_mask]
    
    return signal


def module_signal(module, module_mask, topk, as_mask=True):
    grad = module.weight.grad.data
    grad_mask = get_top_percentile(grad, topk=topk, as_mask=as_mask)
    signal = ((grad_mask - module_mask) == 1).int()
    
    return signal


def signal_getter(model, keeped_mask, topk, as_mask=True):
    signal_mask = {}
    
    for name, module in model.named_modules():
        key = f"{name}.weight_mask"
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module_mask = keeped_mask[key]
            signal_mask[key] = module_signal(module, module_mask, topk, as_mask)
            
    return signal_mask


def signal_aggregator(signal_mask_list, keeped_mask, topk=0.05):
    """collect values from dict list by corresponding keys"""
    
    recovery_mask = dict.fromkeys(keeped_mask)
    
    for key in keeped_mask.keys():
        signal = 0
        recent_mask = keeped_mask[key]
        
        for elem in signal_mask_list:
            signal += elem[key]
        
        signal = get_top_percentile(signal, topk, as_mask=True, min_zero=True)
        signal = ((signal + recent_mask) >= 1).int()
        
        recovery_mask[key] = signal
        
    return recovery_mask
