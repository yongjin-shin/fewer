import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import remove, CustomFromMask, is_pruned


__all__ = ['list_organizer', 'base_organizer', 'mask_collector', 'mask_adder', 'mask_merger']

def list_organizer(pack, plan):
    """make plan as list"""
    plan_list = []
    for i in range(len(pack)):
        plan_list += ([pack[i]] * plan[i])
    
    return plan_list

def base_organizer(plan_list):
    """make base sparsity list"""
    base_list = [0]
    for i in range(len(plan_list)):
        base_list.append(base_list[-1]+plan_list[i])
        
    return base_list
            
def mask_collector(model):
    """get current mask of model"""
    model_masks = {}

    for module_name, module in model.named_modules():
        key = f"{module_name}.weight_mask"
        if isinstance(module, torch.nn.Conv2d):
            model_masks[key] = (module.weight != 0).int()
            
        if isinstance(module, torch.nn.Linear):
            model_masks[key] = (module.weight != 0).int()

    return model_masks

def mask_adder(model, masks):
    """prune model by given masks"""
    mask_pruner = CustomFromMask(None)
    for module_name, module in model.named_modules():
        key = f"{module_name}.weight_mask"
        if key in masks:
            if isinstance(module, torch.nn.Conv2d):
                _mask = masks[key]
                mask_pruner.apply(module, 'weight', _mask)
            if isinstance(module, torch.nn.Linear):
                _mask = masks[key]
                mask_pruner.apply(module, 'weight', _mask)

def mask_merger(model):
    "remove mask but let weights stay pruned"
    for n, m in model.named_modules():
        if is_pruned(m)==False:
            continue
        if isinstance(m, torch.nn.Conv2d):
            remove(m, name='weight')
        if isinstance(m, torch.nn.Linear):
            remove(m, name='weight')    
