
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

__all__ = ['list_organizer', 'mask_collector']

def list_organizer(pack, plan):
    """make plan as list"""
    plan_list = []
    for i in range(len(pack)):
        plan_list += ([pack[i]] * plan[i])
    
    return plan_list


            
def mask_collector(model):
    """get current mask of mdoel"""
    model_masks = {}

    for module_name, module in model.named_modules():
        key = f"{module_name}.weight_mask"
        if isinstance(module, torch.nn.Conv2d):
            model_masks[key] = (module.weight != 0).int()
            
        if isinstance(module, torch.nn.Linear):
            model_masks[key] = (module.weight != 0).int()

    return model_masks
