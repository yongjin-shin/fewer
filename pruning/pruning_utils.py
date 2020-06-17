import torch, copy
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import remove, CustomFromMask, is_pruned


__all__ = ['plan_organizer', 'mask_collector', 'mask_adder', 'mask_merger', 'mask_evaluator']

def plan_organizer(plan, target_sparsity, plan_type='base'):
    # unpack training plans
    warming_r, pruning_r, tuning_r = plan
    print(plan)
    # pruning plan
    pruning_plan = []

    for r in range(warming_r):
        if plan_type == 'reverse':
            pruning_plan.append(target_sparsity)
        else:
            pruning_plan.append(0)

    for r in range(pruning_r):
        # gradually increase to target sparsity
        if plan_type == 'base':
            sparsity = target_sparsity - target_sparsity * (1- (r+1)/pruning_r)**3
            
        # gradually decay from target sparsity
        elif plan_type == 'reverse':
            sparsity = target_sparsity * (1- (r+1)/pruning_r)**3
        
        pruning_plan.append(sparsity)

    for r in range(tuning_r):
        pruning_plan.append(pruning_plan[-1])
        
    return pruning_plan
    
def mask_collector(model):
    """get current mask of model"""
    model_masks = {}

    for module_name, module in model.named_modules():
        key = f"{module_name}.weight_mask"
        if isinstance(module, torch.nn.Conv2d):
            model_masks[key] = (module.weight_mask != 0).int()
            
        if isinstance(module, torch.nn.Linear):
            model_masks[key] = (module.weight_mask != 0).int()

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

#def mask_merger(model, original_model):
#    """get model parameters to original structure"""
#    temp_model = copy.deepcopy(original_model)

#    for name, temp_module in temp_model.named_modules():
#        if isinstance(temp_module, torch.nn.Conv2d):
#            getattr(temp_model, name).weight.data = getattr(model, name).weight.data
#            getattr(temp_model, name).bias.data = getattr(model, name).bias.data
            
#        if isinstance(temp_module, torch.nn.Linear):
#            getattr(temp_model, name).weight.data = getattr(model, name).weight.data
#            getattr(temp_model, name).bias.data = getattr(model, name).bias.data
        
#    return temp_model

def mask_evaluator(masks):
    """get sparsity of mask"""
    num_elem = 0
    masked_elem = 0
    
    for key, val in masks.items():
            masked_elem += (val == 0).int().sum().item()
            num_elem += val.nelement()

    return round(masked_elem/num_elem, 4)
