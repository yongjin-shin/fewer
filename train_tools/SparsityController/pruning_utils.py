import torch, copy
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import remove, CustomFromMask, is_pruned

__all__ = ['pruner', 'plan_organizer', 'mask_collector', 'mask_adder', 'mask_merger', 'sparsity_evaluator']


@torch.no_grad()
def pruner(model, amount, random=False):
    """
    (amount) total amount of desired sparsity
    """
    for name, module in model.named_modules():
        # prune declared amount of connections in all 2D-conv & Linear layers
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if random:
                prune.random_unstructured(module, name='weight', amount=amount)

            else:
                prune.l1_unstructured(module, name='weight', amount=amount)
                #prune.remove(module, 'weight') # make it permanent
            
    return model


def plan_organizer(plan, target_sparsity, base_sparsity=0, plan_type='base', decay_type='gradual'):
    # unpack training plans
    warming_r, pruning_r, tuning_r = plan
    # pruning plan
    pruning_plan = []

    for r in range(warming_r):
        if plan_type == 'reverse':
            pruning_plan.append(target_sparsity)
        else:
            pruning_plan.append(base_sparsity)

    for r in range(pruning_r):
        # gradually increase to target sparsity
        if plan_type == 'base':
            sparsity = target_sparsity - \
            (target_sparsity-base_sparsity) * _decay_rate(r, pruning_r-1, decay_type)
            
        # gradually decay from target sparsity
        elif plan_type == 'reverse':
            sparsity = base_sparsity + \
            (target_sparsity-base_sparsity) * _decay_rate(r, pruning_r-1, decay_type)
        
        pruning_plan.append(sparsity)

    for r in range(tuning_r):
        pruning_plan.append(pruning_plan[-1])
        
    pruning_plan = [round(elem, 4) for elem in pruning_plan]
        
    return pruning_plan


def _decay_rate(r, pruning_r, decay_type):
    if decay_type == 'gradual':
        ratio = (1- r/pruning_r)**3
        
    elif decay_type == 'linear':
        ratio = (1- r/pruning_r)
        
    elif decay_type == 'reverse_gradual':
        ratio = (1- r/pruning_r)**(1/3)
    return ratio


@torch.no_grad()
def mask_collector(model):
    """get current mask of model"""
    model_masks = {}

    for module_name, module in model.named_modules():
        key = f"{module_name}.weight_mask"
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            model_masks[key] = (module.weight_mask != 0).int()

    return model_masks


@torch.no_grad()
def mask_adder(model, masks):
    """prune model by given masks"""
    mask_pruner = CustomFromMask(None)
    for module_name, module in model.named_modules():
        key = f"{module_name}.weight_mask"
        if key in masks:
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                _mask = masks[key]
                mask_pruner.apply(module, 'weight', _mask)

                
@torch.no_grad()
def mask_merger(model):
    """remove mask but let weights stay pruned"""
    for name, module in model.named_modules():
        if is_pruned(module)==False:
            continue
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            remove(module, name='weight') 

            
@torch.no_grad()
def sparsity_evaluator(model, for_mask=False):
    """evaluate sparsity of model"""
    num_sparse, num_weight = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if for_mask:            
                num_sparse += (module.weight_mask == 0).sum().item()
                num_weight += module.weight_mask.nelement()
            
            else:
                num_sparse += (module.weight == 0).sum().item()
                num_weight += module.weight.nelement()
            
    sparsity = round(num_sparse/num_weight, 4)
        
    return sparsity


    
