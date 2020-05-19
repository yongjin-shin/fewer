
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .pruning_utils import *
from torch.nn.utils.prune import l1_unstructured, remove, CustomFromMask, is_pruned

__all__ = ['PruningHandler']


class PruningHandler():
    def __init__(self, args, prune_method=None, recovery_method=None, globally=True):
        """
        Pruning Handler to control pruning & recovery
        (default setting) : globally prune with L1 norm & no recovery
        """
        # Specify pruning & recovery plan for each round
        self.pruning_plan, self.recovery_plan = self._planner(args)
        
        # Set pruning options
        self.globally = globally
        self.pruning = prune.L1Unstructured if prune_method is None else prune_method
        self.recovery = recovery_method
        
        
    def pruner(self, model, fed_round):
        """Prune weights (expected to be called before distribution)"""
        
        weight_set = self._global_setter(model)
        current_sparsity = self.global_sparsity_evaluator(model)
        amount = self._pruning_ratio_calculator(current_sparsity, fed_round)
        
        if self.globally:
            prune.global_unstructured(weight_set,
                                      pruning_method=self.pruning, 
                                      amount=amount)

            # Deploy permenant pruning (remove reparametrization)
            keeped_masks = mask_collector(model)
            self.mask_merger(model)

        else:
            raise NotImplementedError('recovery is not implemented yet!')
            
            #for name, module in server_model.named_modules():
            #    if isinstance(module, torch.nn.Conv2d):
            #        self.pruning(module, name='weight', amount=pruning_plan['conv'][fed_round])
            #    if isinstance(module, torch.nn.Linear):
            #        self.pruning(module, name='weight', amount=pruning_plan['fc'][fed_round])
            
        return model, keeped_masks
    
    def mask_adder(self, model, masks):
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

    def mask_merger(self, model):
        "remove mask but let weights stay pruned"
        for n, m in model.named_modules():
            if is_pruned(m)==False:
                continue
            if isinstance(m, torch.nn.Conv2d):
                remove(m, name='weight')
            if isinstance(m, torch.nn.Linear):
                remove(m, name='weight')    
    
    def recoverer(self, model, recovery_signals, fed_round):
        """
        To be implemented
        Recover weights (expected to be called before aggregation)
        """
        
        return recovery_signals

    def global_sparsity_evaluator(self, model):
        """Evaluate current sparsity of model"""
        num_sparse, num_weight = 0, 0
        weight_set = self._global_setter(model)
        
        for elem in weight_set:
            num_sparse += (elem[0].weight == 0).sum().item()
            num_weight += elem[0].weight.nelement()

        sparsity = round(num_sparse/num_weight, 4)

        return sparsity
    
    def _global_setter(self, model):
        """Specify target weight_set to prune"""
        modules = model.__dict__['_modules'].keys()
        weight_set = []

        for module in modules:
            if ('conv' in module) or ('fc' in module):
                module_weight = (getattr(model, module), 'weight')
                weight_set.append(module_weight)

        weight_set = tuple(weight_set)

        return weight_set
    
    def _pruning_ratio_calculator(self, current_sparsity, fed_round):
        """Calculate pruning amount based on sparsity and round"""

        amount = 0
        target_amount = self.pruning_plan[fed_round]

        amount = min(current_sparsity + target_amount, 0.999)
        
        return amount
    
    def _planner(self, args):        
        if args.enable_pruning:
            assert sum(args.pruning_plan) == args.nb_rounds,\
            'plan should should be same with nb_rounds!'
            
            pruning_plan = list_organizer(args.pruning_pack,
                                  args.pruning_plan)
        else:
            pruning_plan  [0] * args.nb_rounds

        if args.enable_recovery:
            assert sum(args.recovery_plan) == args.nb_rounds,\
            'plan should should be same with nb_rounds!'
            
            recovery_plan = list_organizer(args.recovery_pack,
                                  args.recovery_plan)
        else:
            recovery_plan =  [0] * args.nb_rounds
        
        return pruning_plan, recovery_plan
