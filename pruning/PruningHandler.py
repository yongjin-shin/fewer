import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pruning_utils import *

__all__ = ['PruningHandler']


class PruningHandler():
    def __init__(self, args, prune_method=None, recovery_method=None, globaly=True):
        """
        Pruning Handler to control pruning & recovery
        (default setting) : globally prune with L1 norm & no recovery
        """
        # Specify pruning & recovery plan for each round
        self.pruning_plan = self._planner(args.pruning_plan)
        self.recovery_plan = self._planner(args.recovery_plan)
        
        # Set pruning options
        self.globaly = globaly
        self.pruning = prune.l1_unstructured if prune_method is None else prune_method
        self.recovery = recovery_method
        
        
    def pruner(self, model, fed_round):
        """Prune weights (expected to be called before distribution)"""
            weight_set = self._global_setter(model)
            
            prune.global_unstructured(weight_set,
                                      pruning_method=self.pruning, 
                                      amount=self.pruning_plan[fed_round])
        
        
        #for name, module in server_model.named_modules():
        #    if isinstance(module, torch.nn.Conv2d):
        #        self.pruning(module, name='weight', amount=pruning_plan['conv'][fed_round])
        #    if isinstance(module, torch.nn.Linear):
        #        self.pruning(module, name='weight', amount=pruning_plan['fc'][fed_round])
                
    def recoverer(self, model, recovery_signals, fed_round):
        """
        To be implemented
        Recover weights (expected to be called before aggregation)
        """
        
        return recovery_signal

    def global_sparsity_evaluator(model):
        """Evaluate current sparsity of model"""
        num_sparse, num_weight = 0, 0
        weight_set = self._global_setter(model)
        
        for elem in weight_set:
            num_sparse += (elem[0].weight == 0).sum().item()
            num_weight += elem[0].weight.nelement()

        sparsity = round(num_sparse/num_weight, 4)

        return sparsity
    
    def _global_setter(model):
        """Specify target weight_set to prune"""
        modules = model.__dict__['_modules'].keys()
        weight_set = []

        for module in modules:
            if ('conv' in module) or ('fc' in module):
                module_weight = (getattr(model, module), 'weight')
                weight_set.append(module_weight)

        weight_set = tuple(weight_set)

        return weight_set

    def _planner(plan, mode='pruning'):
        if mode == 'pruning':
            pass
        else mode == 'recovery':
            pass
        
        return plan
