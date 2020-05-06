import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pruning_utils import *

__all__ = ['PruningHandler']


class PruningHandler():
    def __init__(self, args, prune_method=None, recovery_method=None, globally=True):
        """
        Pruning Handler to control pruning & recovery
        (default setting) : globally prune with L1 norm & no recovery
        """
        # Specify pruning plan for each round
        self.pruning_plan = self._planner(args.enable_pruning,
                                          args.pruning_plan, 
                                          mode='pruning')
        
        # Specify recovery plan for each round
        self.recovery_plan = self._planner(args.enable_recovery,
                                           args.recovery_plan , 
                                           mode='recovery')
        
        # Set pruning options
        self.globally = globally
        self.pruning = prune.l1_unstructured if prune_method is None else prune_method
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
        else:
            raise NotImplementedError('recovery is not implemented yet!')
                
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
        target_amount = self.pruning_plan[fed_round]
        amount = round(target_amount / (1-current_sparsity + 1e-7), 5)
        return amount
    
    def _planner(self, plan, mode='pruning'):
        if mode == 'pruning':
            return list_organizer(plan)
        
        else mode == 'recovery':
            # To be implemented
            return list_organizer(plan)
        
        return plan
