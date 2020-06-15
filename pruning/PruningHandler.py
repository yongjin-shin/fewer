
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .pruning_utils import *
from torch.nn.utils.prune import l1_unstructured, remove, CustomFromMask, is_pruned

__all__ = ['PruningHandler']


class PruningHandler():
    def __init__(self, args, prune_method=None, globally=True):
        """
        Pruning Handler to control pruning & recovery
        (default setting) : globally prune with L1 norm & no recovery
        """
        # Specify pruning & recovery plan for each round
        self.pruning_plan = self._planner(args)
        
        # Set pruning options
        self.globally = globally
        self.pruning = prune.L1Unstructured if prune_method is None else prune_method

    def pruner(self, model, fed_round):
        """Prune weights (expected to be called before distribution)"""
        
        weight_set = self._global_setter(model)
        #current_sparsity = self.global_sparsity_evaluator(model)
        amount = self._pruning_ratio_calculator(fed_round)
        
        if self.globally:
            keeped_masks = mask_collector(model)
            mask_test = mask_evaluator(keeped_masks)
            print(mask_test)
            prune.global_unstructured(weight_set,
                                      pruning_method=self.pruning, 
                                      amount=amount)
            print(amount)
            keeped_masks = mask_collector(model)
            mask_test = mask_evaluator(keeped_masks)
            print(mask_test)
            
        else:
            raise NotImplementedError('Structured pruning is not implemented yet!')
            
            #for name, module in server_model.named_modules():
            #    if isinstance(module, torch.nn.Conv2d):
            #        self.pruning(module, name='weight', amount=pruning_plan['conv'][fed_round])
            #    if isinstance(module, torch.nn.Linear):
            #        self.pruning(module, name='weight', amount=pruning_plan['fc'][fed_round])
            
        return model, keeped_masks


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
    
    def _pruning_ratio_calculator(self, fed_round):
        """
        pruning amount based on pruning_plan
        ![Caution] This assumes that the model has no applied mask!
        """
        amount = self.pruning_plan[fed_round]

        return amount
    
    def _planner(self, args):        
        if args.pruning:
            assert sum(args.plan) == args.nb_rounds,\
            'plan should should be same with nb_rounds!'
            
            pruning_plan = plan_organizer(args.plan, 
                                          args.target_sparsity, 
                                          args.plan_type)
            
        else:
            pruning_plan = [0] * args.nb_rounds
        
        return pruning_plan
