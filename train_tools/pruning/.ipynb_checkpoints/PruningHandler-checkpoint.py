import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .pruning_utils import *

__all__ = ['PruningHandler']


class PruningHandler():
    def __init__(self, args):
        """
        Pruning Handler to control pruning & recovery
        """
        # Specify pruning & recovery plan for each round
        self.pruning_plan = self._planner(args)

    def round_pruner(self, model, fed_round, merge=False):
        """Prune weights (expected to be called before distribution)"""
        
        amount = self.pruning_plan[fed_round]
        model = pruner(model, amount)
        keeped_masks = mask_collector(model)
        
        if merge:
            mask_merger(model)

        return model, keeped_masks
    
    def _planner(self, args):        
        if args.pruning:
            assert sum(args.plan) == args.nb_rounds,\
            'plan should should be same with nb_rounds!'
            
            pruning_plan = plan_organizer(args.plan, 
                                          args.target_sparsity,
                                          args.base_sparsity,
                                          args.plan_type,
                                          args.decay_type)
            
        else:
            pruning_plan = [0] * args.nb_rounds
        
        return pruning_plan
