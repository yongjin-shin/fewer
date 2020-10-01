import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .pruning_utils import *
from .recovering_utils import *

__all__ = ['SparsityHandler']


class SparsityHandler():
    def __init__(self, args):
        """
        Sparsity Handler to control pruning & recovery
        """
        # Specify pruning & recovery plan for each round
        self.args = args
        self.sparsity_plan = self._planner(args)
        
    def round_sparsifier(self, model, fed_round, recovery_signals=None, keeped_mask=None, merge=False):
        """Prune weights (expected to be called before distribution)"""
        
        amount = self.sparsity_plan[fed_round]
        prev_amount = self.sparsity_plan[fed_round - 1]
        add_amount = round(prev_amount - amount, 6)

        if len(recovery_signals) == 0 or add_amount <= 0:
            if fed_round == 0:
                model = pruner(model, amount, random=True)
                print('\n------model is randomly initial pruned-----')
            else:
                model = pruner(model, amount)
            
            keeped_mask = mask_collector(model)
        
        else:
            recovery_mask = signal_aggregator(recovery_signals, keeped_mask, topk=add_amount)
            mask_adder(model, recovery_mask)
            keeped_mask = recovery_mask

        if merge:
            mask_merger(model)

        return model, keeped_mask
    
    def get_local_signal(self, local, keeped_mask, topk=0.05, as_mask=True, method='stack_grad'):
        """collect recovery signal from locals by the given method"""
        if method == 'stack_grad':
            local.stack_grad()
            signal_mask = signal_getter(local.model, keeped_mask, topk, as_mask)
    
        return signal_mask

    def get_masked_grad(self, local, mask):
        ret = {}
        for name, module in local.model.named_modules():
            key = f"{name}.weight_mask"
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module_mask = mask[key]
                grad = module.weight.grad.data
                masked_grad = grad * module_mask
                ret[f"{name}_weight"] = masked_grad.clone()
                ret[f"{name}_bias"] = module.bias.grad.data.clone()

        return ret

    def get_server_signal(self, server, keeped_mask, topk=0.05, as_mask=True, method='stack_grad'):
        if method == 'stack_grad':
            server.stack_grad()
            signal_mask = signal_getter(server.model, keeped_mask, topk, as_mask)
        return [signal_mask]

    def _planner(self, args):        
        if args.pruning:
            assert sum(args.plan) == args.nb_rounds,\
            'plan should should be same with nb_rounds!'
            
            sparsity_plan = plan_organizer(args.plan,
                                           args.target_sparsity,
                                           args.base_sparsity,
                                           args.recovery_sparsity,
                                           args.plan_type,
                                           args.decay_type)
            
        else:
            sparsity_plan = [0] * args.nb_rounds
        
        return sparsity_plan
