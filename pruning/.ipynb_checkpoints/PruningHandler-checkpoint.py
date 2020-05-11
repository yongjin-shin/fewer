import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

__all__ = ['PruningHandler']

class PruningHandler():    
    def __init__(self, pruning_plan, recovery_plan, recovery_method=None, prune_method=None, globaly=False):
        self.pruning_plan = pruning_plan
        self.recovery_plan = recovery_plan
        
        self.pruning = prune.l1_unstructured if prune_method is None else prune_method
        self.recovery = recovery_method
        self.globaly = globaly
        
        
    def pruner(self, server_model, fed_round):
        if self.globaly:
            weight_set = _global_setter(server_model)
            prune.global_unstructured(weight_set, pruning_method=self.pruning, amount=pruning_plan['global'][fed_round])
        
        for name, module in server_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.pruning(module, name='weight', amount=pruning_plan['conv'][fed_round])
            if isinstance(module, torch.nn.Linear):
                self.pruning(module, name='weight', amount=pruning_plan['fc'][fed_round])
                
        
    def recovery(self, local_model, fed_round):
        """"""
        recovery_signal = []
        for name, module in local_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                signal = self.recovery(module, amount=recovery_plan['conv'][fed_round])
            if isinstance(module, torch.nn.Linear):
                signal = self.recovery(module, amount=recovery_plan['fc'][fed_round]) 
            
            recovery_signal.append((name, signal))
        
        return recovery_signal

    
    def _global_setter(self, model):
        modules = model.__dict__['_modules'].keys()
        conv_set, fc_set = [], []
        for module in modules:
            if ('conv' in module) or ('fc' in module):
                conv_set.append((getattr(model, module), 'weight'))        
        
        return weight_set
        