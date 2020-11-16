import torch
import torch.nn.functional as F


__all__ = ['Redpill']


class Redpill():
    def __init__(self, server, device='cuda:0'):
        self.server = server
        self.device = device
        self.round_mixed = []
        
    def local_mixer(self, local):
        """Gets mixed features from locals."""
        pass
        
        
        
    def server_aligner(self):
        """Post-trains server model by alignment set."""
        pass
    
    
    def _alignment_set_builder(self):
        """Builds alignment set by collected mixed local features."""
        pass
    
    def _discard_sets(self):
        "Discards collected alignment set"
        pass
    
    