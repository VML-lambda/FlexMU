
import torch
import torch.nn as nn

from utils.propagation import kernel_factory, PropagatorBase
from .activations import activation_selector

class BaseNet(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super(BaseNet, self).__init__()

        M = config['model']
        self.activation_init = activation_selector(M['activation_init'])
        self.activation_last = activation_selector(M['activation_last'])

        P = config['propagation_config']
        forward_config = P['forward']
        backward_config = P['backward']
        # I promise no use of variable "config" below

        self.H_forward = kernel_factory(forward_config, device)
        self.H_backward = kernel_factory(backward_config, device)
            
        self.propagator_forward = PropagatorBase(self.H_forward, forward_config['isQE'])
        self.propagator_backward = PropagatorBase(self.H_backward, backward_config['isQE'])

        self.devive = device
        
        
    def forward(self, target_amp):
        pass  