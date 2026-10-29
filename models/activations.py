import numpy as np
from typing import Tuple

import torch
import torch.nn as nn


def activation_selector(
    act_config: dict
):
    A = act_config
    act_name = A['name']
    alpha    = A.get('alpha', None)
    beta     = A.get('beta', None)

    act_name = str.lower(act_name)
    if act_name == 'tanh':
        return nn.Hardtanh(-torch.pi, torch.pi)
    elif act_name == 'flexmu':
        return FlexMU(alpha=alpha, beta=beta)
    else:
        raise ValueError(f"act_name: {act_name} doesn't exist")
    

class FlexMU(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 0.318):
        super().__init__()
        assert 0 <= alpha <= 1.0, f"The alpha in FlexMU shoud be in range [0,1] your alpha is {alpha}"
        if alpha < 0.1:
            alpha = 0.1  # for numerical safeguard

        self.alpha = alpha
        self.beta = beta
        div = torch.arctanh(torch.tensor([self.alpha]))
        self.register_buffer("div", div)

    def forward(self, x):
        z = (self.beta*x + 1.0) % 2.0 - 1.0
        out = torch.arctanh(self.alpha * z) / self.div
        return torch.pi * out
