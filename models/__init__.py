import torch

from .dprc import DPRC
from .holonet import HoloNet
from .nhvc import NHVC


def model_selector(model_name: str, config: dict, device: torch.device):

    model_name = str.lower(model_name)

    if model_name == "holonet":
        return HoloNet(config, device).to(device)
    elif model_name == "nhvc":
        return NHVC(config, device).to(device)
    elif model_name == "dprc":
        return DPRC(config, device).to(device)
    else:
        raise ValueError(f"model_name: {model_name} doesn't exist")
