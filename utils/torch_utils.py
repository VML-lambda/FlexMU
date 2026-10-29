from utils.general import LOGGER, prefix

import os
import math
from copy import deepcopy
from contextlib import contextmanager

import torch.distributed as dist
import torch
import torch.nn as nn

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Context manager ensuring ordered operations in distributed training by making all processes wait for the leading
    process.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

def smart_optimizer(model, lr=0.001):
    g = [[], []]
    seen = set()
    for v in model.modules():
        for p_name, p in v.named_parameters():
            if id(p) in seen:  
                continue
            seen.add(id(p))  
            if p_name.endswith("quantiles"):
                g[1].append(p)
            else:
                g[0].append(p)

    net_optimizer = torch.optim.Adam(
        [p for p in g[0] if p.requires_grad], lr=lr
    )  

    aux_optimizer = torch.optim.Adam(
        g[1], lr=lr
    ) if len(g[1]) > 0 else None

    LOGGER.info(
        f"{prefix('[net_optimizer]')}{type(net_optimizer).__name__}(lr={lr}) with parameter groups \n" 
        f"{prefix('[aux_optimizer]')}{type(aux_optimizer).__name__}(lr={lr}) with parameter groups \n"
        f"{prefix('')}{len(g[0])} model param, {len(g[1])} quantiles \n"
    )
    return net_optimizer, aux_optimizer

def smart_resume(
    ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True, device="cuda"
):
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            for param in optimizer.state.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)  # ✅ 옵티마이저 state를 `device`로 변환
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(device)
        except:
            LOGGER.info(prefix('[optimizer]') + 'optimizer load failed but continue')
            pass
    
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(
            f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs"
        )
    if epochs < start_epoch:
        LOGGER.info(
            f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs."
        )
        epochs += ckpt["epoch"]  # finetune additional epochs
    return start_epoch, epochs

def select_device(device="", batch_size=0, newline=True):
    s = ""
    device = (
        str(device).strip().lower().replace("cuda:", "").replace("none", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device 
        # set environment variable - must be before assert is_available()
        
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = (
            device.split(",") if device else "0"
        )  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert (
                batch_size % n == 0
            ), f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " "
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            # bytes to MB
            s = f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = "cuda:0"

    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)

def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )

def de_parallel(model):
    return model.module if is_parallel(model) else model

def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  
        # decay exponential ramp (to help early epochs)

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        copy_attr(self.ema, model, include, exclude)


