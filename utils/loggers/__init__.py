"""Logging utils."""

import json
import os
import warnings
from pathlib import Path

import torch

from utils.general import LOGGER, prefix

from torch.utils.tensorboard import SummaryWriter


LOGGERS = ("csv", "tb")
RANK = int(os.getenv("RANK", -1))

class Loggers:
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  
        self.include = include
        self.keys = [
            "train/mse_loss",
            "val/mse_loss",
            "val/psnr",
            "x/lr",
        ] 
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        if "tb" in self.include:
            self.logger.info(f"{prefix('[TensorBoard]')}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

    def on_fit_epoch_end(self, vals, epoch):

        LOGGER.info(prefix("[fit_epoch_end]") + f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        LOGGER.info(prefix("[fit_epoch_end]") + f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        x = dict(zip(self.keys, vals))
        
        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = "" if file.exists() else (
                ("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n %
                        tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

