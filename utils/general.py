import os
import yaml
import glob
import random
import inspect
import numpy as np
import logging
import logging.config
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# [LOGGING] #
NUM_THREADS = min(
    8, max(1, os.cpu_count() - 1)
)
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format

LOGGING_NAME = "NHVC"


def set_logging(name, verbose=True):
    """Configures logging with specified verbosity; `name` sets the logger's name, `verbose` controls logging level."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )


set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(
    LOGGING_NAME
)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

def prefix(s, prefix_width=17):
    return s.ljust(prefix_width)

# [SEED] #

def init_seeds(seed=42, deterministic=False):
    """
    Initializes RNG seeds and sets deterministic options if specified.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)

def check_suffix(file="example.yaml", suffix=(".yaml",), msg=""):
    """Validates if a file or files have an acceptable suffix, raising an error if not."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def check_file(file, suffix=""):
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()

    if os.path.isfile(file) or not file:  # exists
        return file

    else:  # search
        files = []
        for d in "data", "src/models", "config":  # search directories
            files.extend(
                glob.glob(str(ROOT / d / "**" / file), recursive=True)
            )  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file
    
def check_yaml(file, suffix=(".yaml", ".yml")):
    """Searches/downloads a YAML file, verifies its suffix (.yaml or .yml), and returns the file path."""
    return check_file(file, suffix)

def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

def yaml_save(file="data.yaml", data=None):
    """Safely saves `data` to a YAML file specified by `file`, converting `Path` objects to strings; `data` is a
    dictionary.
    """
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in data.items()},
            f,
            sort_keys=False,
        )

def methods(instance):
    """Returns list of method names for a class/instance excluding dunder methods."""
    return [
        f
        for f in dir(instance)
        if callable(getattr(instance, f)) and not f.startswith("__")
    ]

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Logs the arguments of the calling function, with options to include the filename and function name."""
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info((s) + ", ".join(f"{k}={v}" for k, v in args.items()))

def check_dataset(data):
    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in "train", "val":
        assert k in data, f"data.yaml '{k}:' field missing"

    # Resolve paths
    # optional 'path' default to '.'
    path = Path(data.get("path"))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path
    for k in "train", "val":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    return data

def check_poh_target_dataset(data):
    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in "train", "val":
        assert k in data, f"data.yaml '{k}:' field missing"

    poh_path = Path(data.get("poh_path"))
    target_path = Path(data.get("target_path"))

    for k in "train", "val":
        if data.get(k):  # prepend path
            xp = (poh_path / data[k]).resolve()
            xt = (target_path / data[k]).resolve()
            data["poh_" + k] = str(xp)
            data["target_" + k] = str(xt)

    return data
