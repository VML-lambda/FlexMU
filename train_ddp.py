import argparse
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import val
import yaml
from models import model_selector
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.callbacks import Callbacks
from utils.dataloader import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_yaml,
    increment_path,
    init_seeds,
    methods,
    prefix,
    print_args,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.operations import crop_image
from utils.torch_utils import (
    de_parallel,
    select_device,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

torch._dynamo.config.optimize_ddp = False

# Root directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# For multi-GPU
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def parse_pretrained_weight(value: str):
    normalized = value.strip().lower()
    if normalized in {"false", "none", "null", "no", "0"}:
        return False
    return value


def apply_cli_overrides(config: dict, opt) -> list[str]:
    applied = []

    def apply_value(path: str, value: Any):
        set_nested(config, path, value)
        applied.append(f"{path}={value}")

    if opt.channel is not None:
        channel = opt.channel.lower()
        apply_value("data.channel", channel)

        wavelength_map = config.get("wavelength", {})
        if channel not in wavelength_map:
            raise KeyError(
                f"channel '{channel}' is missing in config['wavelength']; available={list(wavelength_map.keys())}"
            )
        wavelength = wavelength_map[channel]
        prop_cfg = config.get("propagation_config", {})
        for direction in ("forward", "backward"):
            if isinstance(prop_cfg.get(direction), dict):
                prop_cfg[direction]["wavelength"] = wavelength
        applied.append(f"propagation_config.[forward|backward].wavelength={wavelength}")

    if opt.pretrained_weight is not None:
        apply_value("model.pretrained_weight", parse_pretrained_weight(opt.pretrained_weight))

    if opt.epochs is not None:
        apply_value("training.epochs", opt.epochs)
    if opt.lr0 is not None:
        apply_value("training.lr0", opt.lr0)
    if opt.nbs is not None:
        apply_value("training.nbs", opt.nbs)
    if opt.sampling_interval is not None:
        apply_value("sampling_interval", opt.sampling_interval)

    if opt.act_init_name is not None:
        apply_value("model.activation_init.name", opt.act_init_name.lower())
    if opt.act_init_alpha is not None:
        apply_value("model.activation_init.alpha", opt.act_init_alpha)
    if opt.act_init_beta is not None:
        apply_value("model.activation_init.beta", opt.act_init_beta)

    if opt.act_last_name is not None:
        apply_value("model.activation_last.name", opt.act_last_name.lower())
    if opt.act_last_alpha is not None:
        apply_value("model.activation_last.alpha", opt.act_last_alpha)
    if opt.act_last_beta is not None:
        apply_value("model.activation_last.beta", opt.act_last_beta)

    for raw_item in opt.set_items:
        if "=" not in raw_item:
            raise ValueError(f"--set expects KEY=VALUE, got '{raw_item}'")
        key, raw_value = raw_item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set key is empty in '{raw_item}'")
        value = yaml.safe_load(raw_value)
        apply_value(key, value)

    return applied


def train(opt, device, callbacks):
    save_dir, batch_size, data, config, workers = (
        Path(opt.save_dir),
        opt.batch_size,
        opt.data,
        opt.config,
        opt.workers,
    )
    # Cuda
    cuda = device.type != "cpu"

    # Directories
    w_save_dir = save_dir / "weights"
    w_save_dir.mkdir(parents=True, exist_ok=True)
    last, best = w_save_dir / "last.pt", w_save_dir / "best.pt"

    sample_save_dir = save_dir / "samples"
    sample_save_dir.mkdir(parents=True, exist_ok=True)

    val_save_dir = save_dir / "val"
    val_save_dir.mkdir(parents=True, exist_ok=True)

    # Configurations
    config = check_yaml(config)  # .yaml auto search from folder
    if isinstance(config, str):
        with open(config, errors="ignore") as f:
            config = yaml.safe_load(f)
    applied_overrides = apply_cli_overrides(config, opt)
    if applied_overrides:
        LOGGER.info(prefix("[Overrides]") + ", ".join(applied_overrides))
    LOGGER.info(
        prefix("[Configurations]") + ", ".join(f"{k}={v}" for k, v in config.items())
    )
    opt.config = config.copy()  # for saving hyps to checkpoints

    # Save run settings
    yaml_save(save_dir / "config.yaml", config)
    yaml_save(save_dir / "config_resolved.yaml", config)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        loggers = Loggers(
            save_dir=save_dir,
            opt=opt,
            logger=LOGGER,
            include=tuple(include_loggers),
        )
        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Data
    data_dict = None
    init_seeds(opt.seed + 1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data = check_yaml(data)
        data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]

    # Model
    M = config["model"]
    model_name = M["name"]
    model = model_selector(model_name, config, device)
    pretrained_weight = M.get("pretrained_weight", False)

    # training
    T = config["training"]
    lr0 = T["lr0"]
    nbs = T["nbs"]  # nominal batch size
    epochs = T["epochs"]

    # accumulate loss before optimizing
    # This ensures the theoretical batch size is maintained even when the GPU memory is insufficient.
    accumulate = max(round(nbs / batch_size), 1)
    optimizer, _ = smart_optimizer(model, lr0)

    # resume
    resume = last.exists()

    if resume:
        ckpt = torch.load(last, map_location="cpu")
        csd = ckpt["model"]
        model.load_state_dict(csd, strict=False)  # load
        model = model.to(device)
        LOGGER.info(
            f"{prefix('[resume]')} Transferred {len(csd)}/{len(model.state_dict())} items from {last}"
        )

    if pretrained_weight and not resume:
        ckpt = torch.load(pretrained_weight)
        csd = ckpt["model"]
        model.load_state_dict(csd, strict=False)
        model = model.to(device)
        LOGGER.info(f"{prefix('[model]')} weight loaded from {pretrained_weight}")

    # Compile
    # model = torch.compile(model)

    # Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.8, patience=500
    )

    # Resume
    start_epoch = 0
    if resume:
        start_epoch, epochs = smart_resume(
            ckpt=ckpt,
            optimizer=optimizer,
            weights=last,
            epochs=epochs,
            resume=resume,
            device=device,
        )
        del ckpt, csd

    # Train loader
    train_loader, _ = create_dataloader(
        train_path,
        batch_size=batch_size // WORLD_SIZE,
        config=config["data"],
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=workers,
        prefix=prefix("[train loader]"),
        shuffle=True,
        seed=opt.seed,
    )

    # Validation loader
    if RANK in {-1, 0}:
        val_loader, _ = create_dataloader(
            val_path,
            batch_size=1,
            config=config["data"],
            augment=False,
            cache=opt.cache,
            rank=-1,
            workers=2,
            prefix=prefix("[valid loader]"),
            shuffle=False,
            seed=opt.seed,
        )

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )

    # Start training
    nb = len(train_loader)  # number of batches
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move

    # loss
    recon_loss = nn.MSELoss().to(device)
    best_loss = float("inf")
    mse, psnr = 0, 0

    # Forward propagation
    T = config["target"]
    roi_res = T["roi_res"]
    recon_scale = T["recon_scale"]

    callbacks.run("on_train_start")
    LOGGER.info(
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {save_dir}\n"
        f"Starting training for {epochs} epochs..."
    )

    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        callbacks.run("on_train_epoch_start")
        model.train()

        mloss = torch.zeros(1, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)

        LOGGER.info(
            ("\n" + "%12s" * 5)
            % ("Epoch", "GPU_mem", "train_loss", "Instances", "Size")
        )
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()

        # batch -------------------------------------------------------------
        for i, (imgs, masks, paths) in pbar:
            callbacks.run("on_train_batch_start")
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.float() / 255.0
            imgs = torch.where(
                imgs <= 0.04045, imgs / 12.92, ((imgs + 0.055) / 1.055) ** (2.4)
            )  # Linearize Intensity
            imgs = torch.sqrt(imgs)  # to Amplitude
            imgs = imgs.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                z = model(imgs)
                slm = torch.exp(1j * z["output_poh"])
                prop_forward = de_parallel(model).propagator_forward(slm)

                recon_amp = prop_forward["g_out"].abs()
                recon_amp = crop_image(recon_amp, roi_res)
                target_amp = crop_image(imgs, roi_res)

                loss = recon_loss(recon_amp * recon_scale, target_amp)

            loss.backward()

            if ni - last_opt_step >= accumulate:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                # update mean losses
                mloss = (mloss * i + loss.item()) / (i + 1)
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 1) % (f"{epoch}/{epochs - 1}", mem, *mloss)
                )
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step(mloss.item())

        if RANK in {-1, 0}:
            if epoch % config["sampling_interval"] == 0:
                mse, psnr, _ = val.val(
                    model=de_parallel(model).to(device),
                    dataloader=val_loader,
                    config=config,
                    fn_prefix=f"{epoch}",
                    savedir=sample_save_dir,
                    device=device,
                )
                LOGGER.info(prefix("[valid]") + ("%10s" * 2) % ("mse", "psnr"))
                LOGGER.info(prefix("[valid]") + ("%10f" * 2) % (mse, psnr))

            log_vars = list(mloss) + [mse, psnr] + lr
            callbacks.run("on_fit_epoch_end", log_vars, epoch + 1)

            # Save model
            ckpt = {
                "epoch": epoch,
                "loss": mloss,
                "model": deepcopy(de_parallel(model)).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "opt": vars(opt),
                "date": datetime.now().isoformat(),
            }

            if mloss[0] < best_loss:
                best_loss = mloss[0]
                best_epoch = epoch

            # Save last, best and delete
            torch.save(ckpt, last)
            if epoch == best_epoch:
                torch.save(ckpt, best)
            if opt.save_period > 0 and epoch % opt.save_period == 0:
                torch.save(ckpt, w_save_dir / f"epoch{epoch}.pt")
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        for f in last, best:
            if f.exists():
                if f is best:
                    ckpt = torch.load(f, map_location="cpu")
                    model = model_selector(model_name, config, device)
                    csd = ckpt["model"]
                    model.load_state_dict(csd)
                    model = model.to(device)

                    mse, psnr = val.val(
                        model=model,
                        dataloader=val_loader,
                        config=config,
                        fn_prefix=f"best_epoch_{best_epoch}",
                        savedir=val_save_dir,
                        device=device,
                    )

        LOGGER.info(prefix("[last valid]") + ("%10s" * 2) % ("mse", "psnr"))
        LOGGER.info(prefix("[last valid]") + ("%10f" * 2) % (mse, psnr))

        header = ("%11s  " * 2) % ("MSE", "PSNR")
        values = ("%11f  " * 2) % (mse, psnr)

        with open(val_save_dir / "output.txt", "w") as f:
            f.write(header + "\n")
            f.write(values + "\n")

    torch.cuda.empty_cache()
    return 0


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config.yaml path")
    parser.add_argument("--data", type=str, help="dataset.yaml path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--cache", type=str, default="ram", help="image --cache ram/disk"
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    parser.add_argument("--project", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every x epochs (disabled if < 1)",
    )
    parser.add_argument("--seed", type=str, default="42", help="Global training seed")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Automatic DDP Multi-GPU argument, do not modify",
    )
    parser.add_argument(
        "--channel",
        type=str,
        choices=("r", "g", "b"),
        default=None,
        help="Override data.channel and propagation wavelength",
    )
    parser.add_argument(
        "--pretrained-weight",
        type=str,
        default=None,
        help="Override model.pretrained_weight (use 'false' to disable)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--lr0", type=float, default=None, help="Override training.lr0")
    parser.add_argument("--nbs", type=int, default=None, help="Override training.nbs")
    parser.add_argument(
        "--sampling-interval",
        type=int,
        default=None,
        help="Override sampling_interval",
    )
    parser.add_argument(
        "--act-init-name",
        type=str,
        default=None,
        help="Override model.activation_init.name",
    )
    parser.add_argument(
        "--act-init-alpha",
        type=float,
        default=None,
        help="Override model.activation_init.alpha",
    )
    parser.add_argument(
        "--act-init-beta",
        type=float,
        default=None,
        help="Override model.activation_init.beta",
    )
    parser.add_argument(
        "--act-last-name",
        type=str,
        default=None,
        help="Override model.activation_last.name",
    )
    parser.add_argument(
        "--act-last-alpha",
        type=float,
        default=None,
        help="Override model.activation_last.alpha",
    )
    parser.add_argument(
        "--act-last-beta",
        type=float,
        default=None,
        help="Override model.activation_last.beta",
    )
    parser.add_argument(
        "--set",
        dest="set_items",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Generic override using dotted keys, e.g. --set target.recon_scale=0.9",
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):

    if RANK in {-1, 0}:
        print_args(vars(opt))

    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=True))
    opt.seed = int(opt.seed)

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, (
            f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        )
        assert torch.cuda.device_count() > LOCAL_RANK, (
            "insufficient CUDA devices for DDP command"
        )
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),
        )

    # Train
    train(opt, device, callbacks)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
