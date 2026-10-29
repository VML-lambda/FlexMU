
import argparse
import json
import cv2
import csv
import os
import subprocess
import sys, math
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import model_selector
from utils.loggers import LOGGER
from utils.general import print_args

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from utils.general import (
    TQDM_BAR_FORMAT,
    prefix,
    increment_path,
    check_yaml,
    yaml_save,
    check_dataset
)
from utils.dataloader import create_dataloader
from utils.operations import crop_image, magnitude_spectrum, phasemap_8bit
from utils.torch_utils import select_device

from torchmetrics.image import PeakSignalNoiseRatio as PSNR

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

def compute_bpp(amp, likelihoods):
    size = amp.size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(l).sum() / (-math.log(2) * num_pixels) 
               for l in likelihoods.values()).item()


@torch.inference_mode()
def val(
    model,
    dataloader,
    config,
    fn_prefix = 'val',
    sample_fn = '0887.png',
    savedir = None,
    save_all_poh = False,
    save_all_recon = False,
    device:torch.device ='cpu',  
):
    """Evaluates model on a dataset and logs performance metrics, results are saved to specific directories."""
    LOGGER.info(prefix("[VAL]") + "Validation Start")
    nb = len(dataloader)
    
    # Configure
    model = model.to(device)
    model.eval()
    
    sample_recon, sample_poh = None, None
    all_poh_dir, all_recon_dir = None, None
    per_image_metrics = []
    if savedir is not None and save_all_poh:
        all_poh_dir = savedir / "all_poh"
        all_poh_dir.mkdir(parents=True, exist_ok=True)
    if savedir is not None and save_all_recon:
        all_recon_dir = savedir / "all_recon"
        all_recon_dir.mkdir(parents=True, exist_ok=True)

    # loss
    recon_loss = nn.MSELoss().to(device)
    psnr_loss = PSNR(data_range=1.0, reduction='elementwise_mean').to(device)
    mloss = torch.zeros(2)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

    with torch.no_grad():
        for i, (imgs, masks, paths) in pbar:
            imgs = imgs.float() / 255.
            imgs = torch.where(imgs <= 0.04045, imgs / 12.92, ((imgs + 0.055) / 1.055)**(2.4)) # Linearize Intensity 
            imgs = torch.sqrt(imgs) # to Amplitude
            imgs = imgs.to(device, non_blocking=True)

            T = config['target']
            roi_res     = T['roi_res']

            z = model(imgs) 
            slm = torch.exp(1j*z['output_poh'])
            prop = model.propagator_forward(slm)
            recon_field = prop['g_out']

            if all_poh_dir is not None:
                for batch_idx, img_path in enumerate(paths):
                    poh_name = Path(img_path).name
                    poh_path = all_poh_dir / poh_name
                    poh_map = phasemap_8bit(z['output_poh'][batch_idx].detach().permute((1, 2, 0)))
                    cv2.imwrite(str(poh_path), poh_map)

            recon_amp = recon_field.abs()
            recon_amp = crop_image(recon_amp, roi_res)

            target_amp = crop_image(imgs, roi_res)
            recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

            batch_mse = recon_loss(recon_amp , target_amp).item()
            batch_psnr = psnr_loss(recon_amp , target_amp).item()
            mloss[0] += batch_mse
            mloss[1] += batch_psnr

            for batch_idx, img_path in enumerate(paths):
                filename = Path(img_path).name
                per_image_metrics.append({
                    "filename": filename,
                    "mse": batch_mse,
                    "psnr": batch_psnr,
                })

                if all_recon_dir is not None:
                    recon_path = all_recon_dir / filename
                    recon_img = (recon_amp[batch_idx].detach().permute((1, 2, 0)).cpu().numpy().clip(0, 1) * 255.0).round().astype(np.uint8)
                    cv2.imwrite(str(recon_path), recon_img)

            if paths[0].endswith(sample_fn):
                sample_recon = recon_amp[0]
                sample_poh = z['output_poh'][0]

                magnitude = torch.abs(prop['G_out'][0])
                log_magnitude = torch.log1p(magnitude).permute(1,2,0)
                
                log_magnitude_np = log_magnitude.squeeze().cpu().numpy()
                log_magnitude_np -= log_magnitude_np.min()  
                log_magnitude_np /= log_magnitude_np.max()  
                log_magnitude_np *= 255                    
                log_magnitude_np = log_magnitude_np.astype(np.uint8)
                
                hist = torch.histc(sample_poh.cpu().flatten(),bins=255,min=-3.5,max=3.5)
                bins = torch.linspace(-3.5, 3.5, steps=256)


    total_loss = (mloss.cpu() / len(dataloader)).tolist()

    if savedir != None:
        recon_fn = str(savedir / f"{fn_prefix}_recon.png")
        poh_fn = str(savedir / f"{fn_prefix}_poh.png")
        spec_fn = str(savedir / f"{fn_prefix}_spec.png")
        hist_fn = str(savedir / f"{fn_prefix}_hist.png")
        cv2.imwrite(recon_fn, (sample_recon.detach().permute((1,2,0)).cpu().numpy().clip(0,1) * 255.).astype(int))
        cv2.imwrite(poh_fn, phasemap_8bit(sample_poh.detach().permute((1,2,0))))
        cv2.imwrite(spec_fn, log_magnitude_np)
        
        plt.bar(bins[:-1].cpu().numpy(), hist.cpu().numpy(), width=0.1, alpha=0.7)
        plt.savefig(hist_fn)
        plt.close()

    if savedir is not None and per_image_metrics:
        metrics_path = savedir / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "mse", "psnr"])
            writer.writeheader()
            writer.writerows(per_image_metrics)

    return total_loss[0], total_loss[1], per_image_metrics

def main(opt):
    save_dir, config, data = (
        Path(opt.save_dir),
        opt.config,
        opt.data
    )

    val_save_dir = save_dir / "val"
    val_save_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    config = check_yaml(config)
    if isinstance(config, str):
        with open(config, errors="ignore") as f:
            config = yaml.safe_load(f)  # load configs dict
    LOGGER.info(prefix("[Configurations]") + ", ".join(f"{k}={v}" for k, v in config.items()))
    opt.config = config.copy()  # for saving configs to checkpoints

    device = select_device(opt.device, batch_size=1)

    # Save run settings
    yaml_save(val_save_dir / "config.yaml", config)
    yaml_save(val_save_dir / "opt.yaml", vars(opt))

    data = check_yaml(data)
    data_dict = check_dataset(data)  # check if None
    val_path = data_dict["val"]

    # Model
    M = config['model']
    model_name = M['name']
    model = model_selector(model_name, config, device)
    weight = torch.load(opt.weight, map_location=device)['model']

    torch.save({'model': weight}, val_save_dir / 'weight.pt')

    model.load_state_dict(weight)
    model = model.to(device)

    val_loader, _ = create_dataloader(
        val_path,
        batch_size = 1,
        config=config['data'],
        augment=False,
        cache='ram',
        rank=-1,
        workers=2,
        prefix=prefix("[valid loader]"),
        shuffle=False,
        seed=opt.seed
    )   

    mse, psnr, per_image_metrics = val(
        model      = model, 
        dataloader = val_loader, 
        config     = config, 
        savedir    = val_save_dir,
        save_all_poh = opt.save_all_poh,
        save_all_recon = opt.save_all_recon,
        device     = device,
    )
    
    header = ("%11s  " * 2) % ("MSE", "PSNR")
    values = ("%11f  " * 2) % (mse, psnr)

    with open(val_save_dir / 'output.txt', 'w') as f:
        f.write(header + '\n')
        f.write(values + '\n')

    summary = {
        "num_images": len(per_image_metrics),
        "mse": mse,
        "psnr": psnr,
    }
    with open(val_save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="baseline.yaml")
    parser.add_argument("--data",       type=str, default="div2k.yaml")
    parser.add_argument("--weight",     type=str, default=None, help='pretrained/nhvc_base_g.pt')
    parser.add_argument("--project",    type=str, default="NHVC")
    parser.add_argument("--name",       type=str, default="pretrained")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save-all-poh", action="store_true", help="save POH for every validation image")
    parser.add_argument("--save-all-recon", action="store_true", help="save recon for every validation image")

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    print_args(vars(opt))
    assert opt.weight != None, "weight shouldn't be None"
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=True))
    main(opt)
