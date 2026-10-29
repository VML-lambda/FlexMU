
import glob
import math
import cv2
import os
import random
import shutil
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

from utils.operations import srgb_gamma2lin, pad_crop_to_res

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset, dataloader, distributed
from tqdm import tqdm

from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    NUM_THREADS,
)
from utils.torch_utils import torch_distributed_zero_first

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower(
) == "true"  # global pin_memory for dataloaders

color_map = {'b': 0,'g': 1,'r': 2, 'rgb':3}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(
    path,
    batch_size,
    config=None,
    augment=False,
    cache=False,
    rank=-1,
    workers=8,
    prefix="",
    shuffle=False,
    seed=0,
):
    with torch_distributed_zero_first(rank):
        dataset = AmplitudeDataset(
            path,
            batch_size,
            config=config,  
            augment=augment, 
            cache_images=cache,
            prefix=prefix,
            rank=rank,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size >
             1 else 0, workers])  # number of workers
    sampler = None if rank == - \
        1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    # only DataLoader allows for attribute updates
    loader = InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=AmplitudeDataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset

# Inherit from DistributedSampler and override iterator
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        """Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) /
                self.num_replicas) + 1  # num_replicas == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx))
                        )[:padding_size]

        return iter(idx)

class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler",
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)

class AmplitudeDataset(Dataset):
    def __init__(
        self,
        path,
        batch_size=16,
        config=None,
        augment=False,
        cache_images=False,
        prefix="",
        rank=-1,
        seed=0,
    ):
        self.path           = path
        self.augment        = augment
        self.image_res      = config['image_res']
        self.homography_res = config['homography_res']
        self.channel        = config['channel']
        self.flipud         = config['flipud']
        self.fliplr         = config['fliplr']
        
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1)
                              if x.startswith("./") else x for x in t]
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep)
                                   for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(
                f"{prefix}Error loading data from {path}: {e}") from e

        # Create indices
        n = len(self.im_files)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        if rank > -1:  # DDP indices (see: SmartDistributedSampler)
            # force each rank (i.e. GPU process) to sample the same subset of data on every epoch
            self.indices = self.indices[np.random.RandomState(
                seed=seed).permutation(n) % WORLD_SIZE == RANK]

        # Cache images into RAM for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.load_image
            results = ThreadPool(NUM_THREADS).imap(
                lambda i: (i, fcn(i)), self.indices)
            pbar = tqdm(results, total=len(self.indices), bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                self.ims[i] = x
                b += self.ims[i].nbytes * WORLD_SIZE
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """Checks if available RAM is sufficient for caching images, adjusting for a safety margin."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio_h = self.image_res[0] / im.shape[0]
            ratio_w = self.image_res[1] / im.shape[1]
            b += im.nbytes * ratio_h * ratio_w
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        # to cache or not to cache, that is the question
        cache = mem_required * (1 + safety_margin) < mem.available
        if not cache:
            LOGGER.info(
                f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.im_files)

    def __getitem__(self, index):
        """Fetches the dataset item at the given index, considering linear, shuffled, or weighted sampling."""
        index = self.indices[index]

        # Load image
        img = self.load_image(index)

        if self.augment:
            # Flip up-down
            if random.random() < self.flipud:
                img = np.flipud(img)

            # Flip left-right
            if random.random() < self.fliplr:
                img = np.fliplr(img)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        img = pad_crop_to_res(img, self.homography_res) 
        img = pad_crop_to_res(img, self.image_res)

        # Generate Mask
        mask = np.ones(img.shape[:])
        mask = pad_crop_to_res(mask, self.homography_res)
        mask = pad_crop_to_res(mask, self.image_res)
        mask = mask.astype(bool)

        return torch.from_numpy(img), torch.from_numpy(mask), self.im_files[index]

    def load_image(self, i):
        channel = color_map[self.channel]
        im, f = (
            self.ims[i],
            self.im_files[i],
        )
        if im is None:  # not cached in RAM
            im = cv2.imread(f)[...,channel:channel+1]  # BGR use c:c+1 to maintain tensor size
            assert im is not None, f"Image Not Found {f}"
        return im

    @staticmethod
    def collate_fn(batch):
        im, mask, path = zip(*batch)  # transposed
        return torch.stack(im, 0), torch.stack(mask, 0), path
    
