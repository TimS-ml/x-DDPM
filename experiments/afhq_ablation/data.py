"""AFHQv2 cat+dog @ 64x64 data pipeline.

Reads pre-resized PNGs from ``<root>/afhqv2_catdog_64/{cat,dog}/*.png`` (built by
the one-shot streaming download of ``huggan/AFHQv2``; see the autonomy log for
the exact provenance).

First campaign on AFHQ is unconditional: the library's ``FIDEvaluation`` calls
``next(dl)`` and expects a bare image tensor, and ``GaussianDiffusion.forward``
does not accept classes; adding a conditional path across all three backbones
plus both diffusion wrappers is a larger surgery than this run needs. Labels are
kept on disk (folder layout) so a future class-conditional campaign can pick
them up without redownloading.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class AFHQCatDog(Dataset):
    """Cat+dog images from ``<root>/afhqv2_catdog_64/{cat,dog}/*.png``.

    Yields only the image tensor (labels dropped) so it drops into the existing
    training and FID paths unchanged.
    """

    SUBDIR = "afhqv2_catdog_64"
    CLASSES = ("cat", "dog")

    def __init__(self, root: str, transform=None):
        base = Path(root) / self.SUBDIR
        if not base.is_dir():
            raise FileNotFoundError(
                f"expected AFHQ cat+dog images under {base}; run the dataset "
                f"prep step in the autonomy log to populate it"
            )
        paths: list[Path] = []
        for cls in self.CLASSES:
            cls_dir = base / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"missing class directory {cls_dir}")
            paths.extend(sorted(cls_dir.glob("*.png")))
        if not paths:
            raise RuntimeError(f"no .png files under {base}/{{cat,dog}}")
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_dataloader(
    root: str,
    batch_size: int,
    train: bool = True,
    augment: bool = True,
    num_workers: int = 4,
    download: bool = True,  # kept for signature-compat with the CIFAR harness
) -> DataLoader:
    """Return an infinite-friendly ``DataLoader`` over AFHQ cat+dog @ 64x64.

    ``download`` is ignored (data prep is a one-shot step; see the autonomy log).
    ``train`` currently has no effect: this campaign uses the single combined
    cat+dog split (~10.7k images) for both training and FID's real-image stats.
    """
    del download, train  # both consumed for signature-compat only

    tfs = []
    if augment:
        tfs.append(transforms.RandomHorizontalFlip())
    tfs.append(transforms.ToTensor())  # -> [0, 1], (C, H, W)
    transform = transforms.Compose(tfs)

    ds = AFHQCatDog(root=root, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
