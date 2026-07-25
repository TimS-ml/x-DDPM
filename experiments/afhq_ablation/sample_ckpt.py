"""Generate a sample grid PNG from a trained checkpoint's EMA weights.

Usage:
    python experiments/afhq_ablation/sample_ckpt.py --run unet-ddpm [--n 64] [--ckpt best.pt]

Rebuilds the run's diffusion wrapper, loads the EMA weights from the checkpoint,
samples `n` images (perfect square -> sqrt(n) x sqrt(n) grid) and writes a PNG to
results/<run>/samples/<ckpt_stem>_grid.png. Independent of the training process.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ema_pytorch import EMA  # noqa: E402

from configs import RUNS  # noqa: E402
from backbones import build_backbone  # noqa: E402
from diffusions import build_diffusion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--results-root", default=str(HERE / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = RUNS[args.run]
    device = "cuda"
    run_dir = Path(args.results_root) / cfg.name
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    backbone = build_backbone(cfg)
    diffusion = build_diffusion(cfg, backbone).to(device)
    ema = EMA(diffusion, beta=cfg.ema_decay, update_every=cfg.ema_update_every)
    ema.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema.load_state_dict(ckpt["ema"])
    step = ckpt.get("step", -1)
    best = ckpt.get("best_fid", float("nan"))
    print(f"[{cfg.name}] loaded {ckpt_path.name} step={step} best_fid={best:.4f}", flush=True)

    ema.ema_model.eval()
    with torch.inference_mode():
        imgs = ema.ema_model.sample(batch_size=args.n)
    grid = make_grid(imgs.clamp(0, 1).cpu(), nrow=int(math.sqrt(args.n)))

    out = Path(args.out) if args.out else run_dir / "samples" / f"{ckpt_path.stem}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out)
    print(f"[{cfg.name}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
