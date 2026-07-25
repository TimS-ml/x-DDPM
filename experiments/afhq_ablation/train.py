"""Single-run trainer for the CIFAR-10 ablation.

Usage:
    python experiments/afhq_ablation/train.py --run unet-ddpm [--smoke]

Custom loop (approach B): keeps full control over EMA cadence, sampling/FID
timing, wandb logging and checkpoint/resume. Reuses the library's backbones,
diffusion wrappers and FIDEvaluation; does not touch library source.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ema_pytorch import EMA  # noqa: E402
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation  # noqa: E402

from configs import RUNS  # noqa: E402
from backbones import build_backbone, count_params, human  # noqa: E402
from diffusions import build_diffusion  # noqa: E402
from data import build_dataloader, cycle  # noqa: E402


def load_dotenv(path: Path) -> None:
    """Minimal .env parser; maps non-standard WANDB_API -> WANDB_API_KEY."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    if "WANDB_API_KEY" not in os.environ and "WANDB_API" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_API"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_ctx(dtype_str: str):
    if dtype_str == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if dtype_str == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--results-root", default=None,
                    help="default: results/ (real) or results_smoke/ (--smoke)")
    ap.add_argument("--wandb-mode", default=None, help="online | offline | disabled")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    cfg = RUNS[args.run]
    if args.smoke:
        cfg = cfg.smoke()

    # keep smoke artifacts in a separate tree so they never shadow real runs
    if args.results_root is None:
        args.results_root = str(HERE / ("results_smoke" if args.smoke else "results"))

    load_dotenv(REPO_ROOT / ".env")
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = "cuda"
    results_root = Path(args.results_root)
    run_dir = results_root / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    # real CIFAR stats are model-independent -> share across runs; separate cache for smoke
    fid_dir = results_root / ("_fid_cache_smoke" if args.smoke else "_fid_cache")
    fid_dir.mkdir(parents=True, exist_ok=True)

    # ---- build model / diffusion / ema / optimizer ----
    model = build_backbone(cfg).to(device)
    diffusion = build_diffusion(cfg, model).to(device)
    ema = EMA(diffusion, beta=cfg.ema_decay, update_every=cfg.ema_update_every)
    ema.to(device)
    opt = Adam(diffusion.parameters(), lr=cfg.lr)

    n_params = count_params(model)
    print(f"[{cfg.name}] backbone={cfg.backbone} diffusion={cfg.diffusion} "
          f"params={human(n_params)} steps={cfg.train_steps}", flush=True)

    # ---- data ----
    train_dl = cycle(build_dataloader(args.data_root, cfg.batch_size, train=True,
                                      augment=True, num_workers=args.num_workers))
    fid_real_dl = cycle(build_dataloader(args.data_root, cfg.fid_batch_size, train=True,
                                         augment=False, num_workers=args.num_workers))

    fid = FIDEvaluation(
        batch_size=cfg.fid_batch_size,
        dl=fid_real_dl,
        sampler=ema.ema_model,
        channels=cfg.channels,
        stats_dir=str(fid_dir),
        device=device,
        num_fid_samples=cfg.fid_samples,
    )

    # ---- resume ----
    start_step = 0
    best_fid = float("inf")
    resumed_wandb_id = None
    latest_ckpt = run_dir / "latest.pt"
    if latest_ckpt.exists():
        ck = torch.load(latest_ckpt, map_location=device)
        diffusion.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best_fid = ck.get("best_fid", float("inf"))
        resumed_wandb_id = ck.get("wandb_id")
        print(f"[{cfg.name}] resumed from step {ck['step']} (best_fid={best_fid:.3f})", flush=True)

    # ---- wandb ----
    import wandb
    mode = args.wandb_mode or ("offline" if args.smoke else "online")
    suffix = "-smoke" if args.smoke else ""
    # Recover the wandb id from the checkpoint on resume; otherwise mint a fresh
    # unique id. A fresh id avoids colliding with a deleted/tombstoned run id,
    # which makes resume="allow" hang at init (90s timeout). Display name stays
    # cfg.name so the project view is clean.
    wandb_id = resumed_wandb_id or f"{cfg.name}{suffix}-{time.strftime('%m%d%H%M%S')}"
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.name + suffix,
        id=wandb_id,
        resume="allow",
        mode=mode,
        config={**dataclasses.asdict(cfg), "n_params": n_params},
    )

    def save_ckpt(path: Path, step: int, best: float) -> None:
        torch.save({
            "step": step,
            "model": diffusion.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "best_fid": best,
            "cfg_name": cfg.name,
            "wandb_id": wandb_id,
        }, path)

    def evaluate(global_step: int) -> None:
        nonlocal best_fid
        ema.ema_model.eval()
        # sample grid from EMA weights
        try:
            with torch.inference_mode():
                imgs = ema.ema_model.sample(batch_size=cfg.sample_grid)
            grid = make_grid(imgs.clamp(0, 1).cpu(), nrow=int(math.sqrt(cfg.sample_grid)))
            save_image(grid, samples_dir / f"step_{global_step:07d}.png")
            wandb.log({"samples": wandb.Image(grid)}, step=global_step)
        except Exception as exc:  # sampling should not kill training
            print(f"[{cfg.name}] sample failed @ {global_step}: {exc}", flush=True)
        # FID from EMA weights
        try:
            score = float(fid.fid_score())
            wandb.log({"eval/fid": score}, step=global_step)
            print(f"[{cfg.name}] step {global_step} FID {score:.3f}", flush=True)
            if score < best_fid:
                best_fid = score
                save_ckpt(run_dir / "best.pt", global_step - 1, best_fid)
        except Exception as exc:
            print(f"[{cfg.name}] FID failed @ {global_step}: {exc}", flush=True)
            wandb.log({"eval/fid_error": 1.0}, step=global_step)

    # ---- train loop ----
    diffusion.train()
    torch.cuda.reset_peak_memory_stats()
    last_log_time = time.time()
    last_log_step = start_step

    for step in range(start_step, cfg.train_steps):
        img = next(train_dl).to(device)
        with amp_ctx(cfg.amp_dtype):
            loss = diffusion(img)
        loss.backward()
        clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        ema.update()

        if step % cfg.log_every == 0:
            now = time.time()
            ips = (step - last_log_step) / max(now - last_log_time, 1e-6)
            vram = torch.cuda.max_memory_allocated() / 1e9
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": opt.param_groups[0]["lr"],
                "perf/it_per_sec": ips,
                "perf/vram_gb": vram,
            }, step=step)
            print(f"[{cfg.name}] step {step} loss {loss.item():.4f} "
                  f"{ips:.2f} it/s vram {vram:.2f}G", flush=True)
            last_log_time, last_log_step = now, step

        global_step = step + 1
        if global_step % cfg.eval_every == 0 or global_step == cfg.train_steps:
            evaluate(global_step)
            save_ckpt(latest_ckpt, step, best_fid)
            diffusion.train()

    (run_dir / "done").write_text(f"step={cfg.train_steps} best_fid={best_fid:.4f}\n")
    wandb.log({"eval/best_fid": best_fid})
    wandb.finish()
    print(f"[{cfg.name}] DONE best_fid={best_fid:.4f}", flush=True)


if __name__ == "__main__":
    main()
