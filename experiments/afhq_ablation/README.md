# Diffusion backbone ablation harness

A single-GPU (16 GB) ablation harness for image-generation diffusion. Compares three backbones (`Unet`, `KarrasUnet`, `UViT`) under two diffusion formulations (`GaussianDiffusion` / DDPM and `ElucidatedDiffusion` / EDM), with per-backbone learning rates and a serial subprocess queue.

- First campaign (CIFAR-10 32x32): results and finding in `RESULTS.md`.
- Next campaign plan: `../../TODO.md`.

## Layout

| File | What it does |
|------|--------------|
| `configs.py` | `ExperimentConfig` + `RUNS` registry + `_BACKBONE_KWARGS` + `_BACKBONE_LR` (per-backbone learning rate) |
| `backbones.py` | `build_backbone` + `BackboneAdapter` unifying `Unet`/`KarrasUnet`/`UViT` + `count_params` |
| `diffusions.py` | `build_diffusion` (`ddpm` = `GaussianDiffusion`, `edm` = `ElucidatedDiffusion`) |
| `data.py` | Image-only CIFAR-10 loader plus an infinite cycle wrapper |
| `train.py` | Single-run training loop: bf16, EMA, periodic FID, sample grids written to disk, resume from `latest.pt` |
| `run_all.py` | Serial queue runner. One subprocess per run. Skips runs with a `done` marker unless `--force` is passed. |
| `sample_ckpt.py` | Offline sample-grid generator from a saved checkpoint |
| `paramtable.py` | Prints parameter counts so the three backbones stay within roughly 10% of each other |

## Ablation matrix

| run | backbone | diffusion |
|-----|----------|-----------|
| `unet-ddpm` | Unet | GaussianDiffusion, DDIM with 100 sampling steps |
| `karras-ddpm` | KarrasUnet | GaussianDiffusion |
| `uvit-ddpm` | UViT | GaussianDiffusion |
| `unet-edm` | Unet | ElucidatedDiffusion, Heun with 32 sampling steps |
| `karras-edm` | KarrasUnet | ElucidatedDiffusion |
| `uvit-edm` | UViT | ElucidatedDiffusion |

Self-conditioning is disabled for all runs so the three backbones can share a single `backbone(x, time)` calling convention. UViT does not support self-conditioning; turning it off in `Unet`/`KarrasUnet` keeps the comparison fair.

## Learning rate: KarrasUnet needs about 15x more than the other two

KarrasUnet is the magnitude-preserving architecture from EDM2. Every `Conv2d` and `Linear` inside it force-normalizes each row of its weight to L2 norm `sqrt(fan_in)` on every forward pass, so only the *direction* of each filter row receives real updates. Its effective learning rate is roughly `sqrt(fan_in)` times lower than a standard non-MP network at the same nominal `lr`.

At CIFAR-scale channel counts, the median `sqrt(fan_in)` is around 15. Training KarrasUnet at the same LR as `Unet` (`2e-4` here) means it learns about 25 times too slowly, and 80k steps is nowhere near enough to converge. In practice, `karras-edm` bottoms out around FID 89 at `2e-4`. Raising it to `3e-3` (the matched-constant approximation of the paper's shipped `InvSqrtDecayLRSched` with peak around `1e-2`) drops it to 18.23 and puts it on the same trajectory as `unet-edm`.

`configs._BACKBONE_LR` sets `lr = 3e-3` and `dropout = 0` for the `karras` backbone, and leaves `Unet` and `UViT` at `2e-4`. The two are not magnitude-preserving and do not need the adjustment.

## Usage

```bash
# print parameter counts and confirm the budgets match within ~10%
python experiments/afhq_ablation/paramtable.py

# smoke test: 300 steps per config with a mini FID, wandb offline
python experiments/afhq_ablation/run_all.py --smoke

# single run
python experiments/afhq_ablation/train.py --run unet-edm

# full priority-ordered queue
python experiments/afhq_ablation/run_all.py

# subset
python experiments/afhq_ablation/run_all.py --only karras-edm unet-edm uvit-edm

# regenerate a sample grid from a saved checkpoint
python experiments/afhq_ablation/sample_ckpt.py --run karras-edm --n 64
```

`run_all.py` launches each run as its own subprocess so CUDA state stays clean. When a run finishes, it writes `results/<name>/done`; later invocations skip it unless you pass `--force`.

wandb is optional. `train.py` reads `WANDB_API_KEY` from the environment. If it is not set but `WANDB_API` is (for example in a repo-root `.env`), the script maps it across. Pass `--wandb-mode disabled` to skip logging.

## Outputs

- `results/<run>/latest.pt`: model, EMA, optimizer, step, wandb id. Training resumes from this file if it exists.
- `results/<run>/best.pt`: checkpoint at the lowest FID seen so far.
- `results/<run>/done`: written when a run finishes cleanly.
- `results/<run>/samples/step_*.png`: 8x8 sample grid at every FID evaluation.
- `results/_fid_cache/dataset_stats.npz`: real-data statistics, computed once and shared across runs.

If training crashes, delete `done` if present and rerun the queue. It will reload `latest.pt` and pick up from the last checkpoint.
