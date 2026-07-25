# CIFAR-10 diffusion backbone ablation: results

First experimental campaign on this harness. Frozen writeup; live plan is in `../../TODO.md`.

## Setup

- Dataset: CIFAR-10 32x32 unconditional (`data.ImageOnlyCIFAR10`, 50k train images).
- Hardware: single GPU with 16 GB VRAM.
- Backbones (~16-18M params, budget-matched within 10%):
  - `Unet` (`dim=82, dim_mults=(1,2,4)`)
  - `KarrasUnet` (`dim=58, dim_max=192, num_downsamples=3, num_blocks_per_stage=2`)
  - `UViT` (`dim=80, dim_mults=(1,2,4), vit_depth=6`)
- Diffusion: `GaussianDiffusion` (DDPM, DDIM 100 steps) vs `ElucidatedDiffusion` (EDM, Heun 32 steps).
- Training: batch 128, Adam, grad clip 1.0, bf16 autocast, EMA decay 0.999 / update every 10, 80k steps.
- Learning rate: `2e-4` for `Unet` and `UViT`, `3e-3` for `KarrasUnet` (see below).
- Evaluation: `pytorch-fid` against 10k CIFAR-10 training images with 10k generated samples, every 10k steps.

## Final results (80k steps)

| run | backbone | diffusion | best FID |
|-----|----------|-----------|----------|
| `unet-edm` | Unet | EDM | **15.98** |
| `karras-edm` | KarrasUnet | EDM | 18.23 |
| `uvit-edm` | UViT | EDM | 20.13 |
| `unet-ddpm` | Unet | DDPM | 20.27 |

Four of the six planned cells were run. `karras-ddpm` and `uvit-ddpm` were dropped after the LR finding below, which shifted focus to the EDM axis.

Sample grids at step 80k:
- `results_cifar10/unet-edm/samples/step_0080000.png`
- `results_cifar10/karras-edm/samples/step_0080000.png`
- `results_cifar10/uvit-edm/samples/step_0080000.png`
- `results_cifar10/unet-ddpm/samples/best_grid.png` (generated after training via `sample_ckpt.py`; the training-time sample hook was added later)

Best checkpoints and full FID trajectories are in `results_cifar10/<run>/`.

## Key finding: KarrasUnet needs ~15x higher LR

`KarrasUnet` (EDM2's magnitude-preserving UNet) force-normalizes each row of every `Conv2d` and `Linear` weight to L2 norm `sqrt(fan_in)` on every forward pass. Only the *direction* of each filter row actually receives updates, so its effective learning rate is roughly `sqrt(fan_in)` times lower than a standard non-MP network at the same nominal `lr`.

At CIFAR-scale channel counts the median `sqrt(fan_in)` is around 15. Training `KarrasUnet` at `Unet`'s `lr = 2e-4` made it learn about 25 times too slowly:

| run | LR | best FID |
|-----|----|----------|
| `karras-ddpm` at 80k | 2e-4 | 80.6 |
| `karras-edm` at 70k (killed early) | 2e-4 | 89.3 |
| `karras-edm` at 80k | 3e-3 | **18.23** |

`configs._BACKBONE_LR` encodes the per-backbone override. `Unet` and `UViT` are not magnitude-preserving and keep `2e-4`.

The tell before I understood the mechanism: training loss looked normal (~0.07, comparable to `Unet`) but sampling produced structureless noise. Low loss with bad samples means the backbone is not actually learning the noise-conditioning dependency well enough; for MP nets the usual cause is undertraining from a mismatched LR.

The full recipe from the `KarrasUnet` author is `InvSqrtDecayLRSched` with peak LR around `1e-2` plus a short warmup. Flat `3e-3` is the matched-constant approximation and was enough for this budget. Bumping to the full schedule would likely close the remaining gap to `unet-edm`.

## Caveats

- One seed per configuration. No error bars.
- `unet-edm` (15.98) vs `unet-ddpm` (20.27) is above single-seed noise, but the two setups differ in loss weighting, time parametrization, sampler, and objective. The gap does not cleanly isolate the diffusion formula.
- The DDPM row is incomplete for `KarrasUnet` and `UViT`. With the LR fix they would probably be competitive; not verified.
- Demo-scale numbers. Published SOTA on unconditional CIFAR-10 is around FID 2 with ~55M parameters and much longer training. Do not treat 15.98 as a strong absolute number.

## Related files

- Harness code: `configs.py`, `backbones.py`, `diffusions.py`, `data.py`, `train.py`, `run_all.py`, `sample_ckpt.py`.
- Archived checkpoints and FID trajectories: `results_cifar10/`.
