# Diffusion backbone ablations: results

Two campaigns on the same harness, both complete. Campaign 1 covers CIFAR-10 32x32 (4 of 6 cells); Campaign 2 covers AFHQv2 cat+dog 64x64 (all 6 cells).

Shared harness: `configs.py`, `backbones.py`, `diffusions.py`, `data.py`, `train.py`, `run_all.py`, `sample_ckpt.py`.

---

# Campaign 1 — CIFAR-10 32x32

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

---

# Campaign 2 — AFHQv2 cat+dog 64x64

Second campaign: same six-cell design (3 backbones x 2 diffusion formulations), at 2x the resolution and ~1.7x the parameter budget, on a much smaller dataset. Goal was to check whether the CIFAR backbone ordering survives a resolution and dataset change. It does — and the complete DDPM row, which Campaign 1 never got, overturns Campaign 1's reading of the diffusion axis.

## Setup

- Dataset: AFHQv2 cat + dog at 64x64, unconditional. Built from `huggan/AFHQv2` (parquet, streamed), Lanczos-resized, filtered to the cat and dog classes: 5558 cat + 5169 dog = **10727 images**. Cached at `data/afhqv2_catdog_64/{cat,dog}/`.
- Unconditional despite the class folders: only `KarrasUnet` accepts `num_classes` natively, so class-conditioning all three backbones would need library surgery. Labels are preserved on disk for a possible conditional follow-up; `data.AFHQCatDog` drops them.
- Hardware: single laptop RTX 4090, 16 GB VRAM.
- Backbones (~26-29M params, 8.8% spread):
  - `Unet` (`dim=108, dim_mults=(1,2,4)`) — 27.59M
  - `KarrasUnet` (`dim=72, dim_max=192, num_downsamples=4, num_blocks_per_stage=2, attn_res=(16,8), dropout=0.0`) — 26.26M
  - `UViT` (`dim=96, dim_mults=(1,2,4), vit_depth=8`) — 28.58M
- Diffusion, optimizer, EMA, LR overrides: unchanged from Campaign 1.
- Training: batch 64 (halved from CIFAR's 128; 64x64 is 4x the activation memory), 80k steps.
- Evaluation: `pytorch-fid`, 5000 generated samples against 5000 training images, every 10k steps.

## Final results (80k steps, all six cells complete)

| run | backbone | diffusion | best FID | wall |
|-----|----------|-----------|----------|------|
| `unet-edm` | Unet | EDM | **10.07** | 22.1 h |
| `unet-ddpm` | Unet | DDPM | 10.49 | 22.3 h |
| `karras-edm` | KarrasUnet | EDM | 10.55 | 14.7 h |
| `karras-ddpm` | KarrasUnet | DDPM | 13.70 | 13.9 h |
| `uvit-edm` | UViT | EDM | 14.39 | 13.1 h |
| `uvit-ddpm` | UViT | DDPM | 14.50 | 16.7 h |

Serial queue, single GPU, 102.8 h total. Sample grids at `results/<run>/samples/step_0080000.png`; best checkpoints and full trajectories under `results/<run>/`.

As a 3x2 grid:

| backbone | EDM | DDPM | DDPM − EDM |
|----------|-----|------|-----------|
| `Unet` | **10.07** | 10.49 | +0.43 |
| `KarrasUnet` | 10.55 | 13.70 | +3.14 |
| `UViT` | 14.39 | 14.50 | +0.11 |

## FID trajectories

| step | `unet-edm` | `unet-ddpm` | `karras-edm` | `karras-ddpm` | `uvit-edm` | `uvit-ddpm` |
|------|-----------|------------|-------------|--------------|-----------|------------|
| 10k | 29.84 | 25.45 | 127.83 | 125.62 | 52.22 | 48.49 |
| 20k | 18.30 | 17.42 | 38.91 | 33.89 | 26.79 | 27.34 |
| 30k | 15.90 | 15.55 | 20.64 | 21.95 | 22.06 | 25.57 |
| 40k | 14.22 | 14.13 | 15.35 | 17.85 | 19.50 | 24.16 |
| 50k | 12.67 | 12.96 | 13.12 | 16.18 | 17.92 | 20.88 |
| 60k | 11.38 | 12.04 | 11.83 | 15.28 | 16.07 | 18.34 |
| 70k | 10.39 | 10.85 | 10.87 | 14.32 | 15.51 | 15.63 |
| 80k | **10.07** | **10.49** | **10.55** | **13.70** | **14.39** | **14.50** |

## Finding 1: the backbone ordering reproduces, under both diffusions

`Unet` < `KarrasUnet` < `UViT` holds in every comparison available:

- AFHQ under EDM: 10.07 < 10.55 < 14.39
- AFHQ under DDPM: 10.49 < 13.70 < 14.50
- CIFAR-10 under EDM (Campaign 1): 15.98 < 18.23 < 20.13

Three datasets-by-formulation combinations, two resolutions, two parameter budgets, same order every time. This is the most transferable result from either campaign.

The `Unet` / `KarrasUnet` margin is thin under EDM (0.49) and wide under DDPM (3.20). The `UViT` deficit is large under both.

## Finding 2: the EDM advantage is specific to KarrasUnet, not general

| backbone | EDM | DDPM | gap |
|----------|-----|------|-----|
| `Unet` | 10.07 | 10.49 | 0.43 |
| `KarrasUnet` | 10.55 | 13.70 | **3.14** |
| `UViT` | 14.39 | 14.50 | 0.11 |
| `Unet` (CIFAR-10, Campaign 1) | 15.98 | 20.27 | 4.29 |

On the two non-magnitude-preserving backbones the two formulations are a tie: 0.43 and 0.11, both inside what a 5000-sample FID with one seed can resolve. Only `KarrasUnet` shows a real gap, and it is 7-30x the other two.

Campaign 1 could not see this. Its only usable DDPM cell was `unet-ddpm`, and its `karras-ddpm` was one of the runs destroyed by the `2e-4` LR. With `_BACKBONE_LR` now applying `3e-3` to `KarrasUnet` under both diffusion wrappers, all three pairs are measurable — and the conclusion one would have drawn from Campaign 1's single pair ("EDM beats DDPM by about 4 FID") does not survive.

A hypothesis, not tested here: `KarrasUnet` is magnitude-preserving, trains slowly under a flat LR, and is the most undertrained of the three at 80k. EDM's loss weighting distributes gradient signal across noise levels differently, which plausibly helps a net that has not converged more than one that nearly has. If that is the mechanism, the 3.14 gap is a symptom of the Campaign 1 LR problem rather than a property of the diffusion formulation. Running `KarrasUnet` under the proper `InvSqrtDecayLRSched` would test it.

**Unexplained:** CIFAR's `Unet` pair gave 4.29 while AFHQ's gives 0.43. Same backbone, same two wrappers, same harness. Dataset, resolution, and FID sample count all changed at once, so this campaign cannot attribute it.

## Finding 3: DDPM leads early on every backbone; EDM overtakes

At 10k steps DDPM is ahead in all three pairs — by 4.39 on `Unet`, 2.21 on `KarrasUnet`, 3.73 on `UViT`. EDM then overtakes, and the crossover comes earlier the weaker the backbone:

| backbone | DDPM lead at 10k | EDM overtakes | gap at 80k |
|----------|------------------|---------------|-----------|
| `Unet` | 4.39 | 40k–50k | 0.43 |
| `KarrasUnet` | 2.21 | 20k–30k | 3.14 |
| `UViT` | 3.73 | 10k–20k | 0.11 |

So EDM's benefit is not faster early optimization. At a budget of 10k steps DDPM would have won all three cells.

The endgames differ. `KarrasUnet` diverges monotonically after crossover. `Unet` converges back to near-parity. `UViT` diverges to 4.65 by 40k and then closes almost completely, with `uvit-ddpm` dropping 15.63 → 14.50 in the final 10k — the fastest late-stage improvement of any run. A longer budget would plausibly flip the `UViT` pair.

## Finding 4: nothing converged

All six runs were still descending at 80k, including the last interval:

| run | 70k → 80k |
|-----|----------|
| `unet-edm` | 10.39 → 10.07 |
| `unet-ddpm` | 10.85 → 10.49 |
| `karras-edm` | 10.87 → 10.55 |
| `karras-ddpm` | 14.32 → 13.70 |
| `uvit-edm` | 15.51 → 14.39 |
| `uvit-ddpm` | 15.63 → 14.50 |

This is a fixed-compute comparison at 80k steps, not a best-achievable comparison. The three closest numbers in the whole grid — `unet-edm` 10.07 / `unet-ddpm` 10.49 / `karras-edm` 10.55, spanning 0.49 FID — sit well inside the range these curves were still moving through in their final interval.

## Caveats

- One seed per cell. No error bars. Any difference under ~1 FID in this grid should be read as "indistinguishable": that covers `unet-edm` vs `unet-ddpm` vs `karras-edm` (0.49 total spread) and `uvit-edm` vs `uvit-ddpm` (0.11).
- **AFHQ and CIFAR FIDs are not comparable to each other.** Different dataset, different resolution, 5000 samples here versus 10000 on CIFAR. FID is biased upward at smaller sample counts. Compare only within a campaign.
- 5000-sample FID against a 10727-image dataset is a noisy estimator, and the reference set is the training set, so these are train-set FIDs with no held-out split.
- EDM and DDPM differ in loss weighting, time parametrization, sampler, and objective simultaneously. None of the gaps above isolates any one of those.
- Sampling cost differs between the two formulations: EDM uses 32 Heun steps (~63 NFE), DDPM uses 100 DDIM steps (100 NFE). Wall-clock totals do not cleanly show this — `unet-ddpm` (22.3 h) and `unet-edm` (22.1 h) came out nearly equal and `karras-ddpm` was *faster* than `karras-edm` — because GPU thermal throttling varied across the four days and dominates the comparison. Do not read run duration as sampler cost.
- The GPU sat at 88-89 C for essentially the whole campaign, holding 1.1-2.4 it/s depending on backbone. This affects wall-clock, not results.
- Unconditional only. Class labels exist on disk but no cell used them.
- Demo-scale. Published AFHQ numbers use far larger models and much longer schedules.

## What would be worth running next

- A second seed on the `Unet` and `UViT` pairs, to establish whether 0.43 and 0.11 are noise. Cheapest way to firm up Finding 2.
- `KarrasUnet` with `InvSqrtDecayLRSched` at peak `1e-2` under both wrappers, to test whether the 3.14 gap is an LR artifact.
- A longer budget on the `UViT` pair, where `uvit-ddpm` was still improving fastest at cutoff.

## Related files

- Campaign 1 checkpoints and FID trajectories: `results_cifar10/`.
- Campaign 2 checkpoints, sample grids, and FID trajectories: `results/<run>/`.
- Campaign 2 training log: `logs/queue_260723_1130.log`.
- wandb project: `x-ddpm-afhq-ablation`.
