"""Print a parameter-count table for the three backbones.

Run after editing dims in configs.py to confirm the backbones are matched to
within ~10%. Builds models on CPU (no CUDA needed).

    python experiments/afhq_ablation/paramtable.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs import RUNS  # noqa: E402
from backbones import build_backbone, count_params, human  # noqa: E402


def main() -> None:
    print(f"{'run':16s} {'backbone':8s} {'diffusion':10s} {'params':>10s}")
    counts: dict[str, int] = {}
    for name, cfg in RUNS.items():
        model = build_backbone(cfg)
        n = count_params(model)
        counts[name] = n
        print(f"{name:16s} {cfg.backbone:8s} {cfg.diffusion:10s} {human(n):>10s}")

    ddpm = {k: v for k, v in counts.items() if k.endswith("-ddpm")}
    if ddpm:
        lo, hi = min(ddpm.values()), max(ddpm.values())
        spread = (hi - lo) / lo * 100
        print(f"\nddpm backbone spread: {human(lo)} .. {human(hi)}  (+{spread:.1f}%)")
        print("target: <=10% spread; edit _BACKBONE_KWARGS in configs.py if larger")


if __name__ == "__main__":
    main()
