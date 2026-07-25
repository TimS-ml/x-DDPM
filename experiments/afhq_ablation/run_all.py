"""Serial queue runner: launches each ablation run as its own subprocess.

Independent processes give each run a clean CUDA context. A crashed run is
recorded and the queue continues. Completed runs (``results/<name>/done``) are
skipped unless ``--force``.

Usage:
    python experiments/afhq_ablation/run_all.py                 # all, priority order
    python experiments/afhq_ablation/run_all.py --only unet-ddpm
    python experiments/afhq_ablation/run_all.py --smoke         # 300-step smoke over all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from configs import RUN_ORDER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of run names (default: all in priority order)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--results-root", default=None,
                    help="default: results/ (real) or results_smoke/ (--smoke)")
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--force", action="store_true", help="ignore done markers")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    # keep smoke artifacts in a separate tree so they never shadow real runs
    if args.results_root is None:
        args.results_root = str(HERE / ("results_smoke" if args.smoke else "results"))

    runs = args.only or RUN_ORDER
    results_root = Path(args.results_root)
    summary: list[tuple[str, str]] = []

    for name in runs:
        done = results_root / name / "done"
        if done.exists() and not args.force and not args.smoke:
            print(f"[queue] skip {name} (done)", flush=True)
            summary.append((name, "skipped"))
            continue

        cmd = [args.python, str(HERE / "train.py"), "--run", name,
               "--results-root", args.results_root, "--data-root", args.data_root]
        if args.smoke:
            cmd.append("--smoke")

        print(f"[queue] === {name} ===  {' '.join(cmd)}", flush=True)
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
        dt = (time.time() - t0) / 60
        status = "ok" if ret.returncode == 0 else f"FAIL({ret.returncode})"
        summary.append((name, f"{status} {dt:.1f}min"))
        print(f"[queue] {name}: {status} in {dt:.1f}min", flush=True)

    print("\n[queue] summary:")
    for name, st in summary:
        print(f"  {name:16s} {st}")


if __name__ == "__main__":
    main()
