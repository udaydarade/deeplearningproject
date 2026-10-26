"""CLI runner for the multi-dataset crash-prediction benchmark.

Examples (from the project root):

  py -3.12 compute_features.py                 # build unified_panel.csv once
  py -3.12 run_benchmark.py --pilot            # smoke test, 2 tasks
  py -3.12 run_benchmark.py                    # full grid (both levels)
  py -3.12 run_benchmark.py --level ticker --models lstm,gru --tag mytag

Results stream incrementally into ``data/benchmark/all_results.csv`` and a
``benchmark_<tag>.log`` transcript.
"""
from __future__ import annotations

import argparse

from pipeline import benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["group", "ticker", "both"], default="both")
    parser.add_argument("--datasets", default="all",
                        help="comma-separated dataset names, default all")
    parser.add_argument("--modes", default="classical,tda,combined,landscape,full")
    parser.add_argument("--models", default=",".join(benchmark.MODEL_NAMES))
    parser.add_argument("--folds", default=None,
                        help="comma-separated fold names, default all three")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="main", help="log/transcript tag")
    parser.add_argument("--resume", action="store_true",
                        help="skip overall cells already present in all_results.csv")
    parser.add_argument("--pilot", action="store_true",
                        help="tiny smoke-test grid: group_a, combined, logreg+lstm, fold 2021+")
    args = parser.parse_args()

    levels = {"group": ["group"], "ticker": ["ticker"], "both": ["group", "ticker"]}[args.level]
    datasets_names = None if args.datasets == "all" else args.datasets.split(",")
    modes = args.modes.split(",")
    model_names = args.models.split(",")
    folds = benchmark.FOLDS if args.folds is None else [
        f for f in benchmark.FOLDS if f[0] in args.folds.split(",")
    ]
    if args.pilot:
        datasets_names, modes, model_names = ["group_a_us_indices"], ["combined"], ["logreg", "lstm"]
        folds = [benchmark.FOLDS[2]]
        epochs, workers = 3, 2

    benchmark.run_benchmark(
        levels=levels, datasets_names=datasets_names, modes=modes,
        model_names=model_names, folds=folds, epochs=args.epochs,
        workers=args.workers, tag=args.tag, resume=args.resume,
    )


if __name__ == "__main__":
    main()
