"""Leakage-controlled walk-forward benchmark across levels, modes and models.

Two evaluation levels share one code path:

* ``group``   - one training pool per market dataset. Entities are the group's
  tickers; each sequence carries the ticker's own classical features and its
  own label, plus the joint multivariate TDA features of the group (available
  only on dates where every group member quotes). Answers: "does TDA help
  predict crashes for this market group?"
* ``ticker``  - one pooled training set across every ticker of every dataset,
  evaluated with a per-ticker breakdown. Answers: "do per-index topological
  signals generalise across markets?"

Protocol (identical to crash_forecasting.py):
chronological folds, a 25-calendar-day purge before each validation and test
block so no 15-session forward label overlaps train and test, the warning
threshold fixed at the validation 95th percentile of scores (~5% alarm days),
and model selection on validation average precision.

Feature modes map differently by level: ``classical`` (CSD only), ``tda``
(joint TDA at group level, per-ticker Takens TDA at ticker level), and
``combined`` (CSD + whichever TDA the level uses).
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import models as nnz

ROOT = Path(__file__).resolve().parents[1]
PANEL_FILE = ROOT / "data" / "benchmark" / "unified_panel.csv"
OUT = ROOT / "data" / "benchmark"

SEQUENCE_LENGTH = 20
HORIZON = 15
PURGE_DAYS = 25
VAL_YEARS = 2
FOLDS = (
    ("fold_2011_2015", "2011-01-01", "2015-12-31"),
    ("fold_2016_2020", "2016-01-01", "2020-12-31"),
    ("fold_2021_plus", "2021-01-01", "2100-01-01"),
)
PROPORTIONAL_FOLD = ("fold_proportional", None, None)
# Datasets whose evaluable history starts after this date cannot be split into
# the calendar folds (train would fall outside the series) and instead get one
# chronological 70/15/15 proportional fold.
CALENDAR_FOLD_CUTOFF = pd.Timestamp("2010-01-01")
FEATURE_WARMUP_DAYS = 290  # ~200 sessions of classical-feature + label warmup

WARN_STAT = ("ac1", "variance", "skew", "kurtosis")
CLASSICAL_COLS = [
    *(f"csd_{stat}_w{w}" for w in (50, 100) for stat in WARN_STAT),
    "realized_vol_w20",
    "ewma_vol_w20",
    "csd_garch_vol",
]
CROSS_COLS = ["csd_meanpaircorr_w50", "csd_maxeig_w50",
              "csd_meanpaircorr_w100", "csd_maxeig_w100"]
PRICE_COLS = [
    "price_return_1d", "price_momentum_5d", "price_momentum_20d", "price_momentum_60d",
    "price_volatility_20d", "price_volatility_60d", "price_downside_vol_20d",
    "price_drawdown_60d", "price_drawdown_252d", "price_volratio_20_60", "price_skew_20d",
]
PEER_COLS = ["peer_return_mean", "peer_return_median", "peer_breadth",
             "peer_dispersion", "peer_vol_mean", "peer_corr_w50"]
TDA_SCALARS = ["l1", "l2", "entropy", "betti", "totalpers"]
GRID_COLS = [f"grid_L{level}_g{j}"
             for level in range(1, 4) for j in range(12)]


def tda_cols(prefix: str) -> list[str]:
    """Scalar summaries + discretised landscape vector for a TDA variant."""
    return [f"{prefix}_{name}" for name in TDA_SCALARS + GRID_COLS]


def tda_scalar_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in TDA_SCALARS]


TDA_INDEX_COLS = tda_cols("tda_idx")
TDA_JOINT_COLS = tda_cols("tda_joint")

MODE_COLUMNS = {
    "group": {
        "classical": CLASSICAL_COLS + CROSS_COLS,
        "tda": tda_scalar_cols("tda_joint"),
        "landscape": TDA_JOINT_COLS,
        "combined": CLASSICAL_COLS + tda_scalar_cols("tda_joint"),
        "full": CLASSICAL_COLS + CROSS_COLS + TDA_JOINT_COLS + tda_scalar_cols("tda_idx"),
        "price": PRICE_COLS,
        "peer": PEER_COLS,
        "everything": (CLASSICAL_COLS + CROSS_COLS + TDA_JOINT_COLS
                       + tda_scalar_cols("tda_idx") + PRICE_COLS + PEER_COLS),
    },
    "ticker": {
        "classical": CLASSICAL_COLS,
        "tda": tda_scalar_cols("tda_idx"),
        "landscape": TDA_INDEX_COLS,
        "combined": CLASSICAL_COLS + tda_scalar_cols("tda_idx"),
        "full": CLASSICAL_COLS + TDA_INDEX_COLS,
        "price": PRICE_COLS,
        "peer": PEER_COLS,
        "everything": CLASSICAL_COLS + TDA_INDEX_COLS + PRICE_COLS + PEER_COLS,
    },
}

MODEL_NAMES = ("logreg", "lstm", "bilstm", "attn_bilstm", "gru", "cnn", "tcn", "transformer")

_PANEL: pd.DataFrame | None = None


# ----------------------------------------------------------------- utilities
def best_threshold(y, p):
    """Operational alarm cut set to the validation 95th percentile of scores."""
    return float(np.quantile(p, 0.95, method="higher"))


def _score_block(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 f1_score, roc_auc_score)

    predicted = p >= threshold
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(np.sum(predicted & (y == 1)) / max(int(predicted.sum()), 1)),
        "recall": float(np.sum(predicted & (y == 1)) / max(int(y.sum()), 1)),
        "f1": float(f1_score(y, predicted.astype(int), zero_division=0)),
        "alarms": int(predicted.sum()),
    }


def load_panel() -> pd.DataFrame:
    frame = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    keep = ["dataset", "ticker", "date", "label",
            *CLASSICAL_COLS, *CROSS_COLS, *PRICE_COLS, *PEER_COLS,
            *TDA_INDEX_COLS, *TDA_JOINT_COLS]
    missing = set(keep) - set(frame.columns)
    if missing:
        raise ValueError(f"unified_panel.csv lacks columns: {sorted(missing)}")
    return frame[keep]


def sequences(frame: pd.DataFrame, columns: list[str]):
    """Sliding (seq_len) windows per ticker over a sorted frame."""
    xs, ys, dates, tickers = [], [], [], []
    for ticker, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("date")
        arr = group[columns].to_numpy(np.float32)
        y = group["label"].to_numpy(np.float32)
        dts = group["date"].to_numpy()
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            xs.append(arr[end - SEQUENCE_LENGTH + 1 : end + 1])
            ys.append(y[end])
            dates.append(dts[end])
            tickers.append(ticker)
    return np.stack(xs), np.asarray(ys), np.asarray(dates), np.asarray(tickers)


def task_seed(spec: dict) -> int:
    payload = json.dumps({k: spec[k] for k in ("level", "dataset", "mode", "model", "fold")}, sort_keys=True)
    return int(hashlib.sha1(payload.encode()).hexdigest()[:8], 16) % (2**31)


# ---------------------------------------------------------------- worker side
def init_worker(panel_path: str) -> None:
    global _PANEL
    torch.set_num_threads(1)
    _PANEL = load_panel() if panel_path is None else pd.read_csv(panel_path, parse_dates=["date"])


def run_task(spec: dict) -> list[dict]:
    panel = _PANEL if _PANEL is not None else pd.read_csv(spec["panel_path"], parse_dates=["date"])
    columns = MODE_COLUMNS[spec["level"]][spec["mode"]]
    subset = panel if spec["level"] == "ticker" else panel[panel["dataset"] == spec["dataset"]]
    subset = subset.dropna(subset=[*columns, "label"])
    x, y, dates, tickers = sequences(subset, columns)

    test_start, test_end = pd.Timestamp(spec["test_start"]), pd.Timestamp(spec["test_end"])
    if spec["fold"] == "fold_proportional":
        # Chronological 70/15/15 split for datasets too short for calendar folds.
        first, last = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
        span = (last - first).days or 1
        test_start = first + pd.Timedelta(days=int(span * 0.70))
        test_end = last
    train_end = test_start - pd.Timedelta(days=PURGE_DAYS)
    val_start = train_end - pd.DateOffset(years=VAL_YEARS)

    train_idx = np.flatnonzero(dates < val_start - pd.Timedelta(days=PURGE_DAYS))
    val_idx = np.flatnonzero((dates >= val_start) & (dates <= train_end))
    test_idx = np.flatnonzero((dates >= test_start) & (dates <= test_end))
    if min(len(train_idx), len(val_idx), len(test_idx)) < 200:
        return []
    if y[test_idx].sum() == 0:
        # A label calibration that leaves zero crashes in the test window
        # (e.g. crypto's first-40% threshold) makes scoring undefined.
        return []

    if spec["model"] == "logreg":
        _, _, val_p, test_p = nnz.train_logit(x, y, train_idx, val_idx, test_idx, seed=task_seed(spec))
    else:
        _, _, val_p, test_p = nnz.train_sequence_model(
            x, y, train_idx, val_idx, test_idx,
            model_name=spec["model"], epochs=spec["epochs"], seed=task_seed(spec),
        )
    threshold = best_threshold(y[val_idx], val_p)

    rows = []
    overall = _score_block(y[test_idx], test_p, threshold)
    overall.update(
        level=spec["level"], dataset=spec["dataset"], ticker=None, fold=spec["fold"],
        model=spec["model"], mode=spec["mode"], entity_count=len(np.unique(tickers[test_idx])),
        train_start=str(pd.Timestamp(dates[train_idx].min()).date()),
        train_end=str(train_end.date()), val_start=str(val_start.date()),
        test_start=str(dates[test_idx].min()), test_end=str(dates[test_idx].max()),
        feature_count=len(columns), sequence_length=SEQUENCE_LENGTH,
    )
    rows.append(overall)

    # Per-ticker breakdown inherits the pooled model's shared alarm threshold.
    for ticker in np.unique(tickers[test_idx]):
        sub = test_idx[tickers[test_idx] == ticker]
        if len(sub) < 30 or y[sub].sum() == 0:
            continue
        row = _score_block(y[sub], test_p[np.searchsorted(test_idx, sub)], threshold)
        row.update(
            level=spec["level"], dataset=spec["dataset"], ticker=ticker, fold=spec["fold"],
            model=spec["model"], mode=spec["mode"], entity_count=1,
            train_start=overall["train_start"], train_end=overall["train_end"],
            val_start=overall["val_start"], test_start=overall["test_start"],
            test_end=overall["test_end"], feature_count=len(columns),
            sequence_length=SEQUENCE_LENGTH,
        )
        rows.append(row)
    return rows


# ----------------------------------------------------------------- driver
def fold_plan(panel: pd.DataFrame, level: str, datasets_names, folds) -> dict:
    """Per-dataset fold list; short-history datasets get one proportional fold."""
    if level == "ticker":
        return {"ALL": list(folds)}
    plan = {}
    for dataset in datasets_names:
        first = pd.Timestamp(panel.loc[panel["dataset"] == dataset, "date"].min())
        first_eval = first + pd.Timedelta(days=FEATURE_WARMUP_DAYS)
        plan[dataset] = ([PROPORTIONAL_FOLD] if first_eval > CALENDAR_FOLD_CUTOFF
                         else list(folds))
    return plan


def build_tasks(levels, datasets_names, modes, model_names, plan, epochs, panel_path):
    specs = []
    for level in levels:
        # The ticker level pools every dataset into ONE training set; group
        # level runs one task set per real dataset name.
        level_datasets = ["ALL"] if level == "ticker" else [
            name for name in datasets_names if name != "ALL"]
        for dataset in level_datasets:
            for mode in modes:
                for model in model_names:
                    for fold_name, start, end in plan[dataset]:
                        specs.append({
                            "task": f"{level}|{dataset}|{mode}|{model}|{fold_name}",
                            "panel_path": panel_path, "level": level, "dataset": dataset,
                            "mode": mode, "model": model, "fold": fold_name,
                            "test_start": start or "2010-01-01", "test_end": end or "2100-01-01",
                            "epochs": epochs,
                        })
    return specs


def run_benchmark(levels=("group", "ticker"), datasets_names=None, modes=("classical", "tda", "combined"),
                  model_names=MODEL_NAMES, folds=FOLDS, epochs=12, workers=8, tag="main",
                  resume=False) -> Path:
    """Execute the grid and stream results to ``data/benchmark/all_results.csv``.

    With ``resume=True``, overall cells already present in the CSV are skipped,
    allowing interrupted runs to continue.
    """
    panel_path = PANEL_FILE
    if not panel_path.exists():
        raise FileNotFoundError(f"Run compute_features.py first ({panel_path} missing)")
    panel = load_panel()
    available = sorted(panel.dataset.unique())
    datasets_names = datasets_names or available
    unknown = set(datasets_names) - set(available)
    if unknown:
        raise ValueError(f"Unknown datasets {sorted(unknown)}; available {available}")

    specs = []
    for level in levels:
        plan = fold_plan(panel, level, datasets_names, folds)
        specs.extend(build_tasks([level], datasets_names, modes, model_names, plan,
                                 epochs, panel_path))
    out_csv = OUT / "all_results.csv"
    if resume and out_csv.exists():
        prior = pd.read_csv(out_csv)
        prior = prior[prior["ticker"].isna()]
        seen = set(zip(prior["level"], prior["dataset"], prior["mode"],
                       prior["model"], prior["fold"]))
        specs = [s for s in specs
                 if (s["level"], s["dataset"], s["mode"], s["model"], s["fold"]) not in seen]
    log_path = OUT / f"benchmark_{tag}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()

    emit(f"[run] tasks={len(specs)} workers={workers} levels={levels} datasets={datasets_names}")
    header_written = out_csv.exists()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(panel_path,)) as pool:
        futures = {pool.submit(run_task, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                rows = future.result()
            except Exception as exc:  # keep the grid running, log the failure
                emit(f"[error] {spec['task']}: {exc!r}")
                continue
            done += 1
            if rows:
                frame = pd.DataFrame(rows)
                frame.to_csv(out_csv, mode="a", header=not header_written, index=False)
                header_written = True
            if rows:
                top = rows[0]
                emit(f"[done {done}/{len(specs)}] {spec['task']} "
                     f"AP={top['average_precision']:.4f} AUC={top['roc_auc']:.4f} "
                     f"n={top['n']} prev={top['prevalence']:.3f}")
            else:
                emit(f"[skip {done}/{len(specs)}] {spec['task']} (insufficient data)")
    log.close()
    print(f"[complete] results -> {out_csv}\n", flush=True)
    return out_csv
