"""Leakage-controlled TDA/CSD crash forecasting benchmark.

The target threshold is estimated separately for each index from 1997-2006:
the 2nd percentile of forward 15-session minimum simple returns. The fixed
threshold is then applied to all later dates. Indicators at date t use data
through t only. Models are evaluated on three chronological test blocks with
15-session purging at train/test boundaries.

Run from the project root with ``py -3.12 crash_forecasting.py``. Outputs are
written under ``data/forecasting/``. This uses precomputed TDA outputs and
does not download data or recompute persistent homology.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = ROOT / "data" / "forecasting"
TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
HORIZON = 15
CALIBRATION_END = pd.Timestamp("2006-12-31")
SEQUENCE_LENGTH = 20
SEED = 17
FOLDS = [
    ("test_2011_2015", "2011-01-01", "2015-12-31"),
    ("test_2016_2020", "2016-01-01", "2020-12-31"),
    ("test_2021_plus", "2021-01-01", "2100-01-01"),
]
DEVICE = torch.device("cpu")


class CrashLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def build_frame() -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, list[str], pd.Timestamp]:
    prices = pd.read_csv(GROUP / "prices.csv", parse_dates=["date"]).set_index("date")[TICKERS]
    returns = np.log(prices / prices.shift(1))

    # Four-dimensional, joint-index TDA features from the paper implementation.
    tda = pd.read_csv(GROUP / "tda_norms_paper_method.csv", parse_dates=["date"])
    tda = tda.pivot(index="date", columns="window", values=["landscape_l1", "landscape_l2"])
    tda.columns = [f"tda_{metric}_w{window}" for metric, window in tda.columns]

    # Per-index classical indicators. Rolling windows include the current day.
    blocks = []
    for ticker in TICKERS:
        r = returns[ticker]
        row = pd.DataFrame(index=returns.index)
        for window in (50, 100):
            roll = r.rolling(window, min_periods=window)
            # Correlation of adjacent pairs inside the same trailing block.
            # pandas' rolling corr is vectorized and avoids a Python callback
            # for every date (which is especially costly on four long series).
            row[f"csd_ac1_w{window}"] = r.rolling(window - 1, min_periods=window - 1).corr(r.shift(1))
            row[f"csd_variance_w{window}"] = roll.var()
            row[f"csd_skew_w{window}"] = roll.skew()
        row["realized_vol_w20"] = r.rolling(20, min_periods=20).std()
        row["ticker"] = ticker
        row.index.name = "date"
        blocks.append(row)
    classical = pd.concat(blocks).reset_index().set_index(["date", "ticker"])

    # Fixed label threshold learned only from the calibration period.
    forward_min = pd.DataFrame(index=prices.index)
    thresholds: dict[str, float] = {}
    calibration_end_position = prices.index.searchsorted(CALIBRATION_END, side="right") - 1
    calibration_last_origin = prices.index[calibration_end_position - HORIZON]
    for ticker in TICKERS:
        # min return from the close at t over closes t+1,...,t+HORIZON
        paths = pd.concat([prices[ticker].shift(-k) / prices[ticker] - 1 for k in range(1, HORIZON + 1)], axis=1)
        forward_min[ticker] = paths.min(axis=1)
        # Only retain origins whose complete 15-session outcome is observable
        # by the calibration price cutoff; this avoids a horizon-edge leak.
        fit = forward_min.loc[forward_min.index <= calibration_last_origin, ticker].dropna()
        thresholds[ticker] = float(fit.quantile(0.02))

    tda_wide = tda.reindex(prices.index)
    rows = []
    for ticker in TICKERS:
        c = classical.xs(ticker, level="ticker")
        merged = pd.concat([tda_wide, c, forward_min[ticker].rename("forward_min_return")], axis=1)
        merged["ticker"] = ticker
        merged["label"] = (merged["forward_min_return"] <= thresholds[ticker]).astype("float")
        merged.loc[merged["forward_min_return"].isna(), "label"] = np.nan
        rows.append(merged)
    frame = pd.concat(rows).rename_axis("date").reset_index().sort_values(["ticker", "date"])
    features = [c for c in frame.columns if c.startswith(("tda_", "csd_", "realized_"))]
    frame = frame.dropna(subset=features + ["label"]).copy()
    frame["label"] = frame["label"].astype(int)
    return frame, thresholds, forward_min, features, calibration_last_origin


def sequences(frame: pd.DataFrame, feature_names: list[str], mode: str):
    chosen = [c for c in feature_names if (mode == "combined" or c.startswith("tda_"))]
    if mode == "classical":
        chosen = [c for c in feature_names if not c.startswith("tda_")]
    xs, ys, dates, tickers = [], [], [], []
    for ticker, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        arr, y = group[chosen].to_numpy(np.float32), group["label"].to_numpy(np.float32)
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            xs.append(arr[end - SEQUENCE_LENGTH + 1:end + 1])
            ys.append(y[end]); dates.append(group.loc[end, "date"]); tickers.append(ticker)
    return np.stack(xs), np.asarray(ys), pd.to_datetime(dates), np.asarray(tickers), chosen


def best_threshold(y, p):
    # Operational threshold fixed on validation to target 5% alert days.
    # This avoids tuning the alarm threshold on the test block and makes the
    # false-alarm burden interpretable. Ties can make the realized rate larger.
    del y
    return float(np.quantile(p, 0.95, method="higher"))


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)), "threshold": float(threshold),
        "precision": float(np.sum((pred == 1) & (y == 1)) / max(pred.sum(), 1)),
        "recall": float(np.sum((pred == 1) & (y == 1)) / max(y.sum(), 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alarms": int(pred.sum()),
    }


def fit_predict(x, y, train_idx, val_idx, test_idx, epochs=30):
    scaler = StandardScaler()
    scaler.fit(x[train_idx].reshape(-1, x.shape[-1]))
    z = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    pos = max(float(y[train_idx].sum()), 1.0)
    neg = max(float(len(train_idx) - y[train_idx].sum()), 1.0)
    model = CrashLSTM(x.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(neg / pos, 20.0)]))
    ds = TensorDataset(torch.from_numpy(z[train_idx]), torch.from_numpy(y[train_idx]))
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    best_ap, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]))).numpy()
        ap = average_precision_score(y[val_idx], val_p) if y[val_idx].sum() else 0.0
        if ap > best_ap + 1e-5:
            best_ap, best_state, stale = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 4: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]))).numpy()
        test_p = torch.sigmoid(model(torch.from_numpy(z[test_idx]))).numpy()
    return model, scaler, val_p, test_p, best_threshold(y[val_idx], val_p)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    frame, thresholds, forward_min, features, calibration_last_origin = build_frame()
    pd.DataFrame({"ticker": list(thresholds), "threshold_2pct": list(thresholds.values()),
                  "calibration_start": "1997-01-02", "calibration_end": CALIBRATION_END.date(),
                  "last_label_origin": calibration_last_origin.date(),
                  "horizon_sessions": HORIZON}).to_csv(OUT / "label_thresholds.csv", index=False)
    frame.to_csv(OUT / "modeling_panel.csv", index=False, date_format="%Y-%m-%d")
    rows, final_models = [], {}
    for fold, test_start, test_end in FOLDS:
        start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
        # 15 trading-session purge approximated by calendar days with extra buffer.
        train_end = start - pd.Timedelta(days=25)
        val_start = train_end - pd.DateOffset(years=2)
        fold_frame = frame
        for mode in ("classical", "tda", "combined"):
            x, y, dates, tickers, chosen = sequences(fold_frame, features, mode)
            train_idx = np.flatnonzero(dates < val_start - pd.Timedelta(days=25))
            val_idx = np.flatnonzero((dates >= val_start) & (dates <= train_end))
            test_idx = np.flatnonzero((dates >= start) & (dates <= end))
            if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
                continue
            model, scaler, val_p, test_p, threshold = fit_predict(x, y, train_idx, val_idx, test_idx, epochs=15)
            result = metrics(y[test_idx], test_p, threshold)
            result.update(fold=fold, model=f"LSTM_{mode}", train_start=str(dates[train_idx].min().date()),
                          train_end=str(dates[train_idx].max().date()), val_start=str(dates[val_idx].min().date()),
                          val_end=str(dates[val_idx].max().date()), test_start=str(dates[test_idx].min().date()),
                          test_end=str(dates[test_idx].max().date()), feature_count=len(chosen))
            rows.append(result)
            print(f"{fold} {mode}: n_train={len(train_idx)} n_val={len(val_idx)} n_test={len(test_idx)} AP={result['average_precision']:.4f}", flush=True)
            if fold == "test_2021_plus":
                final_models[mode] = (model, scaler, chosen, threshold)
            for ticker in TICKERS:
                sub = test_idx[tickers[test_idx] == ticker]
                if len(sub):
                    r = metrics(y[sub], test_p[np.isin(test_idx, sub)], threshold)
                    r.update(fold=fold, model=f"LSTM_{mode}", ticker=ticker)
                    rows.append(r)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "walk_forward_metrics.csv", index=False)
    if final_models:
        import joblib
        for mode, (model, scaler, chosen, threshold) in final_models.items():
            torch.save({"state_dict": model.state_dict(), "feature_names": chosen,
                        "sequence_length": SEQUENCE_LENGTH, "threshold": threshold,
                        "hidden_size": 24, "label_thresholds": thresholds}, OUT / f"{mode}_lstm.pt")
            joblib.dump(scaler, OUT / f"{mode}_scaler.joblib")
    report = [
        "Leakage-controlled TDA versus CSD crash forecast",
        f"Target: forward {HORIZON}-session minimum simple return <= ticker-specific 2nd percentile estimated on origins through {calibration_last_origin.date()} (all outcomes observed by {CALIBRATION_END.date()}).",
        "All feature windows use observations through the forecast date; train, validation, and test dates are chronological.",
        "A 25-calendar-day separation was left before validation and test blocks to purge overlapping 15-session targets.",
        f"Calibration thresholds: {json.dumps(thresholds, sort_keys=True)}",
        f"Rows after feature availability checks: {len(frame)}; positive share by ticker: {frame.groupby('ticker').label.mean().round(4).to_dict()}",
        "Metrics are daily average precision (primary, prevalence-sensitive), ROC AUC, Brier score, and precision/recall at a score threshold set to the validation 95th percentile (about 5% alert days).",
        "The test blocks and per-index results are in walk_forward_metrics.csv. Results are conditional on the chosen calibration threshold and historical sample.",
        "The joint four-index TDA features are shared across target indices; classical features are calculated separately for each index.",
    ]
    (OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nAggregate test-block results:")
    agg = results[results.ticker.isna()].groupby("model")[["average_precision", "roc_auc", "brier", "precision", "recall", "f1"]].mean()
    print(agg.round(4).to_string())
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
