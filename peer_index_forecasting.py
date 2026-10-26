"""Deep crash forecasting from the *other* indices.

For each target index, this experiment removes the target's own returns from
the input. A sequence model receives only the other indices' recent returns,
volatility, cross-sectional mean, dispersion, and correlation. The target
index is used only to construct its forward 15-session crash label.

The experiment is deliberately separate from ``crash_forecasting.py`` and
``pipeline/benchmark.py``. It reuses the existing GRU and Transformer model
implementations, preserving the original CSD/TDA/model-zoo results.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.models import build_model

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = ROOT / "data" / "forecasting"
TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
HORIZON = 15
CALIBRATION_END = pd.Timestamp("2006-12-31")
SEQUENCE_LENGTH = 30
PURGE_DAYS = 25
SEED = 17
FOLDS = [
    ("test_2011_2015", "2011-01-01", "2015-12-31"),
    ("test_2016_2020", "2016-01-01", "2020-12-31"),
    ("test_2021_plus", "2021-01-01", "2100-01-01"),
]
MODELS = ("gru", "transformer")


def build_peer_frame() -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    prices = pd.read_csv(GROUP / "prices.csv", parse_dates=["date"]).set_index("date")[TICKERS]
    prices = prices.dropna()
    returns = np.log(prices / prices.shift(1))
    calibration_position = prices.index.searchsorted(CALIBRATION_END, side="right") - 1
    calibration_last_origin = prices.index[calibration_position - HORIZON]
    thresholds: dict[str, float] = {}
    forward_min = {}
    for ticker in TICKERS:
        paths = pd.concat(
            [prices[ticker].shift(-step) / prices[ticker] - 1 for step in range(1, HORIZON + 1)],
            axis=1,
        )
        forward_min[ticker] = paths.min(axis=1)
        fit = forward_min[ticker].loc[forward_min[ticker].index <= calibration_last_origin].dropna()
        thresholds[ticker] = float(fit.quantile(0.02))

    rows = []
    for target in TICKERS:
        peers = [ticker for ticker in TICKERS if ticker != target]
        peer_returns = returns[peers]
        features = pd.DataFrame(index=returns.index)
        for peer_number, peer in enumerate(peers, start=1):
            features[f"peer_return_{peer_number}"] = peer_returns[peer]
            features[f"peer_vol_{peer_number}"] = peer_returns[peer].rolling(20, min_periods=20).std()
        features["peer_mean_return"] = peer_returns.mean(axis=1)
        features["peer_dispersion"] = peer_returns.std(axis=1)
        features["peer_mean_vol"] = peer_returns.rolling(20, min_periods=20).std().mean(axis=1)
        pairwise = []
        for left_index, left in enumerate(peers):
            for right in peers[left_index + 1:]:
                pairwise.append(peer_returns[left].rolling(50, min_periods=50).corr(peer_returns[right]))
        features["peer_corr_mean"] = pd.concat(pairwise, axis=1).mean(axis=1)
        features["target"] = target
        features["label"] = (forward_min[target] <= thresholds[target]).astype(float)
        features.loc[forward_min[target].isna(), "label"] = np.nan
        rows.append(features.reset_index())
    frame = pd.concat(rows, ignore_index=True)
    feature_names = [column for column in frame.columns if column.startswith("peer_")]
    frame = frame.dropna(subset=feature_names + ["label"]).copy()
    frame["label"] = frame["label"].astype(int)
    frame["target"] = frame["target"].astype(str)
    return frame.sort_values(["target", "date"]).reset_index(drop=True), thresholds, feature_names


def sequences(frame: pd.DataFrame, feature_names: list[str]):
    xs, ys, dates, targets = [], [], [], []
    for target, group in frame.groupby("target", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_names].to_numpy(np.float32)
        labels = group["label"].to_numpy(np.float32)
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            xs.append(values[end - SEQUENCE_LENGTH + 1:end + 1])
            ys.append(labels[end])
            dates.append(group.loc[end, "date"])
            targets.append(target)
    return np.stack(xs), np.asarray(ys), pd.to_datetime(dates), np.asarray(targets)


def score(y: np.ndarray, probabilities: np.ndarray, cutoff: float) -> dict:
    predicted = probabilities >= cutoff
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, probabilities)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, probabilities)),
        "threshold": float(cutoff),
        "precision": float(np.sum(predicted & (y == 1)) / max(int(predicted.sum()), 1)),
        "recall": float(np.sum(predicted & (y == 1)) / max(int(y.sum()), 1)),
        "f1": float(f1_score(y, predicted.astype(int), zero_division=0)),
        "alarms": int(predicted.sum()),
    }


def fit_model(model_name: str, x, y, train_idx, val_idx, test_idx):
    scaler = StandardScaler()
    scaler.fit(x[train_idx].reshape(-1, x.shape[-1]))
    scaled = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    positives = max(float(y[train_idx].sum()), 1.0)
    negatives = max(float(len(train_idx) - y[train_idx].sum()), 1.0)
    model = build_model(model_name, x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(negatives / positives, 20.0)))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=512, shuffle=True,
    )
    best_ap, best_state, stale = -1.0, None, 0
    for _ in range(15):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(torch.from_numpy(scaled[val_idx]))).numpy()
        ap = average_precision_score(y[val_idx], val_p) if y[val_idx].sum() else 0.0
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(model(torch.from_numpy(scaled[val_idx]))).numpy()
        test_p = torch.sigmoid(model(torch.from_numpy(scaled[test_idx]))).numpy()
    cutoff = float(np.quantile(val_p, 0.95, method="higher"))
    return model, scaler, test_p, cutoff


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    frame, thresholds, feature_names = build_peer_frame()
    x, y, dates, targets = sequences(frame, feature_names)
    rows = []
    final_models = {}
    for fold, test_start, test_end in FOLDS:
        start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
        train_end = start - pd.Timedelta(days=PURGE_DAYS)
        val_start = train_end - pd.DateOffset(years=2)
        train_idx = np.flatnonzero(dates < val_start - pd.Timedelta(days=PURGE_DAYS))
        val_idx = np.flatnonzero((dates >= val_start) & (dates <= train_end))
        test_idx = np.flatnonzero((dates >= start) & (dates <= end))
        for model_name in MODELS:
            if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
                continue
            model, scaler, test_p, cutoff = fit_model(model_name, x, y, train_idx, val_idx, test_idx)
            overall = score(y[test_idx], test_p, cutoff)
            overall.update(
                fold=fold, model=model_name, target="ALL",
                train_start=str(dates[train_idx].min().date()),
                train_end=str(dates[train_idx].max().date()),
                test_start=str(dates[test_idx].min().date()),
                test_end=str(dates[test_idx].max().date()),
                feature_count=len(feature_names), sequence_length=SEQUENCE_LENGTH,
            )
            rows.append(overall)
            for target in TICKERS:
                sub = test_idx[targets[test_idx] == target]
                if len(sub) == 0:
                    continue
                result = score(y[sub], test_p[np.isin(test_idx, sub)], cutoff)
                result.update(fold=fold, model=model_name, target=target, feature_count=len(feature_names),
                              sequence_length=SEQUENCE_LENGTH)
                rows.append(result)
            print(f"{fold} {model_name}: AP={overall['average_precision']:.4f} "
                  f"AUC={overall['roc_auc']:.4f} recall={overall['recall']:.4f}", flush=True)
            if fold == "test_2021_plus":
                final_models[model_name] = (model, scaler, cutoff)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "peer_index_deep_learning_metrics.csv", index=False)
    for model_name, (model, scaler, cutoff) in final_models.items():
        torch.save(
            {"state_dict": model.state_dict(), "feature_names": feature_names,
             "sequence_length": SEQUENCE_LENGTH, "threshold": cutoff,
             "model_name": model_name, "target_excluded": True,
             "target_tickers": TICKERS, "label_thresholds": thresholds},
            OUT / f"peer_index_{model_name}.pt",
        )
        joblib.dump(scaler, OUT / f"peer_index_{model_name}_scaler.joblib")
    report = [
        "Peer-index deep-learning crash forecast",
        "The target index is excluded from every input feature; only the other indices are used.",
        f"Targets: {TICKERS}; feature count: {len(feature_names)}; sequence length: {SEQUENCE_LENGTH}",
        f"Label thresholds: {json.dumps(thresholds, sort_keys=True)}",
        "Models: GRU and Transformer from pipeline.models.",
        "A 25-calendar-day purge separates train, validation, and test outcomes.",
        "The warning threshold is the validation 95th percentile.",
    ]
    (OUT / "peer_index_deep_learning_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nAggregate results:")
    print(results[results.target == "ALL"].groupby("model")[
        ["average_precision", "roc_auc", "brier", "precision", "recall", "f1"]
    ].mean().round(4).to_string())


if __name__ == "__main__":
    main()
