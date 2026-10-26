"""Target-specific peer-index deep-learning benchmark.

This is a separate experiment from the original and pooled peer-index runs.
Each target index gets its own model trained only on the other indices. The
input adds multi-scale peer returns, volatility shocks, breadth, correlation,
drawdown, and lagged peer summaries. A validation-selected ensemble combines
GRU, attention-BiLSTM, TCN, and Transformer scores.
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
SEED = 29
MODELS = ("gru", "attn_bilstm", "tcn", "transformer")
FOLDS = (
    ("test_2011_2015", "2011-01-01", "2015-12-31"),
    ("test_2016_2020", "2016-01-01", "2020-12-31"),
    ("test_2021_plus", "2021-01-01", "2100-01-01"),
)


def build_frame() -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    prices = pd.read_csv(GROUP / "prices.csv", parse_dates=["date"]).set_index("date")[TICKERS].dropna()
    returns = np.log(prices / prices.shift(1))
    calibration_position = prices.index.searchsorted(CALIBRATION_END, side="right") - 1
    calibration_last_origin = prices.index[calibration_position - HORIZON]
    forward_min, thresholds = {}, {}
    for target in TICKERS:
        paths = pd.concat(
            [prices[target].shift(-step) / prices[target] - 1 for step in range(1, HORIZON + 1)],
            axis=1,
        )
        forward_min[target] = paths.min(axis=1)
        fit = forward_min[target].loc[forward_min[target].index <= calibration_last_origin].dropna()
        thresholds[target] = float(fit.quantile(0.02))

    rows = []
    for target in TICKERS:
        peers = [ticker for ticker in TICKERS if ticker != target]
        peer = returns[peers]
        frame = pd.DataFrame(index=returns.index)
        for number, ticker in enumerate(peers, start=1):
            series = peer[ticker]
            frame[f"r{number}"] = series
            frame[f"r{number}_lag1"] = series.shift(1)
            frame[f"r{number}_lag5"] = series.rolling(5, min_periods=5).sum()
            vol20 = series.rolling(20, min_periods=20).std()
            vol60 = series.rolling(60, min_periods=60).std()
            frame[f"vol{number}"] = vol20
            frame[f"volshock{number}"] = vol20 / vol60.replace(0, np.nan)
            frame[f"drawdown{number}"] = prices[ticker] / prices[ticker].rolling(60, min_periods=60).max() - 1
        frame["peer_mean"] = peer.mean(axis=1)
        frame["peer_median"] = peer.median(axis=1)
        frame["peer_breadth"] = (peer < 0).mean(axis=1)
        frame["peer_dispersion"] = peer.std(axis=1)
        frame["peer_vol_mean"] = peer.rolling(20, min_periods=20).std().mean(axis=1)
        pair_corr = []
        for left_index, left in enumerate(peers):
            for right in peers[left_index + 1:]:
                pair_corr.append(peer[left].rolling(50, min_periods=50).corr(peer[right]))
        frame["peer_corr_mean"] = pd.concat(pair_corr, axis=1).mean(axis=1)
        frame["target"] = target
        frame["label"] = (forward_min[target] <= thresholds[target]).astype(float)
        frame.loc[forward_min[target].isna(), "label"] = np.nan
        rows.append(frame.reset_index())
    result = pd.concat(rows, ignore_index=True)
    features = [column for column in result.columns if column not in {"date", "target", "label"}]
    result = result.dropna(subset=features + ["label"]).sort_values(["target", "date"]).reset_index(drop=True)
    result["label"] = result["label"].astype(int)
    return result, thresholds, features


def sequences(frame: pd.DataFrame, features: list[str]):
    values, labels, dates = [], [], []
    for _, group in frame.groupby("target", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        x, y = group[features].to_numpy(np.float32), group["label"].to_numpy(np.float32)
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            values.append(x[end - SEQUENCE_LENGTH + 1:end + 1])
            labels.append(y[end])
            dates.append(group.loc[end, "date"])
    return np.stack(values), np.asarray(labels), pd.to_datetime(dates)


def score(y, p, cutoff):
    predicted = p >= cutoff
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)), "threshold": float(cutoff),
        "precision": float(np.sum(predicted & (y == 1)) / max(int(predicted.sum()), 1)),
        "recall": float(np.sum(predicted & (y == 1)) / max(int(y.sum()), 1)),
        "f1": float(f1_score(y, predicted.astype(int), zero_division=0)),
        "alarms": int(predicted.sum()),
    }


def train(model_name, x, y, train_idx, val_idx, test_idx):
    scaler = StandardScaler()
    scaler.fit(x[train_idx].reshape(-1, x.shape[-1]))
    z = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    pos = max(float(y[train_idx].sum()), 1.0)
    neg = max(float(len(train_idx) - y[train_idx].sum()), 1.0)
    model = build_model(model_name, x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(neg / pos, 20.0)))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=512, shuffle=True,
    )
    best_ap, best_state, stale = -1.0, None, 0
    for _ in range(18):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]))).numpy()
        ap = average_precision_score(y[val_idx], val_p) if y[val_idx].sum() else 0.0
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]))).numpy()
        test_p = torch.sigmoid(model(torch.from_numpy(z[test_idx]))).numpy()
    return model, scaler, val_p, test_p


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    frame, thresholds, features = build_frame()
    all_rows, final_models = [], {}
    for fold, test_start, test_end in FOLDS:
        x, y, dates = sequences(frame, features)
        start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
        train_end = start - pd.Timedelta(days=PURGE_DAYS)
        val_start = train_end - pd.DateOffset(years=2)
        train_idx = np.flatnonzero(dates < val_start - pd.Timedelta(days=PURGE_DAYS))
        val_idx = np.flatnonzero((dates >= val_start) & (dates <= train_end))
        test_idx = np.flatnonzero((dates >= start) & (dates <= end))
        test_predictions = {}
        validation_scores = {}
        for target_index, target in enumerate(TICKERS):
            offset = target_index * len(dates) // len(TICKERS)
            # Each target is contiguous because sequences() groups targets in TICKERS order.
            target_mask = np.arange(len(dates))
            if target_index == 0:
                target_mask = target_mask < len(dates) // len(TICKERS)
            elif target_index == len(TICKERS) - 1:
                target_mask = target_mask >= target_index * len(dates) // len(TICKERS)
            else:
                target_mask = (target_mask >= offset) & (target_mask < (target_index + 1) * len(dates) // len(TICKERS))
            tr, va, te = [idx[target_mask[idx]] for idx in (train_idx, val_idx, test_idx)]
            if min(len(tr), len(va), len(te)) == 0:
                continue
            model_outputs = {}
            for model_name in MODELS:
                model, scaler, val_p, test_p = train(model_name, x, y, tr, va, te)
                model_outputs[model_name] = (val_p, test_p)
                validation_scores[model_name] = average_precision_score(y[va], val_p)
                result = score(y[te], test_p, float(np.quantile(val_p, 0.95, method="higher")))
                result.update(fold=fold, model=model_name, target=target, feature_count=len(features),
                              sequence_length=SEQUENCE_LENGTH)
                all_rows.append(result)
                if fold == "test_2021_plus":
                    final_models[(target, model_name)] = (model, scaler)
            ranked = sorted(validation_scores, key=validation_scores.get, reverse=True)
            selected = ranked[:2]
            ensemble_val = np.mean([model_outputs[name][0] for name in selected], axis=0)
            ensemble_test = np.mean([model_outputs[name][1] for name in selected], axis=0)
            cutoff = float(np.quantile(ensemble_val, 0.95, method="higher"))
            result = score(y[te], ensemble_test, cutoff)
            result.update(fold=fold, model="ensemble_top2", target=target,
                          selected_models="+".join(selected), feature_count=len(features),
                          sequence_length=SEQUENCE_LENGTH)
            all_rows.append(result)
            print(f"{fold} {target}: selected={'+'.join(selected)} "
                  f"AP={result['average_precision']:.4f} AUC={result['roc_auc']:.4f} "
                  f"recall={result['recall']:.4f}", flush=True)
    results = pd.DataFrame(all_rows)
    results.to_csv(OUT / "peer_index_deep_learning_v2_metrics.csv", index=False)
    report = [
        "Target-specific peer-index deep-learning benchmark v2",
        "Each target has its own model; target returns are excluded from its features.",
        f"Feature count: {len(features)}; sequence length: {SEQUENCE_LENGTH}; models: {MODELS}",
        f"Label thresholds: {json.dumps(thresholds, sort_keys=True)}",
        "ensemble_top2 averages the two architectures with highest validation average precision.",
    ]
    (OUT / "peer_index_deep_learning_v2_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nAggregate target-specific results:")
    print(results.groupby("model")[["average_precision", "roc_auc", "brier", "precision", "recall", "f1"]].mean().round(4))


if __name__ == "__main__":
    main()
