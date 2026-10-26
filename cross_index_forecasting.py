"""Cross-market attention crash forecaster (separate from the TDA/CSD work).

Compares a target-only LSTM, a leave-one-index-out temporal cross-attention
model, and an all-indices temporal cross-attention model. The peer-only model
masks the target index's price/return history before temporal encoding, then
uses its identity only as a learned query. Labels come from the fixed, calibration-only
thresholds recorded by crash_forecasting.py.

Run with ``py -3.12 cross_index_forecasting.py``. All new outputs go to
``data/cross_index_forecasting/``; existing project outputs are untouched.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
BASE = ROOT / "data" / "forecasting"
OUT = ROOT / "data" / "cross_index_forecasting"
TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
HORIZON = 15
SEQ_LEN = 30
SEED = 41
FOLDS = [
    ("test_2011_2015", "2011-01-01", "2015-12-31"),
    ("test_2016_2020", "2016-01-01", "2020-12-31"),
    ("test_2021_plus", "2021-01-01", "2100-01-01"),
]


def build_market_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build strictly trailing per-index features from a date x ticker price table."""
    prices = prices[TICKERS].astype(float).sort_index()
    r = np.log(prices / prices.shift(1))
    frames = {}
    feature_names = ["return_1d", "momentum_5d", "momentum_20d", "momentum_60d",
                     "volatility_20d", "volatility_60d", "downside_volatility_20d",
                     "drawdown_60d", "drawdown_252d", "volatility_ratio_20_60", "skew_20d"]
    for ticker in TICKERS:
        ri, pi = r[ticker], prices[ticker]
        roll20, roll60 = ri.rolling(20, min_periods=20), ri.rolling(60, min_periods=60)
        feat = pd.DataFrame(index=prices.index)
        feat["return_1d"] = ri
        for n in (5, 20, 60):
            feat[f"momentum_{n}d"] = ri.rolling(n, min_periods=n).sum()
        feat["volatility_20d"] = roll20.std()
        feat["volatility_60d"] = roll60.std()
        downside_sq = ri.clip(upper=0).pow(2)
        feat["downside_volatility_20d"] = np.sqrt(downside_sq.rolling(20, min_periods=20).mean())
        feat["drawdown_60d"] = pi / pi.rolling(60, min_periods=60).max() - 1
        feat["drawdown_252d"] = pi / pi.rolling(252, min_periods=252).max() - 1
        feat["volatility_ratio_20_60"] = feat.volatility_20d / feat.volatility_60d.clip(lower=1e-12)
        feat["skew_20d"] = roll20.skew()
        frames[ticker] = feat[feature_names]
    tensor = np.stack([frames[t].to_numpy(np.float32) for t in TICKERS], axis=1)
    wide = pd.DataFrame(
        tensor.reshape(len(prices), -1), index=prices.index,
        columns=[f"{ticker}__{name}" for ticker in TICKERS for name in feature_names],
    )
    return wide, feature_names


def make_sequences(feature_tensor: np.ndarray, label_matrix: np.ndarray,
                   dates: pd.DatetimeIndex):
    # All index features are complete together; drop early rolling-window NaNs
    # and the final 15 dates whose forward labels are not yet observable.
    valid = np.isfinite(feature_tensor).all(axis=(1, 2)) & np.isfinite(label_matrix).all(axis=1)
    values, labels, dates = feature_tensor[valid], label_matrix[valid], dates[valid]
    windows = np.lib.stride_tricks.sliding_window_view(values, SEQ_LEN, axis=0)
    windows = windows.transpose(0, 3, 1, 2).astype(np.float32, copy=False)
    sample_dates = dates[SEQ_LEN - 1:]
    xs, ys, targets = [], [], []
    for target in range(len(TICKERS)):
        xs.append(windows)
        ys.append(labels[SEQ_LEN - 1:, target])
        targets.append(np.full(len(sample_dates), target, dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys).astype(np.float32), \
        np.concatenate([sample_dates] * len(TICKERS)), np.concatenate(targets)


class TemporalCrossAttention(nn.Module):
    def __init__(self, n_features: int, mode: str, hidden: int = 32):
        super().__init__()
        self.mode = mode
        self.n_assets = len(TICKERS)
        self.hidden = hidden
        self.temporal = nn.LSTM(n_features, hidden, batch_first=True)
        self.asset_embedding = nn.Embedding(self.n_assets, hidden)
        self.target_query = nn.Embedding(self.n_assets, hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1,
                                                     batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x, target):
        batch, steps, assets, _ = x.shape
        if self.mode == "self_only":
            own = x[torch.arange(batch, device=x.device), :, target, :]
            _, (h, _) = self.temporal(own)
            return self.head(h[-1]).squeeze(-1)
        if self.mode == "other_only":
            x = x.clone()
            x[torch.arange(batch, device=x.device), :, target, :] = 0.0
        # First encode each index's entire trailing history. A learned target
        # query then attends to the four encoded market histories.
        per_asset = x.permute(0, 2, 1, 3).reshape(batch * assets, steps, -1)
        _, (h, _) = self.temporal(per_asset)
        asset_ids = torch.arange(assets, device=x.device)
        nodes = h[-1].reshape(batch, assets, self.hidden) + self.asset_embedding(asset_ids)[None, :, :]
        query = self.target_query(target).unsqueeze(1)
        key_padding = None
        if self.mode == "other_only":
            key_padding = torch.zeros(batch, assets, dtype=torch.bool, device=x.device)
            key_padding[torch.arange(batch, device=x.device), target] = True
        context, _ = self.cross_attention(query, nodes, nodes, key_padding_mask=key_padding,
                                          need_weights=False)
        return self.head(context[:, 0, :]).squeeze(-1)


def fit_scaler(x, targets, train_idx, mode):
    pieces = []
    train_targets = targets[train_idx]
    xtrain = x[train_idx]
    for target in range(len(TICKERS)):
        samples = xtrain[train_targets == target]
        if not len(samples):
            continue
        if mode == "self_only":
            samples = samples[:, :, target:target + 1, :]
        elif mode == "other_only":
            keep = [i for i in range(len(TICKERS)) if i != target]
            samples = samples[:, :, keep, :]
        pieces.append(samples.reshape(-1, x.shape[-1]))
    scaler = StandardScaler().fit(np.concatenate(pieces, axis=0))
    z = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    return scaler, z


def eval_metrics(y, p, threshold):
    pred = p >= threshold
    positives = int(y.sum())
    return {
        "n": int(len(y)), "positives": positives, "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if positives else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "precision": float(np.sum(pred & (y == 1)) / max(pred.sum(), 1)),
        "recall": float(np.sum(pred & (y == 1)) / max(positives, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alarms": int(pred.sum()), "alarm_rate": float(pred.mean()),
        "threshold": float(threshold),
    }


def fit_predict(x, y, targets, train_idx, val_idx, test_idx, mode, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    scaler, z = fit_scaler(x, targets, train_idx, mode)
    model = TemporalCrossAttention(x.shape[-1], mode)
    train_ds = TensorDataset(torch.from_numpy(z[train_idx]),
                             torch.from_numpy(targets[train_idx]),
                             torch.from_numpy(y[train_idx]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=512, shuffle=True, generator=generator)
    positives = max(float(y[train_idx].sum()), 1.0)
    negatives = max(float(len(train_idx) - y[train_idx].sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(negatives / positives, 10.0)]))
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    best_ap, best_state, stale = -1.0, None, 0
    for _ in range(10):
        model.train()
        for xb, tb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, tb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(z[val_idx]), torch.from_numpy(targets[val_idx]))
            val_p = torch.sigmoid(val_logits).numpy()
        ap = average_precision_score(y[val_idx], val_p) if y[val_idx].sum() else 0.0
        if ap > best_ap + 1e-5:
            best_ap, best_state, stale = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]), torch.from_numpy(targets[val_idx]))).numpy()
        test_p = torch.sigmoid(model(torch.from_numpy(z[test_idx]), torch.from_numpy(targets[test_idx]))).numpy()
    threshold = float(np.quantile(val_p, 0.95, method="higher"))
    return model, scaler, val_p, test_p, threshold


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(GROUP / "prices.csv", parse_dates=["date"]).set_index("date")[TICKERS]
    panel = pd.read_csv(BASE / "modeling_panel.csv", parse_dates=["date"])
    label_table = panel.pivot(index="date", columns="ticker", values="label").reindex(columns=TICKERS)
    common_dates = prices.index.intersection(label_table.index)
    prices = prices.loc[common_dates]
    label_table = label_table.loc[common_dates]
    wide, feature_names = build_market_features(prices)
    # wide feature file is the input contract for the upload interface.
    wide.to_csv(OUT / "cross_market_feature_panel.csv", date_format="%Y-%m-%d")
    tensor = wide.to_numpy(np.float32).reshape(len(wide), len(TICKERS), len(feature_names))
    x, y, sample_dates, targets = make_sequences(tensor, label_table.to_numpy(float), common_dates)
    modes = ("self_only", "other_only", "all_indices")
    rows, final_models = [], {}
    for fold_i, (fold, start_s, end_s) in enumerate(FOLDS):
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        train_end = start - pd.Timedelta(days=25)
        val_start = train_end - pd.DateOffset(years=2)
        train_idx = np.flatnonzero(sample_dates < val_start - pd.Timedelta(days=25))
        val_idx = np.flatnonzero((sample_dates >= val_start) & (sample_dates <= train_end))
        test_idx = np.flatnonzero((sample_dates >= start) & (sample_dates <= end))
        for mode_i, mode in enumerate(modes):
            model, scaler, val_p, test_p, threshold = fit_predict(
                x, y, targets, train_idx, val_idx, test_idx, mode,
                SEED + fold_i * 100 + mode_i,
            )
            aggregate = eval_metrics(y[test_idx], test_p, threshold)
            aggregate.update(fold=fold, model=mode, ticker="ALL", train_samples=len(train_idx),
                             validation_samples=len(val_idx),
                             test_start=str(pd.Timestamp(sample_dates[test_idx].min()).date()),
                             test_end=str(pd.Timestamp(sample_dates[test_idx].max()).date()),
                             threshold_source="validation_95th_percentile")
            rows.append(aggregate)
            print(f"{fold} {mode}: AP={aggregate['average_precision']:.4f} n={aggregate['n']}", flush=True)
            for ticker_i, ticker in enumerate(TICKERS):
                keep = targets[test_idx] == ticker_i
                if not keep.any():
                    continue
                one = eval_metrics(y[test_idx][keep], test_p[keep], threshold)
                one.update(fold=fold, model=mode, ticker=ticker)
                rows.append(one)
            if fold == "test_2021_plus":
                final_models[mode] = (model, scaler, threshold)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "cross_index_walk_forward_metrics.csv", index=False)
    metadata = {
        "ticker_order": TICKERS, "feature_names": feature_names,
        "sequence_length": SEQ_LEN, "horizon_sessions": HORIZON,
        "random_seed_base": SEED, "label_threshold_file": "../forecasting/label_thresholds.csv",
        "models": modes,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for mode, (model, scaler, threshold) in final_models.items():
        torch.save({"state_dict": model.state_dict(), "mode": mode, "hidden_size": 32,
                    "feature_names": feature_names, "ticker_order": TICKERS,
                    "sequence_length": SEQ_LEN, "threshold": threshold}, OUT / f"{mode}.pt")
        import joblib
        joblib.dump(scaler, OUT / f"{mode}_scaler.joblib")
    summary = result[result.ticker.eq("ALL")].groupby("model")[["average_precision", "roc_auc", "brier", "precision", "recall", "alarm_rate"]].mean()
    report = [
        "Cross-index temporal attention crash forecasting",
        "Crash targets and frozen per-index thresholds are inherited from data/forecasting/label_thresholds.csv.",
        "Inputs are trailing returns, momentum, realized/downside volatility, rolling drawdowns, volatility ratios, and rolling skewness for up to all four indices.",
        "other_only zeroes the target-index input history before temporal encoding, then attends only to the other three indices using a learned target-index query.",
        "all_indices uses all four source histories. self_only uses only the target index as a target-only LSTM comparison.",
        "The test blocks, validation-selected threshold metrics, and per-index scores are in cross_index_walk_forward_metrics.csv.",
        "Scores are single-seed preliminary results; repeated-seed and event-level uncertainty analysis is still needed.",
        "\nMean across the three test blocks:", summary.round(4).to_string(),
    ]
    (OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote new artifacts to {OUT}")


if __name__ == "__main__":
    main()
