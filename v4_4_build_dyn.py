"""Build the past-only V4 dynamic summary block from V4_RAW."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V4 = ROOT / "data" / "v4"


def slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.nan
    x = np.arange(len(values))[valid]
    return float(np.polyfit(x, values[valid], 1)[0])


def add_dynamics(frame: pd.DataFrame, source: str, output: dict) -> None:
    values = frame[source].to_numpy(float)
    for window in (10, 20, 40):
        output[f"{source}_slope{window}"] = slope(values[-window:])
    for window in (20, 40):
        output[f"{source}_std{window}"] = float(np.nanstd(values[-window:]))
    tail = values[-40:]
    output[f"{source}_ac1_40"] = float(pd.Series(tail).autocorr(1)) if np.isfinite(tail).sum() > 2 else np.nan
    past = values[-250:]
    valid = past[np.isfinite(past)]
    if len(valid) >= 20:
        mean, std = valid.mean(), valid.std()
        output[f"{source}_z250"] = float((values[-1] - mean) / std) if std > 0 else 0.0
        output[f"{source}_distmax250"] = float(values[-1] - np.nanmax(valid))
    else:
        output[f"{source}_z250"] = np.nan
        output[f"{source}_distmax250"] = np.nan


def main() -> None:
    raw = pd.read_parquet(V4 / "blocks" / "V4_RAW.parquet").sort_values("date")
    returns = pd.read_parquet(V3 / "returns.parquet").astype(float)
    records = []
    sources = [
        c for c in raw.columns
        if any(token in c for token in ("A_corr", "A_raw", "T_raw"))
        and ("k24_w60" in c)
        and any(token in c for token in ("_h1_l1", "_h1_l2", "_h1_total", "_h0_total"))
    ]
    for i, row in raw.iterrows():
        current = raw.iloc[:i + 1]
        output = {"date": row.date}
        for source in sources:
            add_dynamics(current, source, output)
        date = pd.Timestamp(row.date)
        if date in returns.index:
            block = returns.loc[:date].tail(60)
            corr = block.corr().to_numpy(float)
            upper = corr[np.triu_indices_from(corr, 1)]
            output["mean_pair_corr_w60"] = float(np.nanmean(upper))
            eig = np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0))
            output["lambda1_over_n_w60"] = float(eig[-1] / len(corr))
        records.append(output)
    result = pd.DataFrame(records)
    result.to_parquet(V4 / "blocks" / "V4_DYN.parquet", index=False)
    print(f"[complete] V4_DYN rows={len(result):,} columns={len(result.columns)-1}")


if __name__ == "__main__":
    main()
