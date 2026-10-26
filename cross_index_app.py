"""CSV upload interface for the cross-index attention forecasters.

Launch with ``py -3.12 -m streamlit run cross_index_app.py``. Upload a
date-indexed CSV of daily closes with columns date, ^GSPC, ^DJI, ^IXIC, ^RUT.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from cross_index_forecasting import OUT, TICKERS, SEQ_LEN, TemporalCrossAttention, build_market_features

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "cross_index_forecasting"
V7 = ROOT / "data" / "v7_final"
MODEL_MODES = ["self_only", "other_only", "all_indices"]


@st.cache_resource
def load_models():
    import joblib
    torch.set_num_threads(1)
    loaded = {}
    for mode in MODEL_MODES:
        bundle = torch.load(RESULTS / f"{mode}.pt", map_location="cpu", weights_only=True)
        model = TemporalCrossAttention(len(bundle["feature_names"]), mode, bundle["hidden_size"])
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        scaler = joblib.load(RESULTS / f"{mode}_scaler.joblib")
        loaded[mode] = (model, scaler, bundle)
    return loaded


st.set_page_config(page_title="Cross-Index Crash Forecaster", layout="wide")
st.title("Cross-Index Crash Forecaster")
st.write("Compare a target-index history with signals from its other three indices. The peer-only attention model masks the target index's full price and return history.")

v7_results_path = V7 / "v7_ablation_results.csv"
if v7_results_path.exists():
    v7 = pd.read_csv(v7_results_path)
    st.subheader("Related V7 benchmark")
    st.caption(
        "The V7 synthesis combines classical, SEMIVAR, peer/cross-index, and "
        "scale-normalized TDA features. Results below are chronological "
        "out-of-sample summaries for the configured 2011–2021+ folds."
    )
    summary = v7.groupby(["label", "feature_set", "model"], as_index=False).agg(
        cells=("seed", "count"),
        mean_ap=("ap", "mean"),
        mean_auc=("auc", "mean"),
        event_recall=("event_recall", "mean"),
    )
    st.dataframe(summary.round(4), width="stretch", hide_index=True)

metrics_path = RESULTS / "cross_index_walk_forward_metrics.csv"
if metrics_path.exists():
    m = pd.read_csv(metrics_path)
    m = m[m.ticker.eq("ALL")].groupby("model", as_index=False).agg(
        average_precision=("average_precision", "mean"),
        roc_auc=("roc_auc", "mean"),
        brier=("brier", "mean"),
    )
    st.subheader("Chronological out-of-sample benchmark")
    st.dataframe(m.round(4), width="stretch", hide_index=True)

uploaded = st.file_uploader("Upload daily closing prices CSV", type=["csv"])
st.caption("Required headers: date, ^GSPC, ^DJI, ^IXIC, ^RUT. Include at least 282 consecutive trading dates for the trailing features and 30-session sequence.")
if uploaded is None:
    st.info("Demo file: `data/group_a_original_indices/prices.csv`.")
    st.stop()

try:
    raw = pd.read_csv(uploaded, parse_dates=["date"])
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()
missing = sorted({"date", *TICKERS} - set(raw.columns))
if missing:
    st.error("CSV is missing required columns: " + ", ".join(missing))
    st.stop()
raw = raw[["date", *TICKERS]].dropna().drop_duplicates("date", keep="last").sort_values("date")
if len(raw) < 282:
    st.error(f"Need at least 282 complete rows; found {len(raw)}.")
    st.stop()

prices = raw.set_index("date")[TICKERS].astype(float)
wide, feature_names = build_market_features(prices)
wide = wide.dropna()
if len(wide) < SEQ_LEN:
    st.error("Not enough complete feature rows remain after rolling-window construction.")
    st.stop()
tensor = wide.to_numpy(np.float32).reshape(len(wide), len(TICKERS), len(feature_names))
seq = np.lib.stride_tricks.sliding_window_view(tensor, SEQ_LEN, axis=0).transpose(0, 3, 1, 2)
seq = np.asarray(seq, dtype=np.float32)
loaded = load_models()
scored = []
for ticker_idx, ticker in enumerate(TICKERS):
    row = {"date": wide.index[-1], "ticker": ticker}
    for mode, (model, scaler, bundle) in loaded.items():
        x = seq[-1:].reshape(-1, len(feature_names))
        x = scaler.transform(x).reshape(1, SEQ_LEN, len(TICKERS), len(feature_names)).astype(np.float32)
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.tensor([ticker_idx]))
            row[f"{mode}_score"] = float(torch.sigmoid(logit).item())
        row[f"{mode}_validation_threshold"] = float(bundle["threshold"])
    row["peer_only_warning"] = row["other_only_score"] >= row["other_only_validation_threshold"]
    scored.append(row)

result = pd.DataFrame(scored)
st.subheader("Latest cross-index forecast scores")
st.caption("The score is not a calibrated probability. Warning thresholds were selected on validation data; current market regimes may change alert rates.")
st.dataframe(result.round(4), width="stretch", hide_index=True)
st.download_button("Download scores", result.to_csv(index=False).encode("utf-8"), "cross_index_scores.csv", "text/csv")
