"""Streamlit upload interface for the saved TDA + CSD crash forecaster.

Launch with ``py -3.12 -m streamlit run forecast_app.py``. The input is a
feature-panel CSV with date, ticker and the trained model's indicator columns.
The included data/forecasting/modeling_panel.csv is a ready-to-use example.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "forecasting"
V7 = ROOT / "data" / "v7_final"
MODEL_FILE = OUT / "combined_lstm.pt"
SCALER_FILE = OUT / "combined_scaler.joblib"
SEQ_LEN = 20


class CrashLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


@st.cache_resource
def load_assets():
    import joblib
    torch.set_num_threads(1)
    bundle = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    model = CrashLSTM(len(bundle["feature_names"]), bundle["hidden_size"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    scaler = joblib.load(SCALER_FILE)
    return model, scaler, bundle


st.set_page_config(page_title="Market Crash Early Warning", layout="wide")
st.title("Market Crash Early Warning")
st.write("Explore verified out-of-sample results and score a dated indicator panel for a statistically rare drawdown.")

v7_results_path = V7 / "v7_ablation_results.csv"
precrash_path = V7 / "precrash_ledger" / "precrash_by_event.csv"
if v7_results_path.exists():
    v7 = pd.read_csv(v7_results_path)
    st.subheader("V7 final synthesis: configured-cell results")
    st.caption(
        "240 completed cells across two horizons, five feature sets, four model "
        "families, three chronological folds, and three seeds for GRU/TCN. "
        "These folds are 2011–2021+, not the full 1927–2026 source history."
    )
    v7_summary = v7.groupby(["label", "feature_set", "model"], as_index=False).agg(
        cells=("seed", "count"),
        mean_ap=("ap", "mean"),
        mean_auc=("auc", "mean"),
        event_recall=("event_recall", "mean"),
        alarm_rate=("alarm_day_rate", "mean"),
        mean_lead=("mean_lead_time", "mean"),
    )
    st.dataframe(v7_summary.round(4), width="stretch", hide_index=True)

if precrash_path.exists():
    precrash = pd.read_csv(precrash_path, parse_dates=["crash_start", "crash_end"])
    with st.expander("Strict pre-crash event lookup", expanded=False):
        st.caption(
            "A valid detection is an alarm strictly before crash_start. "
            "Legacy alarms during the crash are excluded."
        )
        selected_asset = st.selectbox(
            "Asset", sorted(precrash["asset"].unique()), key="precrash_asset"
        )
        event_rows = precrash[precrash["asset"].eq(selected_asset)].copy()
        st.dataframe(
            event_rows[
                [
                    "event_id", "label", "crash_start", "crash_end",
                    "w5_models_detected", "w5_detection_fraction",
                    "w5_earliest_alarm", "w5_earliest_lead_sessions",
                    "w20_models_detected", "w20_earliest_lead_sessions",
                ]
            ].sort_values("crash_start").tail(100),
            width="stretch",
            hide_index=True,
        )

if (OUT / "walk_forward_metrics.csv").exists():
    benchmark = pd.read_csv(OUT / "walk_forward_metrics.csv")
    benchmark = benchmark[benchmark.ticker.isna()].groupby("model", as_index=False).agg(
        average_precision=("average_precision", "mean"),
        roc_auc=("roc_auc", "mean"),
        brier=("brier", "mean"),
    )
    benchmark["model"] = benchmark.model.str.replace("LSTM_", "", regex=False).str.capitalize()
    st.subheader("Chronological out-of-sample benchmark")
    st.dataframe(benchmark.round(4), width="stretch", hide_index=True)
    st.caption("Mean across 2011–2015, 2016–2020, and 2021–September 2026 test blocks; see FINAL_PROJECT_REPORT.md for prevalence and alert-rate context.")

if not MODEL_FILE.exists() or not SCALER_FILE.exists():
    st.error("Model artifacts are missing. Run `py -3.12 crash_forecasting.py` from the project folder first.")
    st.stop()

model, scaler, bundle = load_assets()
features = bundle["feature_names"]
threshold = float(bundle["threshold"])
uploaded = st.file_uploader("Upload feature-panel CSV", type=["csv"])
if uploaded is None:
    st.info("Demo input: `data/forecasting/modeling_panel.csv`. Required fields: date, ticker, and the model feature columns. Extra label columns are ignored during scoring.")
    st.stop()

try:
    source = pd.read_csv(uploaded, parse_dates=["date"])
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()
required = {"date", "ticker", *features}
missing = sorted(required - set(source.columns))
if missing:
    st.error("CSV is missing required columns: " + ", ".join(missing))
    st.stop()

source = source.dropna(subset=["date", "ticker", *features]).copy()
if source.empty:
    st.error("No complete dated feature rows were found.")
    st.stop()

scored = []
for ticker, group in source.groupby("ticker", sort=False):
    group = group.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(group) < SEQ_LEN:
        continue
    raw = group[features].to_numpy(float)
    z = scaler.transform(raw).astype(np.float32)
    seq = np.stack([z[i - SEQ_LEN + 1:i + 1] for i in range(SEQ_LEN - 1, len(z))])
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(seq))).numpy()
    result = group.iloc[SEQ_LEN - 1:].copy()
    result["risk_score"] = probs
    result["warning"] = probs >= threshold
    scored.append(result)

if not scored:
    st.error("Each ticker needs at least 20 consecutive complete feature rows.")
    st.stop()
result = pd.concat(scored, ignore_index=True).sort_values(["date", "ticker"])
latest = result.sort_values("date").groupby("ticker", as_index=False).tail(1)

st.subheader("Latest forecast by index")
st.caption(f"Warning threshold selected on validation data: {threshold:.3f}. Scores are not probability-calibrated and a warning does not guarantee a crash.")
st.dataframe(latest[["date", "ticker", "risk_score", "warning"]].sort_values("ticker"), width="stretch", hide_index=True)

st.subheader("Recent forecast history")
tickers = sorted(result.ticker.unique())
selected = st.selectbox("Index", tickers)
chart = result[result.ticker == selected].set_index("date")[["risk_score"]]
st.line_chart(chart)

if "label" in result.columns and result.label.nunique() > 1:
    y = result.label.astype(int).to_numpy()
    p = result.risk_score.to_numpy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average precision", f"{average_precision_score(y, p):.3f}")
    c2.metric("ROC AUC", f"{roc_auc_score(y, p):.3f}")
    c3.metric("Brier score", f"{brier_score_loss(y, p):.3f}")
    st.caption("This upload-level diagnostic is not out-of-sample unless the uploaded rows are a held-out period. The chronological benchmark above is the reported model result.")

st.download_button("Download scored CSV", result.to_csv(index=False).encode("utf-8"), "crash_forecast_scores.csv", "text/csv")
