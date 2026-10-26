"""Streamlit interface for the cross-index attention forecasters.

Launch with ``py -3.12 -m streamlit run cross_index_app.py``. Users can upload
daily closes or use the bundled sample CSV with date, ^GSPC, ^DJI, ^IXIC, and
^RUT columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from cross_index_forecasting import (
    SEQ_LEN,
    TICKERS,
    TemporalCrossAttention,
    build_market_features,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "cross_index_forecasting"
SAMPLE = ROOT / "data" / "group_a_original_indices" / "prices.csv"
MODEL_MODES = ("self_only", "other_only", "all_indices")
REQUIRED_COLUMNS = {"date", *TICKERS}
MINIMUM_ROWS = 282


@st.cache_resource
def load_models():
    import joblib

    torch.set_num_threads(1)
    loaded = {}
    for mode in MODEL_MODES:
        model_path = RESULTS / f"{mode}.pt"
        scaler_path = RESULTS / f"{mode}_scaler.joblib"
        bundle = torch.load(model_path, map_location="cpu", weights_only=True)
        if bundle.get("mode") != mode:
            raise ValueError(f"{model_path.name} contains an unexpected model mode.")
        if bundle.get("ticker_order") != TICKERS:
            raise ValueError(f"{model_path.name} does not match the expected index order.")
        if bundle.get("sequence_length") != SEQ_LEN:
            raise ValueError(f"{model_path.name} does not match the expected sequence length.")
        model = TemporalCrossAttention(
            len(bundle["feature_names"]), mode, bundle["hidden_size"]
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        scaler = joblib.load(scaler_path)
        loaded[mode] = (model, scaler, bundle)
    return loaded


def read_price_csv(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    try:
        raw = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError) as exc:
        return None, f"Could not read the CSV: {exc}"

    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)

    raw = raw[["date", *TICKERS]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for ticker in TICKERS:
        raw[ticker] = pd.to_numeric(raw[ticker], errors="coerce")
    invalid_dates = int(raw["date"].isna().sum())
    if invalid_dates:
        return None, f"Found {invalid_dates} row(s) with an invalid date."

    duplicate_dates = int(raw["date"].duplicated(keep="last").sum())
    raw = raw.drop_duplicates("date", keep="last").sort_values("date")
    missing_values = int(raw[TICKERS].isna().sum().sum())
    raw = raw.dropna(subset=TICKERS)
    nonpositive = (raw[TICKERS] <= 0).any(axis=1)
    nonpositive_rows = int(nonpositive.sum())
    raw = raw.loc[~nonpositive]

    if len(raw) < MINIMUM_ROWS:
        return None, (
            f"Only {len(raw)} valid complete rows remain. At least {MINIMUM_ROWS} "
            "are needed for the 252-session rolling features and 30-session model input."
        )
    if raw["date"].duplicated().any():
        return None, "Dates must be unique after duplicate-date handling."

    raw.attrs["duplicate_dates"] = duplicate_dates
    raw.attrs["missing_value_count"] = missing_values
    raw.attrs["nonpositive_rows"] = nonpositive_rows
    return raw, None


def score_latest_prices(prices: pd.DataFrame) -> pd.DataFrame:
    wide, feature_names = build_market_features(prices)
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna()
    if len(wide) < SEQ_LEN:
        raise ValueError(
            "Not enough complete feature rows remain after rolling-window construction."
        )

    feature_tensor = wide.to_numpy(np.float32).reshape(
        len(wide), len(TICKERS), len(feature_names)
    )
    sequence = feature_tensor[-SEQ_LEN:][None, ...]
    loaded = load_models()
    scaled_sequences = {}
    for mode, (_, scaler, bundle) in loaded.items():
        if bundle["feature_names"] != feature_names:
            raise ValueError(f"{mode} model feature names do not match the uploaded data.")
        scaled = scaler.transform(sequence.reshape(-1, len(feature_names)))
        scaled_sequences[mode] = scaled.reshape(
            1, SEQ_LEN, len(TICKERS), len(feature_names)
        ).astype(np.float32)

    rows = []
    for ticker_idx, ticker in enumerate(TICKERS):
        row = {"date": wide.index[-1].date().isoformat(), "ticker": ticker}
        for mode, (model, _, bundle) in loaded.items():
            with torch.no_grad():
                logit = model(
                    torch.from_numpy(scaled_sequences[mode]),
                    torch.tensor([ticker_idx]),
                )
                row[f"{mode}_score"] = float(torch.sigmoid(logit).item())
            row[f"{mode}_validation_threshold"] = float(bundle["threshold"])
        row["peer_only_warning"] = (
            row["other_only_score"] >= row["other_only_validation_threshold"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Cross-Index Crash-Risk Research Demo",
    page_icon="📈",
    layout="wide",
)
st.title("Cross-Index Crash-Risk Research Demo")
st.markdown(
    "Upload daily closing prices to compare a target index's own history with "
    "signals from its peer markets. The app applies saved models; it does not "
    "train or update them."
)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Choose the bundled example or upload a CSV.\n"
        "2. Include `date`, `^GSPC`, `^DJI`, `^IXIC`, and `^RUT`.\n"
        "3. Provide at least 282 valid daily rows.\n"
        "4. Review the three model scores and export the result."
    )
    st.caption(
        "The latest 30 complete feature sessions are scored. Features use "
        "trailing observations only."
    )
    if SAMPLE.exists():
        st.download_button(
            "Download bundled sample CSV",
            data=SAMPLE.read_bytes(),
            file_name="cross_index_sample_prices.csv",
            mime="text/csv",
            use_container_width=True,
        )

input_choice = st.radio(
    "Choose input data",
    ("Bundled sample", "Upload a CSV"),
    horizontal=True,
)
uploaded = None
if input_choice == "Upload a CSV":
    uploaded = st.file_uploader(
        "Upload daily closing prices",
        type=["csv"],
        help="Required columns: date, ^GSPC, ^DJI, ^IXIC, ^RUT.",
    )
    if uploaded is None:
        st.info("Choose a CSV file to continue.")
        st.stop()
elif not SAMPLE.exists():
    st.error(f"The bundled sample file was not found: {SAMPLE}")
    st.stop()

source = SAMPLE if uploaded is None else uploaded
prices_csv, input_error = read_price_csv(source)
if input_error:
    st.error(input_error)
    st.stop()

prices = prices_csv.set_index("date")[TICKERS].astype(float)
st.subheader("Input data")
metric_columns = st.columns(4)
metric_columns[0].metric("Valid daily rows", f"{len(prices):,}")
metric_columns[1].metric("First date", prices.index.min().strftime("%Y-%m-%d"))
metric_columns[2].metric("Latest date", prices.index.max().strftime("%Y-%m-%d"))
metric_columns[3].metric("Indices", str(len(TICKERS)))

cleaning_notes = []
if prices_csv.attrs["duplicate_dates"]:
    cleaning_notes.append(f"{prices_csv.attrs['duplicate_dates']} duplicate date(s) removed")
if prices_csv.attrs["missing_value_count"]:
    cleaning_notes.append(
        f"{prices_csv.attrs['missing_value_count']} missing/non-numeric price value(s) "
        "removed with their rows"
    )
if prices_csv.attrs["nonpositive_rows"]:
    cleaning_notes.append(
        f"{prices_csv.attrs['nonpositive_rows']} row(s) with nonpositive prices removed"
    )
if cleaning_notes:
    st.warning("Input cleaning: " + "; ".join(cleaning_notes) + ".")
else:
    st.success("Input passed date, price, and completeness checks.")

normalized_prices = prices.div(prices.iloc[0]).mul(100)
st.caption("Price history, rebased to 100 at the first uploaded date")
st.line_chart(normalized_prices)

model_files = [
    file_path
    for mode in MODEL_MODES
    for file_path in (RESULTS / f"{mode}.pt", RESULTS / f"{mode}_scaler.joblib")
]
missing_models = [file_path for file_path in model_files if not file_path.is_file()]
if missing_models:
    st.error(
        "The input is valid, but saved checkpoints and scalers are needed to "
        "produce prediction scores. Run `py -3.12 cross_index_forecasting.py` "
        "when the experiment's training data is available, then rerun the app."
    )
    with st.expander("Missing model artifacts"):
        st.code("\n".join(str(path.relative_to(ROOT)) for path in missing_models))
    st.stop()

st.subheader("Latest model scores")
st.caption(
    "Each score is a model output, not a calibrated probability. The warning "
    "threshold was selected on validation data. A score above it is a research "
    "alert, not a guarantee of a crash."
)
try:
    result = score_latest_prices(prices)
except (OSError, RuntimeError, ValueError, KeyError) as exc:
    st.error(f"Could not score this input: {exc}")
    st.stop()

score_columns = st.columns(len(TICKERS))
for column, (_, row) in zip(score_columns, result.iterrows()):
    column.metric(
        row["ticker"],
        f"{row['other_only_score']:.3f}",
        "Peer-only warning" if row["peer_only_warning"] else "Below peer-only threshold",
        delta_color="inverse",
    )

display = result.rename(
    columns={
        "date": "Forecast date",
        "ticker": "Target index",
        "self_only_score": "Target-history score",
        "other_only_score": "Peer-only score",
        "all_indices_score": "All-indices score",
        "other_only_validation_threshold": "Peer-only threshold",
        "peer_only_warning": "Peer-only warning",
    }
)
st.dataframe(
    display[
        [
            "Forecast date",
            "Target index",
            "Target-history score",
            "Peer-only score",
            "Peer-only threshold",
            "Peer-only warning",
            "All-indices score",
        ]
    ].round(4),
    use_container_width=True,
    hide_index=True,
)

with st.expander("What do the three model views mean?"):
    st.markdown(
        "- **Target history:** sequence model using only the target index's history.\n"
        "- **Peer indices only:** temporal attention model masks the target's "
        "price/return history and attends to the other three indices.\n"
        "- **All four indices:** temporal attention model uses the target and "
        "all three peers.\n\n"
        f"The experiment labels a crash using a {15}-session forward horizon. "
        "The app scores the latest available feature date; it does not observe "
        "future outcomes or establish that an alert is correct."
    )

st.download_button(
    "Download scored CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="cross_index_scores.csv",
    mime="text/csv",
    use_container_width=True,
)

metrics_path = RESULTS / "cross_index_walk_forward_metrics.csv"
if metrics_path.is_file():
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["ticker"].eq("ALL")]
    if not metrics.empty:
        st.subheader("Historical chronological test results")
        st.caption(
            "These benchmark rows are separate from the uploaded inference. "
            "They summarize held-out historical folds and should be interpreted "
            "as experimental evidence, not a performance guarantee."
        )
        metric_columns = [
            column
            for column in (
                "fold",
                "model",
                "average_precision",
                "roc_auc",
                "brier",
                "recall",
                "alarm_rate",
            )
            if column in metrics.columns
        ]
        metrics = metrics[metric_columns].rename(
            columns={
                "fold": "Test period",
                "model": "Model",
                "average_precision": "Average precision",
                "roc_auc": "ROC AUC",
                "brier": "Brier score",
                "recall": "Recall",
                "alarm_rate": "Alarm rate",
            }
        )
        st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
else:
    st.info(
        "Historical benchmark results will appear here after the cross-index "
        "experiment has been run."
    )

st.caption(
    "Research prototype only. This tool is not financial advice and should not "
    "be used as a trading signal."
)
