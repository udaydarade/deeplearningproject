"""Extract the saved multi-entity TDA evidence for the presentation.

No model is fitted here. The cloud-size comparison is a descriptive extraction
from the completed V3-D/V5 walk-forward result files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V5 = ROOT / "data" / "v5"
FIG_DIR = ROOT / "presentation_figs"
SHOWCASE_DIR = FIG_DIR / "multi_entity_tda_showcase"
FIG_DIR.mkdir(exist_ok=True)
SHOWCASE_DIR.mkdir(exist_ok=True)
BLUE, CRIMSON, GOLD, TEAL, GREY = "#1769aa", "#b2182b", "#d98c00", "#16817a", "#4c566a"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12, "legend.fontsize": 9})


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def metric_cols(d: pd.DataFrame) -> pd.DataFrame:
    for c in ["auc", "ap", "prevalence", "event_recall", "recall", "mean_lead_time", "excess_event_recall"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "lift" not in d and {"ap", "prevalence"} <= set(d.columns):
        d["lift"] = d["ap"] / d["prevalence"].replace(0, np.nan)
    return d


def c3_top20() -> pd.DataFrame:
    d = read("data/v5/diagnostic_c3_auc.csv")
    d = d[(d["reference"] == "no") & d["signed_auc"].notna()].copy()
    d["abs_signed"] = d["signed_auc"].abs()
    # One representative score per feature: strongest absolute signed AUC across eras.
    idx = d.groupby(["feature", "block"])["abs_signed"].idxmax()
    out = d.loc[idx, ["feature", "block", "era", "auc", "signed_auc", "abs_signed"]].sort_values("abs_signed", ascending=False).head(20)
    return out.reset_index(drop=True)


def e2_summary() -> pd.DataFrame:
    d = metric_cols(read("data/v5/e2_results.csv"))
    d = d[(d["off_budget"] == False) & d["rule"].eq("RQ") & d["budget"].eq("5%")]
    return d.groupby("feature_set").agg(
        auc=("auc", "mean"), event_recall=("event_recall", "mean"),
        excess_recall=("excess_event_recall", "mean"), lift=("lift", "mean"),
        ap=("ap", "mean"), prevalence=("prevalence", "mean"), rows=("auc", "count"),
    ).reset_index().sort_values("auc", ascending=False)


def benchmark_summary() -> pd.DataFrame:
    d = metric_cols(read("data/benchmark/all_results.csv"))
    datasets = ["international", "energy", "tech", "group_b_cross_asset", "financials", "group_a_us_indices", "ALL"]
    d = d[(d["dataset"].isin(datasets)) & d["mode"].eq("landscape") & d["model"].isin(["tcn", "lstm", "bilstm", "gru"])]
    q = d.groupby(["dataset", "model"]).agg(auc=("roc_auc", "mean"), ap=("average_precision", "mean"), prevalence=("prevalence", "mean"), recall=("recall", "mean")).reset_index()
    q["lift"] = q["ap"] / q["prevalence"].replace(0, np.nan)
    best = q.sort_values(["dataset", "auc"], ascending=[True, False]).groupby("dataset", as_index=False).first()
    best["cloud"] = best["dataset"].map({
        "international": "International 7-index cloud", "energy": "Energy 5-stock cloud",
        "tech": "Tech 6-stock cloud", "group_b_cross_asset": "Cross-Asset 6-ETF cloud",
        "financials": "Financials 6-stock cloud", "group_a_us_indices": "US 4-index cloud", "ALL": "ALL benchmark cloud",
    })
    return best[["cloud", "dataset", "model", "auc", "lift", "recall", "ap", "prevalence"]].sort_values("cloud")


def scaling_summary() -> pd.DataFrame:
    d = metric_cols(read("data/v3/exp_d_results.csv"))
    # R3 is the saved strictest ranking rule; budget 0.5 is the closest
    # saved setting to a 5% alarm budget in this V3-D artifact.
    d = d[(d["rule"].eq("R3")) & d["budget"].eq(0.5) & d["subset"].eq("all")]
    q = d.groupby("universe").agg(
        auc=("auc", "mean"), ap=("ap", "mean"), prevalence=("prevalence", "mean"),
        event_recall=("event_recall", "mean"), lead_days=("day_recall", "mean"),
        folds=("fold", "nunique"), rows=("auc", "count"),
    ).reset_index()
    q["lift"] = q["ap"] / q["prevalence"].replace(0, np.nan)
    sizes = {"us4": 4, "approx30_index_etf": 30, "all_nonstock": 142, "all_assets": 159, "all_assets_since_1997": 159}
    labels = {"us4": "US4", "approx30_index_etf": "≈30 indices/ETFs", "all_nonstock": "142 non-stock", "all_assets": "159 all-assets", "all_assets_since_1997": "159 all-assets (1997+)"}
    q["entities"] = q.universe.map(sizes); q["cloud"] = q.universe.map(labels)
    return q.sort_values("entities")


def executed_summary() -> pd.DataFrame:
    path = V5 / "multi_entity_big_cloud_results.csv"
    if not path.exists():
        return pd.DataFrame()
    d = metric_cols(pd.read_csv(path))
    return d.groupby("feature_set").agg(
        eras=("era", "nunique"), auc=("auc", "mean"), event_recall=("event_recall", "mean"),
        excess_recall=("excess_event_recall", "mean"), lift=("lift", "mean"),
        ap=("ap", "mean"), prevalence=("prevalence", "mean"), off_budget=("off_budget", "sum"),
    ).reset_index()


def make_figure(scale: pd.DataFrame, bench: pd.DataFrame, e2: pd.DataFrame, executed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    if not scale.empty:
        x = scale.entities
        for ax, col, ylabel, title, color in [
            (axes[0], "auc", "ROC AUC", "ROC AUC vs. cloud size", BLUE),
            (axes[1], "event_recall", "Episode catch / recall (%)", "Pre-crash catch rate", CRIMSON),
            (axes[2], "lift", "AP / prevalence", "Lift over random", GOLD),
        ]:
            vals = scale[col] * 100 if col == "event_recall" else scale[col]
            ax.plot(x, vals, "o-", lw=2.8, ms=8, color=color)
            for xx, yy, label in zip(x, vals, scale.cloud):
                ax.annotate(label, (xx, yy), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, rotation=20)
            ax.set_xlabel("Entities in joint TDA cloud"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    if not e2.empty:
        # Add saved E2 points as a visual reference without hiding the scaling series.
        axes[0].scatter([159] * len(e2), e2.auc, marker="x", color=TEAL, label="V5 E2 block means", zorder=4)
        axes[0].legend(fontsize=8)
    if not executed.empty:
        means = executed.set_index("feature_set")
        note = "\n".join(f"{idx}: AUC {row.auc:.3f} | catch {row.event_recall:.1%} | lift {row.lift:.2f}x"
                         for idx, row in means.iterrows())
        axes[2].text(.98, .03, "Executed compact-block means\n" + note, transform=axes[2].transAxes,
                     ha="right", va="bottom", fontsize=7.5,
                     bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=.9))
    fig.suptitle("Multi-entity TDA showcase: larger joint clouds and topology blocks", weight="bold")
    fig.tight_layout()
    path = FIG_DIR / "fig10_multi_entity_tda_showcase.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(path, SHOWCASE_DIR / path.name)


def md_table(d: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
    out = d[cols].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "—")
    return out.to_markdown(index=False)


def write_report(top20: pd.DataFrame, e2: pd.DataFrame, bench: pd.DataFrame, scale: pd.DataFrame, executed: pd.DataFrame) -> None:
    lines = [
        "# Multi-Entity TDA Showcase",
        "",
        "This report extracts saved V2/V3-D/V5 results for the presentation. **No model training or new fitting was run.** Metrics are reported in the scale that makes the detector interpretable on slides: ROC AUC, episode catch rate, lift over random (`AP / prevalence`), and lead-time proxies.",
        "",
        "## Important metric and protocol note",
        "",
        "V5 E2 contains explicit `RQ` 5% rows. The saved V3-D file does not contain an `RQ` rule; it contains `R1/R2/R3` and budgets `0.5/1/2`. Therefore the cloud-size table uses the saved strict `R3`, budget `0.5`, all-subset, chronological fold rows as the closest existing walk-forward scaling extraction. It is not mislabeled as a new 5% RQ experiment.",
        "",
        "The V2 landscape benchmark contains ROC AUC, AP, prevalence, and day-level recall, but not an explicit event-recall column. Its `recall` is therefore labeled **saved recall proxy**, while V5 E2 reports the explicit `event_recall` field.",
        "",
        "## A. Top 20 multi-entity topological features",
        "",
        "For each feature, the strongest absolute signed AUC across saved eras was selected from `diagnostic_c3_auc.csv`; the sign is retained so directionality is visible.",
        "",
        md_table(top20, ["feature", "block", "era", "auc", "signed_auc", "abs_signed"]),
        "",
        "## B. V5 E2 head-to-head block comparison",
        "",
        "Rows are filtered to completed, on-budget `RQ` at the saved 5% alarm budget and averaged across available eras/models.",
        "",
        md_table(e2, ["feature_set", "rows", "auc", "event_recall", "excess_recall", "lift", "ap", "prevalence"]),
        "",
        "The saved E2 CSV contains `M0`, `B3`, and `M0+TDA_small` at this exact filter. It does **not** contain an on-budget `M0+TDA_big`, `DYN`, `CLUSTER`, or `PRICE1D` row at `RQ`/5%; those blocks exist as saved V5 feature artifacts and diagnostics, but their requested head-to-head E2 metric row is not present in the CSV.",
        "",
        "Saved multi-entity feature artifacts available under `data/v5/blocks/`: `V5_TDA.parquet`, `V5_DYN.parquet`, `V5_CLUSTER.parquet`, and `V5_landscape_sampled.parquet`.",
        "",
        "## C. V2 joint multi-entity 36-D landscape benchmark",
        "",
        "Best deep model is selected by mean ROC AUC among TCN/LSTM/BiLSTM/GRU landscape rows. The recall column is the saved day-level recall proxy, not an event ledger recall.",
        "",
        md_table(bench, ["cloud", "model", "auc", "lift", "recall", "ap", "prevalence"]),
        "",
        "## D. Cloud-size scaling from saved V3-D walk-forward rows",
        "",
        md_table(scale, ["cloud", "entities", "folds", "rows", "auc", "event_recall", "lift", "lead_days", "ap", "prevalence"]),
        "",
        "Here `lead_days` is the saved `day_recall` field because V3-D does not provide a dedicated mean-lead-time column. It is included as a lead-time proxy and should not be described as a crash-onset lead unless cross-checked against the event ledger.",
        "",
        "## E. Executed compact big-cloud/combined evaluation",
        "",
        "This section is newly executed from the saved V5 blocks using the existing V5 E2 walk-forward evaluator, `W` scheme, `RQ` 5% row, `logreg`, and recent eras E8/E9/E10. The compact combined block is exactly the 36-D `ALLS` landscape slice plus all saved `CLUSTER` features. `off_budget` is retained so rows are not silently presented as valid 5% alarms.",
        "",
        md_table(executed, ["feature_set", "eras", "auc", "event_recall", "excess_recall", "lift", "ap", "prevalence", "off_budget"]),
        "",
        "Interpretation: in this executed recent-era check, the saved cluster block is the strongest of the tested topology additions, while the compact 36-D landscape+cluster combination does not improve the baseline mean AUC/catch rate. This is an honest result: the project has the big-cloud machinery, but the current compact combined slice is not evidence of universal improvement.",
        "",
        "## Presentation claim",
        "",
        "The strongest defensible claim is that the project already contains multi-entity TDA blocks spanning 4 US indices, sector/international/crypto and cross-asset clouds, approximately 30-index/ETF clouds, and 142–159 global assets. The figure visualizes the saved walk-forward trend and the report preserves the exact source rows and metric definitions so the slide can distinguish descriptive scaling evidence from a newly trained experiment.",
        "",
        "Generated by `extract_multi_entity_tda_showcase.py`.",
    ]
    (ROOT / "MULTI_ENTITY_TDA_SHOWCASE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SHOWCASE_DIR / "MULTI_ENTITY_TDA_SHOWCASE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    top20, e2, bench, scale, executed = c3_top20(), e2_summary(), benchmark_summary(), scaling_summary(), executed_summary()
    make_figure(scale, bench, e2, executed)
    write_report(top20, e2, bench, scale, executed)
    print(f"Top20={len(top20)} E2 rows={len(e2)} benchmark clouds={len(bench)} scaling clouds={len(scale)} executed blocks={len(executed)}")


if __name__ == "__main__":
    main()
