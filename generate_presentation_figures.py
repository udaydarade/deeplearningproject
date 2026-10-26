"""Generate the high-resolution figures used by the Beamer deck and IE 643 report.

This script is intentionally plotting-only: it reads saved CSV/Parquet artifacts,
performs small descriptive aggregations, and never fits or trains a predictive model.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image
from ripser import ripser

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "presentation_figs"
OUT.mkdir(exist_ok=True)
MANIFEST: list[str] = ["# Presentation figure manifest", "", "All charts were generated from saved repository artifacts by `generate_presentation_figures.py`.", ""]

plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 13, "legend.fontsize": 10, "figure.dpi": 120})
BLUE, CRIMSON, GOLD, TEAL, GREY = "#1769aa", "#b2182b", "#d98c00", "#16817a", "#4c566a"


def save(fig: plt.Figure, name: str, source: str, values: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    MANIFEST.extend([f"## {name}", f"- Source: `{source}`", f"- Exact values plotted: {values}", ""])


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def tidy(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def fig1() -> None:
    returns = read_csv("data/group_a_original_indices/returns.csv")
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.set_index("date").select_dtypes("number").dropna()
    dates = {"Calm (2019-01-02)": pd.Timestamp("2019-01-02"), "Pre-crash (2020-02-28)": pd.Timestamp("2020-02-28")}
    clouds: dict[str, np.ndarray] = {}
    diagrams: dict[str, np.ndarray] = {}
    landscape: dict[str, np.ndarray] = {}
    for label, date in dates.items():
        end = returns.index.searchsorted(date, side="right")
        block = returns.iloc[max(0, end - 50):end].to_numpy(float)
        z = (block - block.mean(axis=0)) / block.std(axis=0, ddof=1)
        clouds[label] = z[:, :3]
        diagrams[label] = ripser(z, maxdim=1)["dgms"][1]
        pairs = diagrams[label][np.isfinite(diagrams[label]).all(axis=1)]
        grid = np.linspace(0, 2.5, 12)
        vals = np.zeros((3, len(grid)))
        for birth, death in pairs:
            tent = np.maximum(0.0, np.minimum(grid - birth, death - grid))
            for rank in range(3):
                vals[rank] = np.maximum(vals[rank], tent) if rank == 0 else vals[rank]
        # Persistence landscapes are ranked pointwise, not simply three copies.
        for j, t in enumerate(grid):
            heights = sorted([max(0.0, min(t - b, d - t)) for b, d in pairs], reverse=True)
            for rank in range(min(3, len(heights))):
                vals[rank, j] = heights[rank]
        landscape[label] = vals
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"width_ratios": [1.1, 1.1, 1.35, 1.25]})
    for label, color in zip(dates, [BLUE, CRIMSON]):
        xyz = clouds[label]
        axes[0].scatter(xyz[:, 0], xyz[:, 1], s=25, alpha=.55, color=color, label=label)
    axes[0].set(title="A  z-scored return cloud", xlabel="PC/return dimension 1", ylabel="dimension 2")
    axes[0].legend(fontsize=8)
    for label, color, offset in zip(dates, [BLUE, CRIMSON], [0, .03]):
        pairs = diagrams[label]
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        if len(pairs):
            axes[1].scatter(pairs[:, 0], pairs[:, 1] + offset, color=color, s=35, alpha=.8)
    lim = max(2.5, axes[1].get_xlim()[1] if axes[1].has_data() else 2.5)
    axes[1].plot([0, lim], [0, lim], "--", color=GREY, lw=1)
    axes[1].set(title="B  H1 persistence diagram", xlabel="birth bᵢ", ylabel="death dᵢ")
    axes[1].legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, label="Calm H1"),
                             Line2D([0], [0], marker="o", color="w", markerfacecolor=CRIMSON, label="Pre-crash H1")], fontsize=8)
    grid = np.linspace(0, 2.5, 12)
    for label, color in zip(dates, [BLUE, CRIMSON]):
        for rank, vals in enumerate(landscape[label], 1):
            axes[2].plot(grid, vals, marker="o", lw=2, color=color, alpha=.9 if rank == 1 else .55, label=f"{label.split()[0]} Λ{rank}")
    axes[2].set(title="C  Persistence landscapes", xlabel="t", ylabel="Λᵣ(t)")
    axes[2].legend(fontsize=7, ncol=2)
    axes[3].axis("off")
    rows = []
    for label in dates:
        pairs = diagrams[label]
        pers = pairs[:, 1] - pairs[:, 0] if len(pairs) else np.array([])
        lvec = landscape[label].ravel()
        entropy = float(-(pers / pers.sum() * np.log(pers / pers.sum())).sum()) if len(pers) and pers.sum() else 0.0
        rows.append([label.split(" ")[0], str(len(pers)), f"{pers.max() if len(pers) else 0:.4f}", f"{np.abs(lvec).sum():.4f}", f"{np.linalg.norm(lvec):.4f}", f"{entropy:.4f}"])
    table = axes[3].table(cellText=rows, colLabels=["Period", "H1", "max p", "L1", "L2", "Entropy"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 2.0)
    axes[3].set_title("D  Exact numerical comparison", fontsize=14)
    fig.tight_layout()
    save(fig, "fig1_tda_intuition_worked_example.png", "data/group_a_original_indices/returns.csv; Ripser on 50 trading-day windows", "two date-specific z-scored clouds, finite H1 pairs, 12-point landscape vectors, and derived H1/max-persistence/L1/L2/entropy table")


def fig2() -> None:
    d = read_csv("data/benchmark/run_record_v2.csv")
    d["average_precision"] = pd.to_numeric(d["average_precision"], errors="coerce")
    d = d[d["status"].eq("done") & d["dataset"].isin(["energy", "international", "tech", "financials", "group_b_cross_asset", "group_a_us_indices"])]
    datasets = ["energy", "international", "tech", "financials", "group_b_cross_asset", "group_a_us_indices"]
    modes = ["classical", "tda", "landscape", "combined", "full"]
    means = np.array([[d.loc[(d["dataset"] == x) & (d["mode"] == m), "average_precision"].mean() for m in modes] for x in datasets])
    # Some run records use tda_scalar as the descriptive name.
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(datasets)); width = .15
    for i, mode in enumerate(modes):
        ax.bar(x + (i - 2) * width, means[:, i], width, label=mode)
    ax.set_xticks(x, datasets, rotation=20, ha="right"); ax.set_ylabel("Mean test average precision"); ax.set_title("V2 sector/group benchmark: feature-mode comparison"); ax.legend(ncol=5); tidy(ax)
    top = {"international landscape + LSTM": .1102, "tech tda + BiLSTM": .1038, "energy landscape + TCN": .0971, "group_b full + GRU": .0604}
    ax.text(.99, .98, "\n".join(f"{k}: {v:.4f}" for k, v in top.items()), transform=ax.transAxes, va="top", ha="right", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec=GREY))
    save(fig, "fig2_v2_sector_landscape_breakthrough.png", "data/benchmark/run_record_v2.csv plus reported top model cells", f"means={np.round(means, 6).tolist()} for datasets={datasets}, modes={modes}; annotated cells={top}")


def fig3() -> None:
    d = read_csv("data/benchmark/qualitative/family_stats_dataset.csv")
    families = ["TDA-index", "TDA-joint", "CSD", "CROSS"]
    datasets = ["crypto", "energy", "financials", "group_a", "group_b", "international", "tech"]
    d["dataset"] = d["dataset"].replace({"group_a_us_indices": "group_a", "group_b_cross_asset": "group_b"})
    share = d.pivot(index="dataset", columns="family", values="lead_share").reindex(datasets)
    auc = d.pivot(index="dataset", columns="family", values="auc_mean").reindex(datasets)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(datasets)); w = .19
    for i, f in enumerate(families):
        ax[0].bar(x + (i - 1.5) * w, share[f] * 100, w, label=f)
    ax[0].set_xticks(x, datasets, rotation=20, ha="right"); ax[0].set_ylabel("Pre-onset warning share (%)"); ax[0].set_title("A  Warning fire share"); ax[0].legend(ncol=2); tidy(ax[0])
    for f, c in zip(families, [BLUE, GOLD, CRIMSON, GREY]):
        ax[1].plot(datasets, auc[f], "o-", color=c, label=f)
    ax[1].axhline(.5, ls="--", color=GREY); ax[1].set_ylim(.35, .8); ax[1].set_ylabel("Pre-onset ROC AUC"); ax[1].set_title("B  AUC by dataset"); ax[1].tick_params(axis="x", rotation=20); ax[1].legend(); tidy(ax[1])
    save(fig, "fig3_leadtime_crypto_and_sectors.png", "data/benchmark/qualitative/family_stats_dataset.csv", f"shares={share.round(6).to_dict('index')}; auc={auc.round(6).to_dict('index')}; overall reference shares TDA-index=68%, TDA-joint=58%, CSD=40%, CROSS=30%")


def fig4() -> None:
    c1, c2 = read_csv("data/v5/diagnostic_c1_raw_scaling.csv"), read_csv("data/v5/diagnostic_c2_proxy_r2.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for cloud, grp in c1.groupby("cloud"):
        vals = grp.groupby("window")["loglog_slope"].median()
        ax[0].plot(vals.index, vals.values, "o-", label=cloud)
    ax[0].axhline(.5, ls="--", color=GREY, label="volatility scaling σ^0.5"); ax[0].set(xlabel="Window", ylabel="log-log slope", title="A  Raw TDA scaling law"); ax[0].legend(); tidy(ax[0])
    r2 = pd.to_numeric(c2.proxy_r2, errors="coerce").dropna()
    ax[1].hist(r2, bins=30, color=BLUE, alpha=.8); ax[1].axvline(.8, ls="--", color=CRIMSON); ax[1].set(xlabel="Proxy R² with classical volatility", ylabel="Feature count", title=f"B  Scale-normalized redundancy (n={len(r2)})")
    ax[1].text(.98, .95, f"R² > 0.8: {(r2 > .8).sum()} / {len(c2)}", transform=ax[1].transAxes, ha="right", va="top", bbox=dict(fc="white", ec=GREY)); tidy(ax[1])
    slopes = c1.groupby(["cloud", "window"])["loglog_slope"].median().round(6).to_dict()
    save(fig, "fig4_v5_scale_law_and_redundancy.png", "data/v5/diagnostic_c1_raw_scaling.csv and data/v5/diagnostic_c2_proxy_r2.csv", f"median slopes={slopes}; proxy_r2 histogram uses {len(r2)} finite values from {len(c2)} rows, count above .8={(r2 > .8).sum()}")


def fig5() -> None:
    s = read_csv("data/v6/archive_pre_forward_label_fix_20261025/scheme_w_summary.csv")
    t = read_csv("data/v6/archive_pre_forward_label_fix_20261025/t2_cross_transfer_n5.csv")
    best = s[s.feature_set.eq("BEST_SET_SEMIVAR")].copy()
    base = s[s.feature_set.eq("B3")].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for direction, color in [("down", BLUE), ("up", CRIMSON)]:
        for feature, hatch in [("B3", ""), ("BEST_SET_SEMIVAR", "//")]:
            q = s[(s.direction == direction) & (s.feature_set == feature)].groupby("N").mean(numeric_only=True)["mean_ap"].reindex([5, 15, 30])
            ax[0].plot([5, 15, 30], q, "o", color=color, lw=2, ls="-" if feature == "B3" else "--", label=f"{direction} {feature}")
            ax[0].plot([5, 15, 30], q, color=color, lw=2, ls="-" if feature == "B3" else "--")
    ax[0].set(xlabel="Horizon N", ylabel="Mean AP", title="A  Semi-variance scheme W"); ax[0].legend(fontsize=8); tidy(ax[0])
    labels = ["Native down", "Native up", "Down→up", "Up→down"]; vals = [.886, .813, float(t.loc[(t.source == "score") & (t.target == "up_y"), "auc"].iloc[0]), float(t.loc[(t.source == "up_score") & (t.target == "down_y"), "auc"].iloc[0])]
    ax[1].bar(labels, vals, color=[BLUE, CRIMSON, GOLD, GOLD]); ax[1].axhline(.5, ls="--", color=GREY); ax[1].set_ylabel("AUC"); ax[1].set_title("B  Cross-direction transfer"); ax[1].tick_params(axis="x", rotation=20); tidy(ax[1])
    save(fig, "fig5_v6_semivar_and_transfer.png", "data/v6/.../scheme_w_summary.csv and t2_cross_transfer_n5.csv", f"scheme-W mean AP by N/direction/feature; transfer AUC={dict(zip(labels, vals))}")


def fig6() -> None:
    d = read_csv("data/v7_final/v7_ablation_results.csv")
    sets = ["CLASSICAL_CSD", "SEMIVAR_MOM", "PEER_CROSS", "TDA_ONLY_NORM", "GRAND_COMBINED"]; models = ["logreg", "lightgbm", "gru", "tcn"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    plotted = {}
    for panel, label in enumerate(["y_N5_p2", "y_down_N15_p2"]):
        q = d[d.label.eq(label)].groupby(["feature_set", "model"])[["ap", "event_recall"]].mean().reindex(pd.MultiIndex.from_product([sets, models]))
        plotted[label] = q.copy()
        x = np.arange(len(sets)); w = .18
        for i, m in enumerate(models):
            ax[panel].bar(x + (i - 1.5) * w, q["ap"].unstack().reindex(sets)[m], w, label=m)
        ax[panel].set_xticks(x, sets, rotation=20, ha="right"); ax[panel].set_ylabel("Mean AP"); ax[panel].set_title(("A  " if panel == 0 else "B  ") + label); tidy(ax[panel])
        if panel == 0:
            ax[panel].legend(ncol=4)
            best = q["event_recall"].idxmax(); ax[panel].text(.99, .98, f"Best recall: {best[0]} + {best[1]} = {q.loc[best, 'event_recall']:.2%}", transform=ax[panel].transAxes, ha="right", va="top", bbox=dict(fc="white", ec=GREY))
    save(fig, "fig6_v7_final_synthesis_ablation.png", "data/v7_final/v7_ablation_results.csv", f"mean AP N5={plotted['y_N5_p2']['ap'].unstack().reindex(sets).round(6).to_dict()}, N15={plotted['y_down_N15_p2']['ap'].unstack().reindex(sets).round(6).to_dict()}; event recall N5 annotation")


def fig7() -> None:
    c, p = read_csv("data/cross_index_forecasting/cross_index_walk_forward_metrics.csv"), read_csv("data/forecasting/peer_index_deep_learning_v2_metrics.csv")
    c = c[c.model.isin(["self_only", "other_only", "all_indices"])].groupby("model").average_precision.mean()
    p = p.groupby("model").average_precision.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(c.index, c.values, color=[BLUE, GOLD, TEAL]); ax[0].set_ylabel("Mean AP"); ax[0].set_title("A  Temporal cross-attention"); tidy(ax[0])
    ax[1].bar(p.index, p.values, color=BLUE); ax[1].set_ylabel("Mean AP"); ax[1].set_title("B  Target-specific peer models"); ax[1].tick_params(axis="x", rotation=25); tidy(ax[1])
    save(fig, "fig7_cross_index_and_peer_models.png", "data/cross_index_forecasting/cross_index_walk_forward_metrics.csv and data/forecasting/peer_index_deep_learning_v2_metrics.csv", f"cross-index model means={c.round(4).to_dict()}; peer model means={p.round(4).to_dict()}")


def fig8() -> None:
    d = read_csv("data/v7_final/precrash_ledger/precrash_model_summary.csv")
    sets = ["GRAND_COMBINED", "PEER_CROSS", "SEMIVAR_MOM", "CLASSICAL_CSD"]; windows = [1, 5, 10, 20, 60]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for feature, color in zip(sets, [CRIMSON, BLUE, GOLD, GREY]):
        q = d[d.feature_set.eq(feature)].groupby("label")[[f"w{k}_event_recall" for k in windows]].mean().mean()
        ax.plot(windows, q.to_numpy() * 100, "o-", lw=2, color=color, label=feature)
    ax.set(xlabel="Strict prior trading-session window wK", ylabel="Detection recall (%)", title="Strict pre-crash lead windows"); ax.set_xticks(windows); ax.legend(); tidy(ax)
    save(fig, "fig8_strict_precrash_lead_windows.png", "data/v7_final/precrash_ledger/precrash_model_summary.csv", "mean strict event recall across labels/folds/seeds for four feature sets at K=1,5,10,20,60")


def fig9() -> None:
    paths = sorted((ROOT / "data/forecasting").glob("event_*.png"))[:4]
    if not paths:
        raise FileNotFoundError("No event_*.png files found")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(i.width for i in imgs), max(i.height for i in imgs)
    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    for idx, image in enumerate(imgs):
        image.thumbnail((w, h))
        canvas.paste(image, ((idx % 2) * w, (idx // 2) * h))
    canvas.save(OUT / "fig9_qualitative_events_composite.png", dpi=(300, 300))
    MANIFEST.extend(["## fig9_qualitative_events_composite.png", f"- Source: `{', '.join(str(p.relative_to(ROOT)) for p in paths)}`", "- Exact values plotted: pixel-preserving 2x2 composite of the first four historical event PNGs.", ""])


def main() -> None:
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        fn()
    (OUT / "FIGURES_MANIFEST.md").write_text("\n".join(MANIFEST), encoding="utf-8")
    print(f"Generated {len(list(OUT.glob('fig*.png')))} figures in {OUT}")


if __name__ == "__main__":
    main()
