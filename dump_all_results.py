import pandas as pd

pd.set_option("display.width", 250)


def sec(title):
    print()
    print("#" * 10, title, "#" * 10)


# A. Team-original scripts (pre-benchmark)
sec("A1. crash_forecasting.py (group A, original LSTM)" )
d = pd.read_csv("data/forecasting/walk_forward_metrics.csv")
print(d[d.ticker.isna()].groupby("model")[
    ["average_precision", "roc_auc", "precision", "recall", "f1", "alarms"]].mean().round(4).to_string())
print("(per-ticker AP):")
print(d[d.ticker.notna()].pivot_table(index="ticker", columns="model",
                                      values="average_precision").round(4).to_string())

sec("A2. cross_asset_forecasting.py (6-group panel, LSTM)")
try:
    d2 = pd.read_csv("data/forecasting/cross_asset_walk_forward_metrics.csv")
    o2 = d2[d2.ticker.isna()] if "ticker" in d2.columns else d2
    print(o2.groupby("model")[
        ["average_precision", "roc_auc", "precision", "recall"]].mean().round(4).to_string())
except Exception as e:
    print("ERR", e)

# B. ismail reproduction
sec("A3. ismail_2022_reproduction (statistical tests only)")
try:
    cls = pd.read_csv("data/group_a_original_indices/ismail_2022_reproduction/classification_report.csv")
    print(cls.to_string()[:2000])
except Exception as e:
    print("no classification_report:", e)

# C. v1 unified grid
sec("C1. v1 unified grid: mode x level (mean AP)")
r1 = pd.read_csv("data/benchmark/all_results_v1.csv")
o1 = r1[r1.ticker.isna()]
print(o1.groupby(["level", "mode"])[
    ["average_precision", "roc_auc", "brier"]].mean().round(4).to_string())

sec("C2. v1 unified grid: model ranking")
print(o1.groupby("model")["average_precision"].agg(["mean", "std"]).round(4)
      .sort_values("mean", ascending=False).to_string())

sec("C3. v1 group level: dataset x mode")
g1 = o1[o1.level == "group"]
print(g1.groupby(["dataset", "mode"])[
    ["average_precision", "roc_auc"]].mean().round(4).unstack().to_string())

sec("C4. v1 ticker level: pooled, best cells")
t1 = o1[o1.level == "ticker"]
print(t1.groupby("model")["average_precision"].agg(["mean", "std"]).round(4)
      .sort_values("mean", ascending=False).to_string())

# D. v2 grid (main + partial)
sec("D1. v2 grid (current run + archived partial): mode x level")
list_of = ["data/benchmark/all_results.csv"]
try:
    r2 = pd.concat([pd.read_csv(f) for f in list_of], ignore_index=True)
    dedup = ["level", "dataset", "mode", "model", "fold"]
    o2 = r2[r2.ticker.isna()].drop_duplicates(dedup + ["ticker"], keep="last")
    o2 = o2[o2.ticker.isna()]
    print(o2.groupby(["level", "mode"])[
        ["average_precision", "roc_auc"]].mean().round(4).to_string())
    sec("D2. v2 grid: model ranking (mean AP, all cells)")
    print(o2.groupby("model")["average_precision"].agg(["mean", "std"]).round(4)
          .sort_values("mean", ascending=False).to_string())
    sec("D3. v2 group level: dataset x mode (mean AP)")
    print(o2[o2.level == "group"].groupby(["dataset", "mode"])["average_precision"]
          .mean().round(4).unstack().to_string())
    sec("D4. v2 ticker level: dataset breakdown (mean AP by dataset)")
    tk2 = r2[(r2.level == "ticker") & r2.ticker.notna()]
    if len(tk2):
        print(tk2.groupby(["dataset", "mode"])["average_precision"].mean().round(4)
              .unstack().to_string())
except Exception as e:
    print("ERR", e)

# E. Agents' independent runs
sec("E1. Agent A: cross_index_attention (group A)")
c = pd.read_csv("data/cross_index_forecasting/cross_index_walk_forward_metrics.csv")
print(c[c.ticker == "ALL"].groupby("model")[
    ["average_precision", "roc_auc", "precision", "recall", "alarm_rate"]].mean().round(4).to_string())
print("(per-ticker AP mean across folds):")
print(c[c.ticker != "ALL"].pivot_table(index="ticker", columns="model",
                                       values="average_precision").round(4).to_string())

sec("E2. Agent B: peer v1 (pooled GRU/Transformer, group A)")
p1 = pd.read_csv("data/forecasting/peer_index_deep_learning_metrics.csv")
print(p1.groupby("model")[
    ["average_precision", "roc_auc", "precision", "recall"]].mean().round(4).to_string())

sec("E3. Agent B: peer v2 (per-target + ensemble)")
p2 = pd.read_csv("data/forecasting/peer_index_deep_learning_v2_metrics.csv")
print(p2.groupby("model")[
    ["average_precision", "roc_auc", "precision", "recall", "f1"]].mean().round(4).to_string())
print("(per-target ensemble AP across folds):")
ens = p2[p2.model == "ensemble_top2"]
print(ens.pivot_table(index="target", columns="fold",
                      values="average_precision").round(4).to_string())

# F. Lead-time analysis
sec("F. qualitative lead-time (family AUC / lead / fire-share)")
lt = pd.read_csv("data/benchmark/qualitative/family_stats_dataset.csv")
print(lt.round(3).to_string())
