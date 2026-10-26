import pandas as pd

pd.set_option("display.width", 250)

r = pd.read_csv("data/benchmark/all_results.csv")
o = r[r.ticker.isna()]
g = o[o.level == "group"]
t = o[o.level == "ticker"]

print("unique overall cells:", o.groupby(["level", "dataset", "mode", "model", "fold"]).ngroups)

print()
print("=== v2 FINAL: mode x level (mean AP / AUC) ===")
print(o.groupby(["level", "mode"])[["average_precision", "roc_auc"]].mean().round(4).to_string())

print()
print("=== v2 FINAL: model ranking (mean AP across all group cells) ===")
print(g.groupby("model")["average_precision"].agg(["mean", "std"]).round(4)
      .sort_values("mean", ascending=False).to_string())

print()
print("=== v2 FINAL: dataset x mode, group level (mean AP) ===")
print(g.groupby(["dataset", "mode"])["average_precision"].mean().round(4).unstack().to_string())

print()
print("=== v2 FINAL: per-dataset best (mode, model, AP) ===")
for ds, sub in g.groupby("dataset"):
    m = sub.groupby(["mode", "model"])["average_precision"].mean()
    print(f"{ds}: {m.idxmax()} AP={m.max():.4f}")

print()
print("=== v2 ticker pooled: model ranking ===")
print(t.groupby("model")["average_precision"].agg(["mean", "std"]).round(4)
      .sort_values("mean", ascending=False).to_string())

print()
print("=== v2 ticker pooled: mode x model mean AP ===")
print(t.groupby(["mode"])["average_precision"].mean().round(4).sort_values(ascending=False).to_string())

print()
print("=== coverage check: cells present per dataset x level ===")
print(o.groupby(["level", "dataset"]).apply(
    lambda d: d.groupby(["mode", "model", "fold"]).ngroups).to_string())
