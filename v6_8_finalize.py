"""Create machine-readable V6 completion and result summaries."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"


def main():
    rows = []
    checks = [
        ("phase0_inventory", ["label_inventory.csv", "assets_alive_by_year.csv"]),
        ("phase1_semivar", ["semivar.parquet", "panel.parquet", "registry_v6.csv"]),
        ("phase2_labels", ["label_prevalence.csv", "labels_overlap.csv"]),
        ("phase3_blocked_e_sev", ["sweep_e_sev.csv", "sweep_e_sev_summary.csv"]),
        ("phase3_scheme_w", ["scheme_w_results.csv"]),
    ]
    for phase, files in checks:
        for file in files:
            p = V6 / file
            rows.append({"phase": phase, "file": file, "exists": p.exists(),
                         "bytes": p.stat().st_size if p.exists() else 0})
    pd.DataFrame(rows).to_csv(V6 / "completion_status.csv", index=False)
    if (V6 / "scheme_w_results.csv").exists():
        w = pd.read_csv(V6 / "scheme_w_results.csv")
        w.groupby(["N", "direction", "feature_set", "engine"]).agg(
            rows=("era", "count"), mean_excess=("excess_event_recall", "mean"),
            mean_ap=("ap", "mean"), mean_auc=("auc", "mean"),
            off_budget=("off_budget", "sum")).reset_index().to_csv(
                V6 / "scheme_w_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
