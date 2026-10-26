"""Offline V6 direction comparison from completed Scheme-W result cells."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"


def main():
    w = pd.read_csv(V6 / "scheme_w_results.csv")
    native = (w[w.feature_set == "BEST_SET_SEMIVAR"]
              .groupby(["N", "direction", "engine"])
              .agg(eras=("era", "count"), mean_excess=("excess_event_recall", "mean"),
                   mean_ap=("ap", "mean"), mean_auc=("auc", "mean"),
                   off_budget=("off_budget", "sum"))
              .reset_index())
    native.to_csv(V6 / "t1_native_direction.csv", index=False)
    pivot = native.pivot_table(index=["N", "engine"], columns="direction",
                               values=["mean_excess", "mean_ap", "mean_auc"]).reset_index()
    pivot.to_csv(V6 / "t1_down_up_comparison.csv", index=False)
    status = pd.DataFrame([
        {"item": "universe_extension", "status": "not_run",
         "reason": "Existing V3 universe was retained; no external downloads were needed for the completed causal tests."},
        {"item": "scheme_B_e_sev", "status": "complete", "reason": "110 cells"},
        {"item": "scheme_W_e_sev_and_e_bin", "status": "complete", "reason": "168 cells; N=5,15,30; both directions; B3 and SEMIVAR"},
        {"item": "T1_native_down_up", "status": "complete", "reason": "Aggregated from Scheme-W saved results"},
        {"item": "T2_cross_transfer", "status": "not_run", "reason": "Per-row cross-direction scores were not persisted by the time-boxed W runner"},
        {"item": "T3_conditional_direction", "status": "not_run", "reason": "Requires a separate direction-conditioned fit"},
        {"item": "S1_to_S4_strategies", "status": "not_run", "reason": "No saved component score histories; no unsupported claim made"},
        {"item": "S5_semivar_value", "status": "complete", "reason": "B3 versus BEST_SET_SEMIVAR paired across completed W cells"},
    ])
    status.to_csv(V6 / "v6_remaining_status.csv", index=False)
    print(native.to_string(index=False))
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
