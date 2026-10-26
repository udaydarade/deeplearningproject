from pathlib import Path
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"
rows = [
    {"project": "V6", "component": "blocked E-severity", "status": "complete",
     "output": "sweep_e_sev.csv"},
    {"project": "V6", "component": "real-time Scheme-W", "status": "complete",
     "output": "scheme_w_results.csv"},
    {"project": "V6", "component": "N=5 cross-transfer", "status": "running",
     "output": "scheme_w_n5_scores.parquet; t2_cross_transfer_n5.csv"},
    {"project": "V6", "component": "external universe extension", "status": "not_run",
     "output": "none"},
    {"project": "V6", "component": "S1-S4 strategies", "status": "not_run",
     "output": "none"},
    {"project": "V5", "component": "E2 paired ablations", "status": "running",
     "output": "data/v5/e2_results.csv"},
]
pd.DataFrame(rows).to_csv(V6 / "current_status.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
