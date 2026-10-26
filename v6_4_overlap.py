"""V6 label overlap and BASE agreement diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3, V6 = ROOT / "data" / "v3", ROOT / "data" / "v6"


def main():
    panel = pd.read_parquet(V3 / "panel.parquet",
                            columns=["date", "asset", "asset_class", "y_BASE"])
    p = pd.concat([pd.read_parquet(f, columns=["date", "asset", "u_down_N15",
                                                "u_up_N15", "y_down_N15_p2",
                                                "y_up_N15_p2", "y_move_N15_p2"])
                   for f in (V6 / "labels").glob("*.parquet")], ignore_index=True)
    x = panel.merge(p, on=["date", "asset"], how="inner")
    valid = (x.y_down_N15_p2.notna() & x.y_up_N15_p2.notna() & x.y_BASE.notna())
    down, up = x.y_down_N15_p2.eq(1), x.y_up_N15_p2.eq(1)
    rows = [{
        "valid_rows": int(valid.sum()),
        "p_up_given_down": float(up[valid & down].mean()) if (valid & down).any() else np.nan,
        "p_down_given_up": float(down[valid & up].mean()) if (valid & up).any() else np.nan,
        "rank_corr": float(x.loc[valid, "u_down_N15"].corr(x.loc[valid, "u_up_N15"])),
        "both_share": float((down[valid] & up[valid]).mean()) if valid.any() else np.nan,
        "base_mismatch_share": float((x.loc[valid, "y_BASE"] !=
                                       x.loc[valid, "y_down_N15_p2"]).mean()) if valid.any() else np.nan,
        "shared_rows": len(x),
    }]
    pd.DataFrame(rows).to_csv(V6 / "labels_overlap.csv", index=False)
    x = x.assign(valid=valid, down=down, up=up)
    by_class = x.groupby("asset_class").apply(lambda g: pd.Series({
        "rows": len(g), "valid_rows": int(g.valid.sum()),
        "down": float(g.loc[g.valid, "down"].mean()) if g.valid.any() else np.nan,
        "up": float(g.loc[g.valid, "up"].mean()) if g.valid.any() else np.nan,
        "both": float(g.loc[g.valid, "y_move_N15_p2"].eq(1).mean()) if g.valid.any() else np.nan,
    }), include_groups=False)
    by_class.to_csv(V6 / "labels_overlap_by_class.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    print(by_class.to_string())


if __name__ == "__main__":
    main()
