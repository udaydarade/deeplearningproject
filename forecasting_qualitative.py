"""Create event-aligned, four-index TDA/CSD diagnostic panels."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = ROOT / "data" / "forecasting"
TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
EVENTS = {
    "Dot-com bear market": "2000-03-10",
    "Lehman / global financial crisis": "2008-09-15",
    "US debt-ceiling stress": "2011-08-05",
    "COVID-19 selloff": "2020-02-19",
    "2022 bear market": "2022-01-03",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(OUT / "modeling_panel.csv", parse_dates=["date"])
    prices = pd.read_csv(GROUP / "prices.csv", parse_dates=["date"]).set_index("date")
    for title, event in EVENTS.items():
        event_ts = pd.Timestamp(event)
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        for ax, ticker in zip(axes, TICKERS):
            g = panel[panel.ticker.eq(ticker)].sort_values("date").set_index("date")
            loc = g.index.searchsorted(event_ts)
            start = max(0, loc - 180)
            stop = min(len(g), loc + 61)
            s = g.iloc[start:stop]
            if s.empty:
                continue
            baseline = g.iloc[max(0, loc - 180):loc]
            for col, label, color in [
                ("tda_landscape_l2_w50", "log10 TDA L2 (w=50)", "tab:blue"),
                ("csd_variance_w50", "log10 return variance (w=50)", "tab:orange"),
                ("csd_ac1_w50", "Lag-1 ACF (w=50)", "tab:green"),
            ]:
                values = np.log10(s[col].clip(lower=1e-16)) if col != "csd_ac1_w50" else s[col]
                base = np.log10(baseline[col].clip(lower=1e-16)) if col != "csd_ac1_w50" else baseline[col]
                mu, sd = base.mean(), base.std()
                z = (values - mu) / (sd if sd and np.isfinite(sd) else 1.0)
                z = z.clip(-8, 8)
                ax.plot(s.index, z, lw=1.1, label=label, color=color)
            p = prices.loc[s.index[0]:s.index[-1], ticker]
            dd = (p / p.cummax() - 1) * 100
            ax2 = ax.twinx()
            ax2.plot(dd.index, dd, color="black", lw=1.0, alpha=0.55, label="Price drawdown")
            ax2.set_ylabel("Drawdown (%)")
            ax.axvline(event_ts, color="crimson", ls="--", lw=1.0)
            ax.set_ylabel(f"{ticker}\nfeature z-score")
            ax.grid(alpha=0.2)
            if ticker == TICKERS[0]:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, ncol=4, fontsize=8, loc="upper left")
        axes[-1].set_xlabel("Date (180 trading sessions before to 60 after event)")
        fig.suptitle(f"{title} ({event}): TDA and classical indicators across four indices")
        fig.tight_layout()
        slug = title.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(OUT / f"event_{slug}.png", dpi=150)
        plt.close(fig)
    print(f"Wrote {len(EVENTS)} event panels to {OUT}")


if __name__ == "__main__":
    main()
