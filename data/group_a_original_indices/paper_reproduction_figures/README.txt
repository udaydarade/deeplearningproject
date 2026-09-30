Exact paper-style reproduction with normal-day comparison
The paper did not predict a crash with a binary TDA threshold.
It used w=50 (and w=100 robustness) TDA point-cloud windows, then a separate 500-trading-day rolling analysis of the norm series.
The three paper diagnostics were variance, average low-frequency spectral density, and lag-1 ACF, evaluated over the 250 trading days before fixed event dates.
The paper's quantitative claim was a rising low-frequency spectral-density trend, tested with Mann-Kendall; it reported tau=0.89 and 1.00 for the two original cases.
The plots compare these curves against 5th–95th percentile normal-day bands computed outside ±250 trading days around all reference stress dates.
The paper does not state the exact low-frequency cutoff, so this implementation uses the first five nonzero periodogram bins and documents that limitation.

2000-03-10: 250-day low-frequency-power Kendall tau=-0.601; variance tau=-0.223; ACF tau=-0.771
2008-09-15: 250-day low-frequency-power Kendall tau=0.691; variance tau=0.888; ACF tau=0.421