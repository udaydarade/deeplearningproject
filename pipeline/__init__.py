"""Unified TDA-vs-classical crash-prediction benchmark package.

Modules
-------
datasets   : registry + loaders for the seven market datasets
labels     : objective forward-min crash labels with calibration-only thresholds
features   : classical CSD indicators and TDA features (joint + per-ticker)
models     : deep sequence-model zoo and shared trainers
benchmark  : leakage-controlled walk-forward evaluation at two levels
"""

from . import datasets, labels, features, models, benchmark

__all__ = ["datasets", "labels", "features", "models", "benchmark"]
