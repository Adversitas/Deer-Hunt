"""Gold trading model: features, walk-forward classifier and backtester."""

from .data import align_macro, download_macro, download_yahoo, load_csv, load_macro_csv
from .features import build_features
from .model import LogisticModel
from .backtest import BacktestConfig, walk_forward, summarize

__all__ = [
    "load_csv",
    "load_macro_csv",
    "download_macro",
    "align_macro",
    "download_yahoo",
    "build_features",
    "LogisticModel",
    "BacktestConfig",
    "walk_forward",
    "summarize",
]
