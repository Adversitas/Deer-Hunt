"""Walk-forward backtest: the model only ever sees data available at decision time."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import align_macro, periods_per_year
from .features import build_features
from .model import LogisticModel


@dataclass
class BacktestConfig:
    min_train: int = 750          # bars before the first trade (~3 years daily)
    retrain_every: int = 21       # refit roughly monthly on daily data
    train_window: int | None = None  # None = expanding window
    band: float = 0.02            # trade only when |p - 0.5| > band
    allow_short: bool = False     # gold has a long-run upward drift; shorting is optional
    target_vol: float | None = 0.15  # annualised vol target; None = fixed 1x size
    max_leverage: float = 1.0
    cost_bps: float = 2.0         # cost per unit of turnover, in basis points
    l2: float = 0.01
    macro_lag: int = 1            # bars to delay macro data (publication lag)


def walk_forward(close, cfg=BacktestConfig(), macro=None):
    """Return a DataFrame with probability, position and strategy returns per bar.

    `macro` is an optional raw DataFrame (real_rate / dollar columns, any dates);
    it is aligned to the gold bars and lagged by cfg.macro_lag here.
    """
    if macro is not None:
        macro = align_macro(macro, close.index, cfg.macro_lag)
    X, y, fwd = build_features(close, macro)
    valid = X.notna().all(axis=1)
    X, y, fwd = X[valid], y[valid], fwd[valid]
    n = len(X)
    if n <= cfg.min_train + 1:
        raise ValueError(f"Not enough data: {n} usable bars, min_train={cfg.min_train}")

    ppy = periods_per_year(close.index)
    Xv, yv = X.to_numpy(), y.to_numpy()
    proba = np.full(n, np.nan)
    model = None

    for t in range(cfg.min_train, n):
        if model is None or (t - cfg.min_train) % cfg.retrain_every == 0:
            # Rows < t have known targets at time t (target[t-1] resolves at bar t).
            start = 0 if cfg.train_window is None else max(0, t - cfg.train_window)
            model = LogisticModel(l2=cfg.l2).fit(Xv[start:t], yv[start:t])
        proba[t] = model.predict_proba(Xv[t : t + 1])[0]

    res = pd.DataFrame({"proba": proba, "fwd_ret": fwd.to_numpy()}, index=X.index)
    res = res.iloc[cfg.min_train :]

    signal = np.where(res["proba"] > 0.5 + cfg.band, 1.0, 0.0)
    if cfg.allow_short:
        signal = np.where(res["proba"] < 0.5 - cfg.band, -1.0, signal)

    size = pd.Series(1.0, index=res.index)
    if cfg.target_vol is not None:
        realised = X["vol"].loc[res.index] * np.sqrt(ppy)
        size = (cfg.target_vol / realised).clip(upper=cfg.max_leverage)

    res["position"] = signal * size.to_numpy()
    turnover = res["position"].diff().abs().fillna(res["position"].abs())
    res["cost"] = turnover * cfg.cost_bps / 1e4
    # Position set at the close of t earns the return from t to t+1.
    res["strategy_ret"] = res["position"] * (np.exp(res["fwd_ret"]) - 1) - res["cost"]
    res["buyhold_ret"] = np.exp(res["fwd_ret"]) - 1
    res = res.dropna(subset=["fwd_ret"])
    res.attrs["periods_per_year"] = ppy
    return res


def _stats(returns, ppy):
    equity = (1 + returns).cumprod()
    years = len(returns) / ppy
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(ppy)
    sharpe = returns.mean() / returns.std() * np.sqrt(ppy) if returns.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def summarize(res):
    """Metrics for the strategy next to buy-and-hold over the same period."""
    ppy = res.attrs.get("periods_per_year", 252)
    out = pd.DataFrame(
        {
            "strategy": _stats(res["strategy_ret"], ppy),
            "buy_and_hold": _stats(res["buyhold_ret"], ppy),
        }
    )
    in_market = res["position"] != 0
    hits = np.sign(res["position"][in_market]) == np.sign(res["fwd_ret"][in_market])
    extra = pd.DataFrame(
        {
            "strategy": {
                "time_in_market": in_market.mean(),
                "hit_rate": hits.mean() if in_market.any() else np.nan,
                "trades": int((res["position"].diff().fillna(res["position"]) != 0).sum()),
                "avg_leverage": res["position"].abs()[in_market].mean(),
            }
        }
    )
    return pd.concat([out, extra])
