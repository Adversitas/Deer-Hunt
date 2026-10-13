"""Features computed from closes only. Row t uses information up to and including t."""

import numpy as np
import pandas as pd


def _rsi(close, window):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def build_features(close, horizons=(1, 5, 10, 20, 60), vol_window=20):
    """Return (features, target, forward_return).

    target[t] is 1 when the close rises from t to t+1, else 0. forward_return[t]
    is that log return; both are NaN on the last bar.
    """
    log_close = np.log(close)
    ret = log_close.diff()
    feats = {}

    for h in horizons:
        feats[f"mom_{h}"] = log_close.diff(h)

    vol = ret.rolling(vol_window).std()
    feats["vol"] = vol
    feats["vol_ratio"] = vol / ret.rolling(vol_window * 3).std()
    # Momentum scaled by volatility: a 2% move means more in a quiet market.
    feats["mom_20_z"] = log_close.diff(20) / (vol * np.sqrt(20))

    for w in (20, 50, 200):
        feats[f"dist_ma_{w}"] = log_close - log_close.rolling(w).mean()
    feats["ma_cross"] = log_close.rolling(20).mean() - log_close.rolling(50).mean()

    feats["rsi_14"] = (_rsi(close, 14) - 50) / 50
    hi = close.rolling(60).max()
    lo = close.rolling(60).min()
    feats["range_pos_60"] = (close - lo) / (hi - lo).replace(0, np.nan) - 0.5

    X = pd.DataFrame(feats, index=close.index)
    fwd = ret.shift(-1).rename("fwd_ret")
    y = (fwd > 0).astype(float).where(fwd.notna()).rename("target")
    return X, y, fwd
