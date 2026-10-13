import unittest

import numpy as np
import pandas as pd

from gold_trader import BacktestConfig, align_macro, LogisticModel, build_features, summarize, walk_forward


def random_walk(n=1500, seed=0, drift=0.0002):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(1500 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n))), index=idx)


def trending(n=1500, seed=1):
    """Returns with positive autocorrelation, which momentum features can exploit."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.3 * r[t - 1] + 0.01 * rng.standard_normal()
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(1500 * np.exp(np.cumsum(r)), index=idx)


def macro_driven(n=1500, seed=2):
    """Gold whose next-day return falls after real rates or the dollar rise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    d_rr = 0.05 * rng.standard_normal(n)
    d_usd = 0.004 * rng.standard_normal(n)
    gold_r = 0.005 * rng.standard_normal(n)
    # Today's macro moves drive tomorrow's gold return.
    gold_r[1:] += -0.08 * d_rr[:-1] - 1.0 * d_usd[:-1]
    close = pd.Series(1500 * np.exp(np.cumsum(gold_r)), index=idx)
    macro = pd.DataFrame(
        {"real_rate": 1 + np.cumsum(d_rr), "dollar": 100 * np.exp(np.cumsum(d_usd))}, index=idx
    )
    return close, macro


CFG = BacktestConfig(min_train=500, retrain_every=50)


class TestGoldTrader(unittest.TestCase):
    def test_features_do_not_use_future(self):
        close = random_walk()
        X_full, _, _ = build_features(close)
        X_cut, _, _ = build_features(close.iloc[:1000])
        pd.testing.assert_frame_equal(X_full.iloc[:1000], X_cut)

    def test_walk_forward_has_no_lookahead(self):
        close = random_walk()
        shocked = close.copy()
        shocked.iloc[1200:] *= 3  # change the future only
        a = walk_forward(close, CFG)
        b = walk_forward(shocked, CFG)
        cut = a.index[a.index < close.index[1199]]
        np.testing.assert_allclose(a.loc[cut, "proba"], b.loc[cut, "proba"])
        np.testing.assert_allclose(a.loc[cut, "position"], b.loc[cut, "position"])

    def test_learns_exploitable_pattern(self):
        res = walk_forward(trending(), BacktestConfig(min_train=500, retrain_every=50, allow_short=True))
        stats = summarize(res)["strategy"]
        self.assertGreater(stats["hit_rate"], 0.55)
        self.assertGreater(stats["sharpe"], 1.0)

    def test_positions_respect_limits(self):
        res = walk_forward(random_walk(), CFG)
        self.assertTrue((res["position"] >= 0).all())  # long-only by default
        self.assertTrue((res["position"] <= CFG.max_leverage + 1e-12).all())

    def test_costs_reduce_returns(self):
        close = random_walk()
        free = walk_forward(close, BacktestConfig(min_train=500, retrain_every=50, cost_bps=0))
        costly = walk_forward(close, BacktestConfig(min_train=500, retrain_every=50, cost_bps=20))
        self.assertLess(costly["strategy_ret"].sum(), free["strategy_ret"].sum())

    def test_macro_features_add_edge(self):
        close, macro = macro_driven()
        # Lag 0: the model sees today's macro close, which drives tomorrow's gold move.
        cfg = BacktestConfig(min_train=500, retrain_every=50, allow_short=True, macro_lag=0)
        without = summarize(walk_forward(close, cfg))["strategy"]
        with_macro = summarize(walk_forward(close, cfg, macro))["strategy"]
        self.assertGreater(with_macro["hit_rate"], 0.55)
        self.assertGreater(with_macro["sharpe"], without["sharpe"] + 1.0)

    def test_macro_has_no_lookahead(self):
        close, macro = macro_driven()
        shocked = macro.copy()
        shocked.iloc[1200:] *= 2
        a = walk_forward(close, CFG, macro)
        b = walk_forward(close, CFG, shocked)
        cut = a.index[a.index < close.index[1200]]
        np.testing.assert_allclose(a.loc[cut, "proba"], b.loc[cut, "proba"])

    def test_align_macro_lags_and_fills(self):
        gold_idx = pd.bdate_range("2020-01-01", periods=5)
        macro = pd.DataFrame({"dollar": [1.0, 3.0]}, index=gold_idx[[0, 2]])
        aligned = align_macro(macro, gold_idx, lag=1)["dollar"]
        # Forward-filled to [1, 1, 3, 3, 3], then delayed one bar.
        np.testing.assert_array_equal(aligned.to_numpy(), [np.nan, 1, 1, 3, 3])

    def test_model_recovers_linear_signal(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 3))
        y = (X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(2000) > 0).astype(float)
        p = LogisticModel().fit(X, y).predict_proba(X)
        self.assertGreater(((p > 0.5) == y).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
