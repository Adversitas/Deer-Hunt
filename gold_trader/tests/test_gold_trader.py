import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from gold_trader import BacktestConfig, align_macro, load_etf_holdings, LogisticModel, build_features, summarize, walk_forward


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


def flow_driven(n=1500, seed=3):
    """Gold that follows ETF flows with a two-day delay, so gold's own past says nothing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    flow = 0.002 * rng.standard_normal(n)
    holdings = pd.Series(1000 * np.exp(np.cumsum(flow)), index=idx)
    gold_r = 0.008 * rng.standard_normal(n)
    # With the default 1-bar lag, bar t sees flow[t-1]; it moves gold on t -> t+1.
    gold_r[2:] += 3.0 * flow[:-2]
    close = pd.Series(1500 * np.exp(np.cumsum(gold_r)), index=idx)
    return close, holdings.rename("etf_holdings").to_frame()


SPDR_SAMPLE = """SPDR GOLD TRUST - Historical Archive
Disclaimer: past performance is not indicative of future results.

Date,GLD Close,LBMA Gold Price,NAV per GLD in Gold (ozs),Daily Share Volume,Total NAV Ounces in the Trust,Total NAV Tonnes in the Trust
02-Jan-2024, 190.12, 2064.40, 0.0920, 6000000, "27,965,000.0", 869.83
03-Jan-2024, HOLIDAY, HOLIDAY, HOLIDAY, HOLIDAY, HOLIDAY, HOLIDAY
04-Jan-2024, 189.50, 2050.10, 0.0920, 5500000, "28,012,000.0", " 871.29"
"""


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

    def test_etf_flow_features_add_edge(self):
        close, holdings = flow_driven()
        cfg = BacktestConfig(min_train=500, retrain_every=50, allow_short=True)
        without = summarize(walk_forward(close, cfg))["strategy"]
        with_flows = summarize(walk_forward(close, cfg, holdings))["strategy"]
        self.assertGreater(with_flows["hit_rate"], 0.55)
        self.assertGreater(with_flows["sharpe"], without["sharpe"] + 1.0)

    def test_etf_flows_have_no_lookahead(self):
        close, holdings = flow_driven()
        shocked = holdings.copy()
        shocked.iloc[1200:] *= 0.5
        a = walk_forward(close, CFG, holdings)
        b = walk_forward(close, CFG, shocked)
        cut = a.index[a.index < close.index[1200]]
        np.testing.assert_allclose(a.loc[cut, "proba"], b.loc[cut, "proba"])

    def test_load_spdr_archive_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gld.csv")
            with open(path, "w") as f:
                f.write(SPDR_SAMPLE)
            s = load_etf_holdings(path)
        self.assertEqual(s.name, "etf_holdings")
        self.assertEqual(list(s.index), list(pd.to_datetime(["2024-01-02", "2024-01-04"])))
        np.testing.assert_allclose(s.to_numpy(), [869.83, 871.29])

    def test_model_recovers_linear_signal(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 3))
        y = (X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(2000) > 0).astype(float)
        p = LogisticModel().fit(X, y).predict_proba(X)
        self.assertGreater(((p > 0.5) == y).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
