"""CLI: python -m gold_trader [--csv FILE | --ticker GC=F] [options]"""

import argparse

import pandas as pd

from .backtest import BacktestConfig, summarize, walk_forward
from .data import align_macro, download_macro, download_yahoo, load_csv, load_etf_holdings, load_macro_csv
from .features import build_features
from .model import LogisticModel


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest a walk-forward gold trading model.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="CSV with a date column and a close/price column")
    src.add_argument("--ticker", default="GC=F", help="Yahoo ticker (default: GC=F gold futures)")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--min-train", type=int, default=BacktestConfig.min_train)
    p.add_argument("--retrain-every", type=int, default=BacktestConfig.retrain_every)
    p.add_argument("--band", type=float, default=BacktestConfig.band)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--target-vol", type=float, default=BacktestConfig.target_vol,
                   help="annualised vol target; 0 disables vol targeting")
    p.add_argument("--max-leverage", type=float, default=BacktestConfig.max_leverage)
    p.add_argument("--cost-bps", type=float, default=BacktestConfig.cost_bps)
    p.add_argument("--macro", choices=("fred", "none"), default="fred",
                   help="real-rate and dollar-index features: download from FRED (default) or skip")
    p.add_argument("--macro-csv", help="CSV with date, real_rate, dollar columns (instead of FRED)")
    p.add_argument("--etf", choices=("gld", "none"), default="gld",
                   help="ETF-flow features from SPDR GLD holdings (default) or skip")
    p.add_argument("--etf-csv", help="CSV of ETF gold holdings (date + tonnes/holdings column)")
    p.add_argument("--macro-lag", type=int, default=BacktestConfig.macro_lag,
                   help="bars to delay macro data, for publication lag")
    p.add_argument("--out", help="write per-bar results to this CSV")
    p.add_argument("--plot", help="save an equity curve PNG (needs matplotlib)")
    args = p.parse_args(argv)

    close = load_csv(args.csv) if args.csv else download_yahoo(args.ticker, start=args.start)
    macro = None
    if args.macro_csv:
        macro = load_macro_csv(args.macro_csv)
    elif args.macro == "fred":
        try:
            macro = download_macro(start=close.index[0] - pd.Timedelta(days=400))
        except Exception as e:
            p.error(f"could not download FRED data ({e}); use --macro-csv FILE or --macro none")

    holdings = None
    if args.etf_csv:
        holdings = load_etf_holdings(args.etf_csv)
    elif args.etf == "gld" and not (macro is not None and "etf_holdings" in macro):
        try:
            holdings = load_etf_holdings()
        except Exception as e:
            p.error(f"could not download GLD holdings ({e}); use --etf-csv FILE or --etf none")
    if holdings is not None:
        macro = holdings.to_frame() if macro is None else (
            macro.drop(columns="etf_holdings", errors="ignore").join(holdings, how="outer"))

    cfg = BacktestConfig(
        min_train=args.min_train,
        retrain_every=args.retrain_every,
        band=args.band,
        allow_short=args.allow_short,
        target_vol=args.target_vol or None,
        max_leverage=args.max_leverage,
        cost_bps=args.cost_bps,
        macro_lag=args.macro_lag,
    )
    res = walk_forward(close, cfg, macro)

    print(f"Data: {close.index[0].date()} -> {close.index[-1].date()} ({len(close)} bars)")
    if macro is not None:
        print(f"External features: {', '.join(macro.columns)} (lagged {cfg.macro_lag} bar(s))")
    print(f"Out-of-sample: {res.index[0].date()} -> {res.index[-1].date()} ({len(res)} bars)\n")
    print(summarize(res).to_string(float_format=lambda v: f"{v:,.3f}"))

    # Signal for the next bar, from a model fit on everything known today.
    aligned = None if macro is None else align_macro(macro, close.index, cfg.macro_lag)
    X, y, _ = build_features(close, aligned)
    X = X.dropna()
    known = y.loc[X.index].notna().to_numpy()
    model = LogisticModel(l2=cfg.l2).fit(X[known], y.loc[X.index][known])
    prob = model.predict_proba(X.iloc[[-1]])[0]
    action = "LONG" if prob > 0.5 + cfg.band else ("SHORT" if cfg.allow_short and prob < 0.5 - cfg.band else "FLAT")
    print(f"\nLatest bar {X.index[-1].date()}: P(up next bar) = {prob:.3f} -> {action}")

    if args.out:
        res.to_csv(args.out)
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        (1 + res["strategy_ret"]).cumprod().plot(ax=ax, label="model")
        (1 + res["buyhold_ret"]).cumprod().plot(ax=ax, label="buy & hold")
        ax.set_yscale("log")
        ax.set_title("Gold: walk-forward model vs buy & hold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)


if __name__ == "__main__":
    main()
