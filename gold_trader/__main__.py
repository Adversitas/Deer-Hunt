"""CLI: python -m gold_trader [--csv FILE | --ticker GC=F] [options]"""

import argparse

from .backtest import BacktestConfig, summarize, walk_forward
from .data import download_yahoo, load_csv
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
    p.add_argument("--out", help="write per-bar results to this CSV")
    p.add_argument("--plot", help="save an equity curve PNG (needs matplotlib)")
    args = p.parse_args(argv)

    close = load_csv(args.csv) if args.csv else download_yahoo(args.ticker, start=args.start)
    cfg = BacktestConfig(
        min_train=args.min_train,
        retrain_every=args.retrain_every,
        band=args.band,
        allow_short=args.allow_short,
        target_vol=args.target_vol or None,
        max_leverage=args.max_leverage,
        cost_bps=args.cost_bps,
    )
    res = walk_forward(close, cfg)

    print(f"Data: {close.index[0].date()} -> {close.index[-1].date()} ({len(close)} bars)")
    print(f"Out-of-sample: {res.index[0].date()} -> {res.index[-1].date()} ({len(res)} bars)\n")
    print(summarize(res).to_string(float_format=lambda v: f"{v:,.3f}"))

    # Signal for the next bar, from a model fit on everything known today.
    X, y, _ = build_features(close)
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
