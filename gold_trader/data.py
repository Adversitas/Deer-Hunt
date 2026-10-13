"""Price loading. Everything downstream only needs a date-indexed close series."""

import pandas as pd

DATE_COLUMNS = ("date", "datetime", "time", "timestamp")
CLOSE_COLUMNS = ("adj close", "adj_close", "close", "price", "value")


def _pick(columns, candidates):
    lower = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def load_csv(path):
    """Load a CSV with a date column and a close/price column into a Series."""
    df = pd.read_csv(path)
    date_col = _pick(df.columns, DATE_COLUMNS) or df.columns[0]
    close_col = _pick(df.columns, CLOSE_COLUMNS)
    if close_col is None:
        raise ValueError(f"No close/price column found in {path}: {list(df.columns)}")
    prices = pd.Series(
        pd.to_numeric(df[close_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name="close",
    )
    return _clean(prices)


def download_yahoo(ticker="GC=F", start="2005-01-01", end=None):
    """Download daily closes from Yahoo Finance (needs `pip install yfinance`)."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return _clean(close.rename("close"))


def _clean(prices):
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.dropna()
    prices = prices[prices > 0]
    if len(prices) < 100:
        raise ValueError(f"Need at least 100 prices, got {len(prices)}")
    return prices


def periods_per_year(index):
    """Infer bar frequency (252 for daily, 52 weekly, 12 monthly)."""
    days = pd.Series(index).diff().dt.days.median()
    if days <= 1.5:
        return 252
    if days <= 8:
        return 52
    return 12
