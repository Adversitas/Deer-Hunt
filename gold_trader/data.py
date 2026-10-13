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


# FRED series: 10y TIPS yield (real rate, %) and the Fed's broad trade-weighted dollar index.
FRED_SERIES = {"real_rate": "DFII10", "dollar": "DTWEXBGS"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def download_fred(series_id, start=None):
    """Download one FRED series as a date-indexed Series (no API key needed)."""
    df = pd.read_csv(FRED_URL.format(series_id))
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),  # "." marks holidays
        index=pd.to_datetime(df.iloc[:, 0]),
        name=series_id,
    ).dropna()
    return s[s.index >= pd.Timestamp(start)] if start else s


def download_macro(start=None):
    """Real rate and dollar index from FRED as a DataFrame (columns: real_rate, dollar)."""
    return pd.DataFrame({name: download_fred(sid, start) for name, sid in FRED_SERIES.items()})


def load_macro_csv(path):
    """Load a CSV with a date column plus `real_rate` and/or `dollar` columns."""
    df = pd.read_csv(path)
    date_col = _pick(df.columns, DATE_COLUMNS) or df.columns[0]
    cols = {c.lower().strip(): c for c in df.columns}
    wanted = [name for name in FRED_SERIES if name in cols]
    if not wanted:
        raise ValueError(f"{path} needs a real_rate and/or dollar column: {list(df.columns)}")
    out = df[[cols[name] for name in wanted]].apply(pd.to_numeric, errors="coerce")
    out.columns = wanted
    out.index = pd.to_datetime(df[date_col])
    return out.sort_index()


def align_macro(macro, index, lag=1):
    """Put macro data on the gold price index.

    Each gold bar takes the latest macro value on or before its date, then the
    whole frame is shifted `lag` bars so a bar only sees macro data that was
    already published (FRED values often appear a day or more after the fact).
    """
    macro = macro[~macro.index.duplicated(keep="last")].sort_index()
    on_index = macro.reindex(macro.index.union(index)).ffill().reindex(index)
    return on_index.shift(lag)


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
