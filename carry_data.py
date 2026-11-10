"""Interest-rate and dividend inputs for the engine's carry model (turtle_dual_system.CARRY_MODE = "futures").

The short-term rate is the US 13-week Treasury bill yield (Yahoo symbol ^IRX, daily since 1960), saved to data/tbill_irx.csv the
first time it is downloaded so later runs, and runs without a connection, use the saved copy. Dividend yields are flat per-market
estimates; the S&P 500 has a better, time-varying series (dividend_yield_sp500) for research runs.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TBILL_FILE = os.path.join(DATA_DIR, "tbill_irx.csv")

# Rough long-run dividend yields of the equity indices the app trades (price indices: the dividends are not in the prices).
# Commodities and crypto pay none. Edit freely; the Fees panel scales all of these with one setting.
DEFAULT_DIVIDEND_YIELDS = {"SP500": 0.027, "Nasdaq": 0.009, "Dow": 0.025, "Russell2000": 0.013, "Ger40": 0.025, "Nikkei225": 0.017}


def tbill_series(refresh: bool = False):
    """Annual short-term rate as a fraction, indexed by date; None when there is neither a saved copy nor a connection."""
    if not refresh and os.path.exists(TBILL_FILE):
        s = pd.read_csv(TBILL_FILE, index_col=0, parse_dates=True).iloc[:, 0]
        return s.astype(float)
    try:
        import yfinance as yf
        df = yf.download("^IRX", start="1960-01-01", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = (df["Close"].dropna() / 100.0).astype(float)
        if len(s) < 1000:
            return None
        os.makedirs(DATA_DIR, exist_ok=True)
        s.rename("tbill").to_csv(TBILL_FILE)
        return s
    except Exception:
        return None


def dividend_yields(scale: float = 1.0, assets=None) -> dict:
    """{asset: annual yield as a fraction}, the defaults times `scale`, for the given assets (default: all that have one)."""
    names = DEFAULT_DIVIDEND_YIELDS if assets is None else [a for a in assets if a in DEFAULT_DIVIDEND_YIELDS]
    return {a: DEFAULT_DIVIDEND_YIELDS[a] * scale for a in names}


def dividend_yield_sp500(index: pd.DatetimeIndex) -> pd.Series:
    """Daily S&P 500 dividend yield (annual fraction): implied by the total-return index since 1988, 3.6% before that (the 1965-87 average)."""
    try:
        import yfinance as yf
        px = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=False)
        tr = yf.download("^SP500TR", start="1988-01-01", progress=False, auto_adjust=False)
        for d in (px, tr):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
        daily = (tr["Close"].pct_change() - px["Close"].pct_change().reindex(tr.index)).clip(0, 0.002) * 252
        daily = daily.rolling(60, min_periods=20).mean()
        return daily.reindex(index).ffill().fillna(0.036)
    except Exception:
        return pd.Series(0.027, index=index)
