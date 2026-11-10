"""Long price histories for the mechanism lab, saved under data/ after the first download, and the split that keeps a final exam sealed.

DESIGN markets and years are the only ones a design decision may look at. SEALED markets, and every market's years after DESIGN_END, are
opened once, at the end (see lab_research.sealed_exam).
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# name -> (Yahoo symbol, first date to keep). Several free histories have no real High/Low in their early years (High == Low on a third to all
# of the days: the S&P before the mid-1960s, the Nikkei before 1988, the German index before 1994, the Nasdaq Composite before 1985), and the
# gold and silver futures series are patchy until about 2008. Those years are dropped, and gold and silver use the ETFs (GLD from late 2004,
# SLV from 2006: clean OHLC, so a stop level or an ATR means what it should; the ETF fee drag is about 0.4% a year).
MARKETS = {
    "SP500": ("^GSPC", "1965-01-01"),
    "Nasdaq": ("^NDX", "1985-10-01"),
    "Dow": ("^DJI", "1992-01-02"),
    "Russell2000": ("^RUT", "1987-09-10"),
    "Ger40": ("^GDAXI", "1993-12-01"),
    "Nikkei225": ("^N225", "1988-06-01"),
    "Gold": ("GLD", "2004-11-18"),
    "Silver": ("SLV", "2006-04-28"),
    "Copper": ("HG=F", "2000-08-30"),
    "Bitcoin": ("BTC-USD", "2014-09-17"),
}
DESIGN_MARKETS = ["SP500", "Nasdaq", "Gold"]
SEALED_MARKETS = ["Dow", "Russell2000", "Ger40", "Nikkei225", "Silver", "Copper", "Bitcoin"]
DESIGN_END = "2018-12-31"     # the years after this are sealed for every market


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    return df


def load_market(name: str, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLC for the named market, cleaned like the engine's download (High/Low widened to contain Open and Close)."""
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    if not refresh and os.path.exists(path):
        return pd.read_pickle(path)
    import yfinance as yf
    symbol, start = MARKETS[name]
    df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = _clean(df)
    df.attrs["asset"] = name
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_pickle(path)
    return df


def design_slice(df: pd.DataFrame) -> pd.DataFrame:
    """The market's design years only (up to DESIGN_END)."""
    return df.loc[:DESIGN_END]


def sealed_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[pd.Timestamp(DESIGN_END) + pd.Timedelta(days=1):]


def describe() -> pd.DataFrame:
    rows = []
    for name in MARKETS:
        try:
            df = load_market(name)
        except Exception as exc:   # noqa: BLE001 -- report and carry on
            rows.append({"market": name, "error": str(exc)[:60]})
            continue
        flat = float((df["High"] == df["Low"]).mean() * 100)
        rows.append({"market": name, "from": df.index[0].date(), "to": df.index[-1].date(), "bars": len(df), "flat_hl_%": round(flat, 1),
                     "role": "design" if name in DESIGN_MARKETS else ("sealed" if name in SEALED_MARKETS else "reference")})
    return pd.DataFrame(rows)
