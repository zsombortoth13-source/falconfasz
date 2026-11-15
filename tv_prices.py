"""TradingView daily price files for the Strategy Lab: one CSV per market in data_tradingview/, fetched with tradingview/tv_fetch.py (TradingView's own data feed, standard library only),
read back as the OHLC frames the engine uses. The Lab and the lean app can take their prices from these files instead of Yahoo Finance ("Price source" in the sidebar).

    python tv_prices.py                      # fetch every market (about 2 minutes)
    python tv_prices.py Gold Cocoa           # only these
    python tv_prices.py --desktop            # also copy the CSVs to a folder on the Desktop

The files are what TradingView's anonymous (logged-out) feed gives: DELAYED futures data, the longest daily history it has, no volume for most markets. Stock indices use the index, commodities and
the Nasdaq/Dow/Russell... see MARKETS: the instrument type matches what the Lab got from Yahoo (indices as indices, commodities as front-month continuous futures)."""
import csv
import importlib.util
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data_tradingview")
MANIFEST = os.path.join(DATA_DIR, "_sources.csv")
REMOVED_DIR = os.path.join(DATA_DIR, "removed_monthly_bars")      # the monthly bars cut out of each file (kept for reference, never read by the Lab)

# Strategy Lab market -> TradingView symbol (the same kind of instrument as the Lab's Yahoo ticker: indices as indices, commodities as front-month continuous futures)
MARKETS = {
    "SP500": "TVC:SPX",              # Yahoo ^GSPC
    "Gold": "COMEX:GC1!",            # GC=F
    "Silver": "COMEX:SI1!",          # SI=F
    "Nasdaq": "NASDAQ:NDX",          # ^NDX
    "Bitcoin": "BITSTAMP:BTCUSD",    # BTC-USD
    "Ger40": "TVC:DAX",              # ^GDAXI
    "Platinum": "NYMEX:PL1!",        # PL=F
    "Palladium": "NYMEX:PA1!",       # PA=F
    "Copper": "COMEX:HG1!",          # HG=F
    "Nikkei225": "TVC:NI225",        # ^N225
    "Russell2000": "TVC:RUT",        # ^RUT
    "Dow": "TVC:DJI",                # ^DJI
    "CrudeOil": "NYMEX:CL1!",        # CL=F
    "Corn": "CBOT:ZC1!",             # ZC=F
    "Cocoa": "ICEUS:CC1!",           # CC=F
}
EXTRAS = {"SP500_futures_ES1": "CME_MINI:ES1!"}      # fetched and exported too, not used as a Lab market

# The Lab uses a market only from this date (the files themselves keep everything). Bitcoin: TradingView's file starts in 2011 when it cost $5-$900, and a backtest with per-contract futures fees
# then needs hundreds of thousands of contracts for one position (one 2012 trade paid $281,000 in fees on a $24,000 position and sank the account). 2014-09-17 is where the Yahoo series the Lab
# used before starts, so results stay comparable.
LAB_START = {"Bitcoin": "2014-09-17"}


def _fetcher():
    """tv_fetch.py, from this folder or from tradingview/ (loaded by path: it is a script, not a package)."""
    for path in (os.path.join(ROOT, "tv_fetch.py"), os.path.join(ROOT, "tradingview", "tv_fetch.py")):
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("tv_fetch", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("tv_fetch.py not found next to tv_prices.py or in tradingview/")


def first_daily_position(days, window: int = 60, max_gap: int = 7) -> int:
    """`days`: the bars' dates as whole day numbers, ascending. Returns the position of the first bar that starts a sustained daily stretch (the next `window` gaps are all `max_gap` days or less).
    Everything before it is the sparse part: TradingView's S&P 500 has about one point a month until 1949, not daily bars. 0 when the series is daily from its first bar (or too short to tell)."""
    gaps = np.diff(np.asarray(days, dtype="int64"))
    if len(gaps) < window:
        return 0
    running = np.concatenate([[0], np.cumsum(gaps > max_gap)])
    daily_from = np.flatnonzero(running[window:] - running[:-window] == 0)
    return int(daily_from[0]) if len(daily_from) else 0


def file_for(market: str) -> str:
    return os.path.join(DATA_DIR, f"{market}.csv")


def has(market: str) -> bool:
    return market in MARKETS and os.path.exists(file_for(market))


def load(market: str, start=None, end=None, lab: bool = True) -> pd.DataFrame:
    """The market's daily bars as an OHLC frame indexed by trading date. A futures bar is stamped with its session START (the evening before, 22:00 or 23:00 UTC), so the date is taken two hours later:
    the bar of the session that ENDS on a day carries that day's date, like Yahoo's. Index bars (opening 13:30 / 14:30 UTC and so on) keep their own date.
    lab=True (the default) starts a market no earlier than LAB_START says; lab=False does not apply LAB_START. Bars before the first sustained daily stretch (monthly points) are always left out."""
    df = pd.read_csv(file_for(market))
    cut = first_daily_position((df["unix"].astype("int64").to_numpy() + 2 * 3600) // 86400)      # monthly bars never reach a run, even from an old file
    df = df.iloc[cut:]
    index = pd.DatetimeIndex(pd.to_datetime(df["unix"].astype("int64") + 2 * 3600, unit="s")).normalize()
    out = pd.DataFrame({"Open": df["open"].to_numpy(float), "High": df["high"].to_numpy(float), "Low": df["low"].to_numpy(float), "Close": df["close"].to_numpy(float)}, index=index)
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    first = LAB_START.get(market) if lab else None
    if first is not None:
        start = pd.Timestamp(first) if start is None else max(pd.Timestamp(start), pd.Timestamp(first))
    if start is not None:
        out = out.loc[pd.Timestamp(start):]
    if end is not None:
        out = out.loc[:pd.Timestamp(end)]
    return out


def manifest() -> pd.DataFrame:
    cols = ["market", "requested", "resolved", "bars", "first", "last", "delay_min", "fetched_at", "monthly_removed"]
    return pd.read_csv(MANIFEST).reindex(columns=cols) if os.path.exists(MANIFEST) else pd.DataFrame(columns=cols)


def save_removed(market: str, removed: dict, tv=None):
    """Keeps the monthly bars cut out of a market's file in REMOVED_DIR (one CSV per market); deletes an old copy when nothing was cut this time."""
    path = os.path.join(REMOVED_DIR, f"{market}.csv")
    if not removed:
        if os.path.exists(path):
            os.remove(path)
        return
    os.makedirs(REMOVED_DIR, exist_ok=True)
    (tv or _fetcher()).write_csv(path, removed)


def strip_file(market: str) -> int:
    """Cuts the monthly bars out of an ALREADY DOWNLOADED file in place (the cut bars go to REMOVED_DIR). Returns how many were cut (0 = the file was already daily)."""
    path = file_for(market)
    df = pd.read_csv(path)
    cut = first_daily_position((df["unix"].astype("int64").to_numpy() + 2 * 3600) // 86400)
    if cut == 0:
        return 0
    os.makedirs(REMOVED_DIR, exist_ok=True)
    df.iloc[:cut].to_csv(os.path.join(REMOVED_DIR, f"{market}.csv"), index=False)
    df.iloc[cut:].to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    table = manifest()
    if market in set(table["market"]):
        kept = df.iloc[cut:]
        table.loc[table["market"] == market, ["bars", "first"]] = [len(kept), str(pd.Timestamp(int(kept["unix"].iloc[0]) + 2 * 3600, unit="s").date())]
        table.loc[table["market"] == market, "monthly_removed"] = cut
        table.to_csv(MANIFEST, index=False)
    return cut


def update(market: str, symbol: str = None, log=print) -> dict:
    """Fetches one market's whole daily history and replaces its file. Returns the manifest row; raises on failure (the old file is kept)."""
    symbol = symbol or MARKETS.get(market) or EXTRAS[market]
    tv = _fetcher()
    bars, info, feed = tv.fetch_history(symbol, "1D", 1_000_000, "regular", log=lambda *_: None)
    feed.close()
    if not bars:
        raise RuntimeError(f"{market}: no bars came back for {symbol}")
    times = sorted(bars)
    cut = first_daily_position((np.array(times, dtype="int64") + 2 * 3600) // 86400)           # the monthly bars at the start (S&P 500 before 1950) are cut out, not saved
    removed = {t: bars[t] for t in times[:cut]}
    bars = {t: bars[t] for t in times[cut:]}
    os.makedirs(DATA_DIR, exist_ok=True)
    save_removed(market, removed, tv)
    path = file_for(market)
    tv.write_csv(path + ".tmp", bars)
    os.replace(path + ".tmp", path)
    delay = info.get("delay_seconds") or 0
    row = {"market": market, "requested": symbol, "resolved": info.get("full_name") or symbol, "bars": len(bars), "first": tv.iso(min(bars))[:10], "last": tv.iso(max(bars))[:10],
           "delay_min": int(delay) // 60, "fetched_at": time.strftime("%Y-%m-%d %H:%M"), "monthly_removed": len(removed)}
    table = manifest()
    table = pd.concat([table[table["market"] != market], pd.DataFrame([row])], ignore_index=True)
    table.to_csv(MANIFEST, index=False)
    log(f"{market}: {row['bars']} bars {row['first']} -> {row['last']}  ({row['resolved']}" + (f", delayed {row['delay_min']} min" if row["delay_min"] else "")
        + (f", {len(removed)} monthly bars cut out" if removed else "") + ")")
    return row


def update_all(markets=None, include_extras: bool = True, log=print, pause: float = 1.0) -> list:
    """Fetches every market (or the given ones) one after the other; a market that fails is reported and the others carry on. Returns the manifest rows and a list of failures."""
    names = list(markets) if markets else list(MARKETS) + (list(EXTRAS) if include_extras else [])
    rows, failures = [], []
    for name in names:
        try:
            rows.append(update(name, log=log))
        except Exception as e:                                      # one market failing must not stop the others
            failures.append((name, str(e)))
            log(f"{name}: FAILED - {e}")
        time.sleep(pause)
    return rows, failures


def export_to(folder: str) -> list:
    """Copies every CSV (and the manifest) to `folder`. Returns the file names copied."""
    os.makedirs(folder, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".csv"):
            shutil.copy2(os.path.join(DATA_DIR, name), os.path.join(folder, name))
            copied.append(name)
    return copied


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = [a for a in args if a not in MARKETS and a not in EXTRAS]
    if unknown:
        sys.exit(f"Unknown market(s): {', '.join(unknown)}. Markets: {', '.join(list(MARKETS) + list(EXTRAS))}")
    done, failed = update_all(args or None)
    if "--desktop" in sys.argv:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop", "tradingview_data")
        print(f"Copied {len(export_to(desktop))} files to {desktop}")
    print(f"{len(done)} markets fetched, {len(failed)} failed.")
    sys.exit(1 if failed else 0)
