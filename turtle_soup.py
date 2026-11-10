"""
Turtle Soup -- a mean-reversion strategy engine, built as the deliberate mirror image of
turtle_dual_system.py's breakout trend-following.
============================================================================================
Classic setup (Linda Bradford Raschke / Larry Connors, "Street Smarts"): fade a breakout that
fails to hold. Where the Turtle system BUYS a new N-day HIGH (assuming the breakout continues),
Turtle Soup BUYS the REVERSAL after a new N-day LOW that fails to hold (assuming the breakdown
was a shakeout, not a real move) -- long-only here, consistent with the rest of this project.

WHY THIS EXISTS: the regime diagnostics run earlier in this project consistently found that
turtle_dual_system.py has real edge in trending conditions and loses it in chop (Ger40,
CrudeOil specifically). Rather than keep trying to make a trend-following system tolerate chop,
this is a genuinely different engine designed to have edge IN chop, meant to run ALONGSIDE the
trend systems (see optimize_turtle_soup.py and turtle_ui.py for the combined-portfolio view),
not replace them.

CORE MECHANICS:
  1. QUALIFYING BREACH: close makes a new N-day low (the same rolling-low concept the Turtle
     exit channel already uses), AND this is the FIRST such breach in at least
     `quiet_period_days` -- a "surprise" failure, not an already-established downtrend.
  2. ENTRY: NOT on the breach day itself -- on the first later day price closes back ABOVE the
     breached level (the reversal IS the trade). If `use_plus_one` is on (the "Turtle Soup Plus
     One" variant), the day right after the breach must NOT make an even lower low first --
     slower to trigger, generally cited as more reliable.
  3. STOP: tight, just below the actual extreme low that was made, buffered by a fraction of N.
  4. EXIT: a FIXED profit target in R-multiples (NOT a trailing stop -- mean reversion should
     resolve fast, "let it run" is the wrong instinct here) OR a time stop if neither the target
     nor the stop is hit within a few days (if reversion hasn't happened by then, the thesis is
     probably wrong).
  5. REGIME GATE (optional): a Choppiness Index filter, but INVERTED relative to the Turtle
     system's version -- requires the market to NOT be strongly trending, since that's
     specifically where this strategy's edge should live.

No pyramiding -- this is a single-shot, in-and-out strategy by design, deliberately much
simpler than turtle_dual_system.py's multi-unit position management.
"""

import os
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Nasdaq": "^NDX",
}
START_DATE = (pd.Timestamp.now() - pd.DateOffset(years=30)).strftime("%Y-%m-%d")
END_DATE = None

STARTING_CAPITAL_PER_ASSET = 10000.0
# 5000% = 50x -- deliberate, matches turtle_dual_system.py's own MAX_TOTAL_EXPOSURE_PCT (50x is
# genuinely accessible today via crypto/CFD/forex brokers, a real scenario worth backtesting).
MAX_TOTAL_EXPOSURE_PCT = 5000

DEFAULT_SOUP_PARAMS = {
    "channel_lookback": 20,          # classic Turtle Soup: 20-day low
    "quiet_period_days": 4,          # the breach must be the FIRST in at least this many days
    "reversal_window_days": 3,       # how many days after the breach to watch for the reversal
                                       # entry before giving up on this setup
    "use_plus_one": False,           # "Turtle Soup Plus One": require day-after-breach to NOT
                                       # make a new low before allowing entry -- slower, stricter
    "n_atr_period": 20,
    "risk_per_trade_pct": 2.0,
    "stop_buffer_n_mult": 0.5,       # stop = breach day's low - this * N
    "target_r_mult": 2.0,            # FIXED profit target, in R -- not a trailing stop
    "time_stop_days": 5,             # exit at market if neither target nor stop hit by then
    "use_choppiness_filter": True,   # INVERTED vs. the Turtle system's version --
    "choppiness_period": 14,          # requires CI ABOVE this (i.e. NOT trending) to enter
    "choppiness_min": 38.2,
}

SYSTEMS = [
    {"name": "Soup", **DEFAULT_SOUP_PARAMS},
]

SLIPPAGE_PCT = 0.02
SPREAD_PCT = 0.02
COMMISSION_PCT = 0.01
HOLDING_COST_PCT_PER_YEAR = 2.0


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def download_data(tickers: dict, start: str, end) -> dict:
    import yfinance as yf
    data = {}
    for name, tk in tickers.items():
        df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: no data for {name} ({tk}) -- skipping")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        data[name] = df
        print(f"{name}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return data


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_choppiness_index(df: pd.DataFrame, period: int) -> pd.Series:
    """Identical formula to turtle_dual_system.py / regime_diagnostic.py -- LOW = trending,
    HIGH = choppy. This engine wants HIGH values (chop), the opposite of the Turtle system."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    tr_sum = tr.rolling(period).sum()
    range_high = high.rolling(period).max()
    range_low = low.rolling(period).min()
    span = (range_high - range_low).replace(0, np.nan)
    return 100 * np.log10(tr_sum / span) / np.log10(period)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sysdef in SYSTEMS:
        name = sysdef["name"]
        out[f"low_channel_{name}"] = out["Low"].rolling(sysdef["channel_lookback"]).min().shift(1)
        out[f"n_{name}"] = compute_atr(out, sysdef["n_atr_period"])
        out[f"choppiness_{name}"] = compute_choppiness_index(out, sysdef["choppiness_period"])
    return out


def entry_fill_price(signal_price: float) -> float:
    return signal_price * (1 + (SLIPPAGE_PCT + SPREAD_PCT / 2) / 100.0)


def exit_fill_price(signal_price: float) -> float:
    return signal_price * (1 - (SLIPPAGE_PCT + SPREAD_PCT / 2) / 100.0)


def commission_cost(notional: float) -> float:
    return notional * (COMMISSION_PCT / 100.0)


def daily_holding_cost(notional: float) -> float:
    return notional * (HOLDING_COST_PCT_PER_YEAR / 100.0) / 365.0


def size_unit(equity: float, entry_price: float, stop_price: float, current_notional: float, risk_pct: float):
    """Same authentic-Turtle sizing convention validated earlier in this project: shares sized
    so that a stop-out loses approximately risk_pct% of equity -- divide by the FULL stop
    distance, not a bare volatility unit."""
    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        return None
    risk_amount = equity * (risk_pct / 100.0)
    target_shares = risk_amount / stop_distance
    target_notional = target_shares * entry_price

    max_total_notional = (MAX_TOTAL_EXPOSURE_PCT / 100.0) * equity
    room_left = max_total_notional - current_notional
    if room_left <= 0:
        return None

    final_notional = min(target_notional, room_left)
    final_shares = final_notional / entry_price
    actual_risk_amount = final_shares * stop_distance
    return {"shares": final_shares, "risk_amount": actual_risk_amount, "capital_deployed": final_notional}


def performance_stats(equity_curve: pd.Series, starting_capital: float) -> dict:
    if len(equity_curve) < 2:
        return {"Sharpe": np.nan, "CAGR": np.nan, "Max Drawdown": np.nan, "Final Equity": equity_curve.iloc[-1]
                if len(equity_curve) else starting_capital}
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else np.nan
    n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / starting_capital) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()
    return {"Sharpe": sharpe, "CAGR": cagr, "Max Drawdown": max_dd, "Final Equity": equity_curve.iloc[-1]}


# ----------------------------------------------------------------------------
# SIMULATION
# ----------------------------------------------------------------------------
def simulate_asset(df: pd.DataFrame, starting_capital: float = STARTING_CAPITAL_PER_ASSET) -> dict:
    n_bars = len(df)
    close = df["Close"].values
    low = df["Low"].values
    high = df["High"].values
    dates = df.index

    sys_low_channel = {s["name"]: df[f"low_channel_{s['name']}"].values for s in SYSTEMS}
    sys_n = {s["name"]: df[f"n_{s['name']}"].values for s in SYSTEMS}
    sys_choppiness = {s["name"]: df[f"choppiness_{s['name']}"].values for s in SYSTEMS}

    equity = starting_capital * len(SYSTEMS)
    equity_curve = np.zeros(n_bars)

    systems = {}
    for sysdef in SYSTEMS:
        systems[sysdef["name"]] = {
            "name": sysdef["name"],
            "in_position": False, "shares": 0.0, "entry_price": None, "entry_date": None,
            "entry_iloc": None, "stop_price": None, "target_price": None, "capital_deployed": 0.0,
            "last_breach_iloc": None,       # most recent QUALIFYING breach not yet resolved
            "breach_low": None,             # the actual extreme low made on that breach
            "watching_since_iloc": None,     # start of the reversal-entry watch window
            "plus_one_confirmed": False,     # for use_plus_one: has day-after-breach confirmed?
            "trades": [], "entry_markers": [], "exit_markers": [],
        }

    total_slippage_spread_cost = 0.0
    total_commission_cost = 0.0
    total_holding_cost = 0.0

    def other_systems_notional(exclude_name):
        return sum(s["capital_deployed"] for name, s in systems.items() if name != exclude_name)

    for i in range(n_bars):
        if HOLDING_COST_PCT_PER_YEAR > 0:
            open_notional_today = sum(s["capital_deployed"] for s in systems.values() if s["in_position"])
            if open_notional_today > 0:
                cost_today = daily_holding_cost(open_notional_today)
                equity -= cost_today
                total_holding_cost += cost_today
        equity_curve[i] = equity

        for sysdef in SYSTEMS:
            name = sysdef["name"]
            sys = systems[name]

            if sys["in_position"]:
                stop_hit = low[i] <= sys["stop_price"]
                target_hit = high[i] >= sys["target_price"]
                days_held = i - sys["entry_iloc"]
                time_hit = days_held >= sysdef["time_stop_days"]

                if stop_hit or target_hit or time_hit:
                    if stop_hit:
                        raw_price, reason = sys["stop_price"], "SL"
                    elif target_hit:
                        raw_price, reason = sys["target_price"], "target"
                    else:
                        raw_price, reason = close[i], "time_stop"
                    fill = exit_fill_price(raw_price)
                    comm = commission_cost(sys["shares"] * fill)
                    equity += sys["shares"] * (fill - sys["entry_price"]) - comm
                    total_commission_cost += comm
                    pnl = sys["shares"] * (fill - sys["entry_price"]) - comm - sys["entry_commission"]
                    risk_amount = sys["shares"] * (sys["entry_price"] - sys["stop_price"])
                    r_multiple = pnl / risk_amount if risk_amount > 0 else np.nan
                    sys["trades"].append({
                        "entry_date": sys["entry_date"], "exit_date": dates[i],
                        "entry_price": sys["entry_price"], "exit_price": fill,
                        "shares": sys["shares"], "exit_reason": reason, "pnl": pnl,
                        "risk_amount": risk_amount, "r_multiple": r_multiple,
                        "holding_days": days_held,
                        # frozen at entry, kept even though sys["stop_price"]/["target_price"]
                        # get reset to None right below -- lets a chart draw the actual planned
                        # SL/TP levels for this trade, not just where it happened to exit
                        "stop_price": sys["stop_price"], "target_price": sys["target_price"],
                    })
                    sys["exit_markers"].append((i, fill, reason))
                    sys["in_position"] = False
                    sys["shares"] = 0.0
                    sys["capital_deployed"] = 0.0
                    sys["entry_price"] = None
                    sys["stop_price"] = None
                    sys["target_price"] = None
            else:
                low_channel_here = sys_low_channel[name][i]
                n_here = sys_n[name][i]

                # --- detect a fresh QUALIFYING breach: new N-day low, first in quiet_period_days ---
                is_new_low = (not np.isnan(low_channel_here)) and low[i] < low_channel_here
                if is_new_low:
                    days_since_last_breach = (i - sys["last_breach_iloc"]) if sys["last_breach_iloc"] is not None else None
                    is_first_in_a_while = (days_since_last_breach is None or
                                             days_since_last_breach >= sysdef["quiet_period_days"])
                    if is_first_in_a_while:
                        sys["last_breach_iloc"] = i
                        sys["breach_low"] = low[i]
                        sys["watching_since_iloc"] = i
                        sys["plus_one_confirmed"] = not sysdef["use_plus_one"]
                    else:
                        sys["watching_since_iloc"] = None

                if (sysdef["use_plus_one"] and sys["watching_since_iloc"] is not None
                        and not sys["plus_one_confirmed"] and i == sys["last_breach_iloc"] + 1):
                    if low[i] < sys["breach_low"]:
                        sys["watching_since_iloc"] = None
                    else:
                        sys["breach_low"] = min(sys["breach_low"], low[i])
                        sys["plus_one_confirmed"] = True

                if sys["watching_since_iloc"] is not None and sys["plus_one_confirmed"]:
                    window_expired = (i - sys["watching_since_iloc"]) > sysdef["reversal_window_days"]
                    if window_expired:
                        sys["watching_since_iloc"] = None
                    else:
                        breached_level = sys_low_channel[name][sys["last_breach_iloc"]]
                        reversal = close[i] > breached_level
                        choppiness_ok = (not sysdef["use_choppiness_filter"]) or (
                            not np.isnan(sys_choppiness[name][i])
                            and sys_choppiness[name][i] >= sysdef["choppiness_min"])
                        if reversal and choppiness_ok and not np.isnan(n_here) and n_here > 0:
                            fill = entry_fill_price(close[i])
                            stop_price = sys["breach_low"] - sysdef["stop_buffer_n_mult"] * n_here
                            if stop_price < fill:
                                current_notional = other_systems_notional(name)
                                sized = size_unit(equity, fill, stop_price, current_notional,
                                                    sysdef["risk_per_trade_pct"])
                                if sized is not None and sized["shares"] > 1e-9:
                                    comm = commission_cost(sized["capital_deployed"])
                                    equity -= comm
                                    total_commission_cost += comm
                                    risk_per_share = fill - stop_price
                                    sys["in_position"] = True
                                    sys["shares"] = sized["shares"]
                                    sys["capital_deployed"] = sized["capital_deployed"]
                                    sys["entry_price"] = fill
                                    sys["entry_date"] = dates[i]
                                    sys["entry_iloc"] = i
                                    sys["entry_commission"] = comm
                                    sys["stop_price"] = stop_price
                                    sys["target_price"] = fill + sysdef["target_r_mult"] * risk_per_share
                                    sys["entry_markers"].append((i, fill))
                            sys["watching_since_iloc"] = None

    for sys in systems.values():
        if sys["in_position"]:
            fill = exit_fill_price(close[-1])
            comm = commission_cost(sys["shares"] * fill)
            pnl = sys["shares"] * (fill - sys["entry_price"]) - comm - sys.get("entry_commission", 0)
            risk_amount = sys["shares"] * (sys["entry_price"] - sys["stop_price"])
            sys["trades"].append({
                "entry_date": sys["entry_date"], "exit_date": dates[-1],
                "entry_price": sys["entry_price"], "exit_price": fill, "shares": sys["shares"],
                "exit_reason": "end_of_data", "pnl": pnl, "risk_amount": risk_amount,
                "r_multiple": pnl / risk_amount if risk_amount > 0 else np.nan,
                "holding_days": (n_bars - 1) - sys["entry_iloc"],
                "stop_price": sys["stop_price"], "target_price": sys["target_price"],
            })

    all_trades = [t for s in systems.values() for t in s["trades"]]
    all_trades.sort(key=lambda t: t["entry_date"])

    return {
        "equity_curve": pd.Series(equity_curve, index=dates),
        "trades": all_trades,
        "trades_by_system": {name: s["trades"] for name, s in systems.items()},
        "entry_markers_by_system": {name: s["entry_markers"] for name, s in systems.items()},
        "exit_markers_by_system": {name: s["exit_markers"] for name, s in systems.items()},
        "total_slippage_spread_cost": total_slippage_spread_cost,
        "total_commission_cost": total_commission_cost,
        "total_holding_cost": total_holding_cost,
    }


# ----------------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------------
def episode_stats(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"Episodes": 0, "Win Rate": np.nan, "Avg R": np.nan, "Avg Holding (days)": np.nan,
                "Avg PnL": np.nan, "Profit Factor": np.nan}
    pnls = np.array([t["pnl"] for t in trades])
    r_mults = np.array([t["r_multiple"] for t in trades if pd.notna(t["r_multiple"])])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    return {
        "Episodes": n, "Win Rate": win_rate, "Avg R": r_mults.mean() if len(r_mults) else np.nan,
        "Avg Holding (days)": np.mean([t["holding_days"] for t in trades]),
        "Avg PnL": pnls.mean(), "Profit Factor": profit_factor,
    }


def build_episode_dataframe(results_by_asset: dict) -> pd.DataFrame:
    rows = []
    for asset, result in results_by_asset.items():
        for t in result["trades"]:
            rows.append({
                "asset": asset, "entry_date": t["entry_date"].date(), "exit_date": t["exit_date"].date(),
                "holding_days": t["holding_days"], "exit_reason": t["exit_reason"],
                "win": t["pnl"] > 0, "pnl": round(t["pnl"], 2),
                "r_multiple": round(t["r_multiple"], 3) if pd.notna(t["r_multiple"]) else np.nan,
                "entry_price": round(t["entry_price"], 4),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("TURTLE SOUP -- mean-reversion (fade failed breakouts) strategy")
    print("=" * 78)
    for s in SYSTEMS:
        print(f"\n--- System {s['name']} ---")
        print(f"  Channel: {s['channel_lookback']}-day low, first breach in >= {s['quiet_period_days']}d")
        print(f"  Plus One: {s['use_plus_one']}")
        print(f"  Reversal entry window: {s['reversal_window_days']}d after breach")
        print(f"  Stop: breach low - {s['stop_buffer_n_mult']}N   |   Target: {s['target_r_mult']}R   |   "
              f"Time stop: {s['time_stop_days']}d")
        print(f"  Sizing: (equity x {s['risk_per_trade_pct']}%) / stop distance")
        print(f"  Choppiness filter (must NOT be trending): {s['use_choppiness_filter']}"
              + (f" (CI >= {s['choppiness_min']})" if s['use_choppiness_filter'] else ""))

    print(f"\nStarting capital: ${STARTING_CAPITAL_PER_ASSET:,.0f} per asset")
    print("\nDownloading data...")
    raw = download_data(TICKERS, START_DATE, END_DATE)

    results = {}
    for name, df in raw.items():
        feat = build_features(df)
        results[name] = simulate_asset(feat)

    print("\n" + "=" * 78)
    print("PER-ASSET RESULTS")
    print("=" * 78)
    for name, r in results.items():
        stats = performance_stats(r["equity_curve"], STARTING_CAPITAL_PER_ASSET)
        print(f"\n--- {name} ---")
        print(f"  Sharpe: {stats['Sharpe']:.2f}   CAGR: {stats['CAGR']*100:.2f}%   "
              f"MaxDD: {stats['Max Drawdown']*100:.2f}%   Final Equity: ${stats['Final Equity']:,.0f}")
        ep = episode_stats(r["trades"])
        if ep['Episodes']:
            print(f"  Episodes: {ep['Episodes']}   Win Rate: {ep['Win Rate']*100:.1f}%   "
                  f"Avg R: {ep['Avg R']:.2f}   Profit Factor: {ep['Profit Factor']:.2f}")
        else:
            print("  No closed episodes.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ep_df = build_episode_dataframe(results)
    if not ep_df.empty:
        csv_path = os.path.join(OUTPUT_DIR, "turtle_soup_episodes.csv")
        ep_df.to_csv(csv_path, index=False)
        print(f"\nEpisode data saved to: {csv_path}")


if __name__ == "__main__":
    main()