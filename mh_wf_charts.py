"""Chart data for the Positions tab of the lean walk-forward app (mh_walkforward_ui.py): the out-of-sample trades of every window drawn on one market's price history with
streamlit-lightweight-charts - entry / add / take-profit / stop markers, the stepped position boxes (green = profit above the average entry, red = risk down to the stop which steps up with every
add, teal = stop above the average entry) and a line at every monthly high. Each walk-forward window traded its OWN chosen settings, so the monthly-high lines of a window are drawn with that
window's min / max age, buffer and SL ATR period, and only over that window's test period."""
import re

import numpy as np
import pandas as pd

import turtle_dual_system as td

GREEN, RED, TEAL = "rgba(76,175,80,", "rgba(244,67,54,", "rgba(0,150,136,"
CLEAR = "rgba(0,0,0,0)"
ORANGE = "rgba(255,152,0,0.9)"
_STOP_REASONS = {"SL"}
_TP_REASONS = {"scale_out"}


def chart_layout(dark: bool) -> dict:
    return {"textColor": "#e5e9f0" if dark else "#0f172a", "background": {"type": "solid", "color": "#0e131b" if dark else "#ffffff"}}


def candles(df: pd.DataFrame) -> list:
    return [{"time": ts.strftime("%Y-%m-%d"), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
            for ts, o, h, l, c in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"])]


def _zone(start, end, entry_price: float, edge_price: float, is_profit: bool, rgb: str, alpha: float, title: str = "") -> dict:
    """A shaded rectangle between entry_price and edge_price over [start, end] (a 'Baseline' series: it fills between its flat line and a fixed baseline price)."""
    fill, line = f"{rgb}{alpha})", f"{rgb}0.7)"
    return {"type": "Baseline",
            "data": [{"time": start.strftime("%Y-%m-%d"), "value": float(edge_price)}, {"time": end.strftime("%Y-%m-%d"), "value": float(edge_price)}],
            "options": {"baseValue": {"type": "price", "price": float(entry_price)},
                        "topLineColor": line if is_profit else CLEAR, "topFillColor1": fill if is_profit else CLEAR, "topFillColor2": fill if is_profit else CLEAR,
                        "bottomLineColor": CLEAR if is_profit else line, "bottomFillColor1": CLEAR if is_profit else fill, "bottomFillColor2": CLEAR if is_profit else fill,
                        "lineWidth": 1, "title": title, "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False}}


def _distinct_units(trade: dict) -> list:
    """The units a trade opened, one record each (a unit that was closed in several parts shows up once per part in unit_details)."""
    seen = {}
    for u in trade.get("unit_details") or []:
        seen.setdefault(u["entry_iloc"], u)
    return sorted(seen.values(), key=lambda u: u["entry_iloc"])


def markers_for(trades: list, exit_markers: list) -> list:
    """Entry / add arrows below the bars, take-profit / stop arrows above them with the R multiple and the share of the account each exit made or lost."""
    markers, enrich = [], {}
    for t in trades:
        for u in t.get("unit_details") or []:
            if u.get("exit_date") is None:
                continue
            risk, eq = u.get("risk_amount") or 0.0, u.get("equity_at_entry") or 0.0
            enrich.setdefault((u["exit_date"].strftime("%Y-%m-%d"), u["exit_reason"]), []).append(
                ((u["pnl"] / risk) if risk else float("nan"), (u["pnl"] / eq * 100) if eq else float("nan")))
        for k, u in enumerate(_distinct_units(t)):
            eq = u.get("equity_at_entry") or 0.0
            risk = f" (risk {u['risk_amount'] / eq * 100:.1f}%)" if eq else ""
            markers.append({"time": u["entry_date"].strftime("%Y-%m-%d"), "position": "belowBar", "color": "#2196F3" if k == 0 else "#90CAF9", "shape": "arrowUp",
                            "text": ("Entry" if k == 0 else "Add") + risk})
    for date, price, reason in exit_markers:
        key = (date.strftime("%Y-%m-%d"), reason)
        info = enrich.get(key, []).pop(0) if enrich.get(key) else None
        extra = ""
        if info is not None:
            r, e = info
            extra = (f" {r:+.2f}R" if np.isfinite(r) else "") + (f" ({e:+.1f}% equity)" if np.isfinite(e) else "")
        if reason in _STOP_REASONS:
            markers.append({"time": key[0], "position": "aboveBar", "color": "#F44336", "shape": "arrowDown", "text": "SL" + extra})
        elif reason in _TP_REASONS:
            markers.append({"time": key[0], "position": "aboveBar", "color": "#4CAF50", "shape": "arrowDown", "text": "TP" + extra})
        else:
            markers.append({"time": key[0], "position": "aboveBar", "color": "#9E9E9E", "shape": "circle", "text": {"end_of_data": "End"}.get(reason, reason[:4]) + extra})
    markers.sort(key=lambda m: m["time"])
    return markers


def stepped_boxes(df: pd.DataFrame, trades: list, since=None) -> list:
    """Per trade: green area from the average entry up to the close while the position is open (to the exit price on the last day); red area from the average entry down to the stop, in steps -
    every add re-sets the stop to stop multiple x ATR under the newest unit - and teal once the stop is above the average entry."""
    series, close = [], df["Close"]
    for t in trades:
        units = _distinct_units(t)
        if not units or (since is not None and t["exit_date"] < since):
            continue
        sum_q = sum_pq = 0.0
        stop = None
        for k, u in enumerate(units):
            q = float(u.get("initial_shares") or u.get("shares") or 0.0)
            sum_q += q
            sum_pq += q * float(u["entry_price"])
            avg = sum_pq / sum_q if sum_q > 0 else float(u["entry_price"])
            stop = float(u["entry_price"]) - float(u["stop_n_mult"]) * float(u["n_value"])          # the shared stop sits under the NEWEST unit
            start = u["entry_date"]
            end = units[k + 1]["entry_date"] if k + 1 < len(units) else t["exit_date"]
            if not end > start:
                continue
            locked = stop > avg
            series.append(_zone(start, end, avg, stop, is_profit=locked, rgb=TEAL if locked else RED, alpha=0.30 if locked else 0.20, title="stop" if k == 0 else ""))
            days = df.loc[start:end]
            if len(days) >= 2:
                values = [max(avg, float(c)) for c in close.loc[days.index]]
                if k + 1 == len(units) and t.get("exit_price") is not None:
                    values[-1] = max(avg, float(t["exit_price"]))
                series.append({"type": "Baseline", "data": [{"time": d.strftime("%Y-%m-%d"), "value": v} for d, v in zip(days.index, values)],
                               "options": {"baseValue": {"type": "price", "price": float(avg)}, "topLineColor": f"{GREEN}0.5)", "topFillColor1": f"{GREEN}0.16)", "topFillColor2": f"{GREEN}0.16)",
                                           "bottomLineColor": CLEAR, "bottomFillColor1": CLEAR, "bottomFillColor2": CLEAR, "lineWidth": 1, "title": "", "priceLineVisible": False,
                                           "lastValueVisible": False, "crosshairMarkerVisible": False}})
    return series


def monthly_high_lines(df: pd.DataFrame, windows: list, since=None) -> list:
    """A line at every monthly high, per window with that window's own settings: from the bar that set it until its breakout (the first close above high + buffer x ATR) or until it is older than
    max age, dotted while younger than min age. Only the part inside the window's test period is drawn. `windows`: [{'params', 'test_start', 'test_end'}]."""
    top, pos, nxt = td.monthly_high_months(df.index, df["High"])
    close = df["Close"].to_numpy(dtype=float)
    last = len(df) - 1
    atr_cache, out = {}, []
    for w in windows:
        if since is not None and w["test_end"] < since:
            continue
        p = w["params"]
        n_period = int(p.get("n_atr_period", 40))
        if n_period not in atr_cache:
            atr_cache[n_period] = np.asarray(td.compute_atr(df, n_period), dtype=float)
        atr = atr_cache[n_period]
        min_age, max_age, buffer = int(p.get("mh_min_age", 20)), int(p.get("mh_max_age", 365)), float(p.get("entry_buffer_n_mult", 0.0))
        lo = int(df.index.searchsorted(max(w["test_start"], since) if since is not None else w["test_start"], side="left"))
        hi = min(int(df.index.searchsorted(w["test_end"], side="right")) - 1, last)
        if hi <= lo:
            continue
        for g in range(len(top)):
            if np.isnan(top[g]):
                continue
            p0 = int(pos[g])
            if p0 > hi:
                break
            end = min(p0 + max_age, last)
            t0 = int(nxt[g])
            if t0 <= end:
                broken = close[t0:end + 1] > top[g] + buffer * atr[t0:end + 1]
                if broken.any():
                    end = t0 + int(np.argmax(broken))
            if end < lo:
                continue
            young_end = min(p0 + min_age, end)
            for a, b, dotted in ((p0, young_end, True), (young_end, end, False)):
                a2, b2 = max(a, lo), min(b, hi)
                if b2 > a2:
                    out.append({"type": "Line", "data": [{"time": df.index[a2].strftime("%Y-%m-%d"), "value": float(top[g])}, {"time": df.index[b2].strftime("%Y-%m-%d"), "value": float(top[g])}],
                                "options": {"color": ORANGE, "lineWidth": 1, "lineStyle": 3 if dotted else 0, "title": "", "priceLineVisible": False, "lastValueVisible": False,
                                            "crosshairMarkerVisible": False}})
    return out


def asset_trades(chart_data: list, asset: str) -> tuple:
    """(trades, exit_markers, windows) of one market over every window, in window order. windows = [{'window', 'params', 'test_start', 'test_end'}] of the windows that traded it."""
    trades, exits, windows = [], [], []
    for w in chart_data or []:
        d = (w.get("assets") or {}).get(asset)
        if not d:
            continue
        ts = [t for lst in d["trades_by_system"].values() for t in lst]
        ms = [m for lst in d["exit_markers_by_system"].values() for m in lst]
        windows.append({"window": w["window"], "params": w["params"], "test_start": w["test_start"], "test_end": w["test_end"]})
        for t in ts:
            t = dict(t)
            t["_window"] = w["window"]
            trades.append(t)
        exits.extend(ms)
    return trades, exits, windows


def markets_with_trades(chart_data: list) -> list:
    return sorted({a for w in chart_data or [] for a, d in (w.get("assets") or {}).items() if any(d["trades_by_system"].values())})


def trade_table(trades: list) -> pd.DataFrame:
    rows = []
    for t in sorted(trades, key=lambda x: x["exit_date"]):
        units = _distinct_units(t)
        rows.append({"window": t.get("_window"), "first entry": units[0]["entry_date"].date() if units else None, "last exit": t["exit_date"].date(), "units": t.get("n_units"),
                     "P&L $": round(float(t["pnl"]), 0), "R": round(float(t["r_multiple"]), 2) if np.isfinite(t.get("r_multiple", np.nan)) else None, "last exit reason": t.get("exit_reason")})
    return pd.DataFrame(rows)
