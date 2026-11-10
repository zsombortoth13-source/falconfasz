"""Short-term mean-reversion strategy: the counterweight to the trend-following system.

IDEA  After a sharp short-term sell-off in a market that is otherwise in an uptrend, prices tend to
snap back within a few days. Buy the dip, sell the bounce, and always carry a hard stop because the
occasional dip that keeps falling is what hurts this style. Optionally the mirror image (sell sharp
rallies in downtrends) when direction = "both".

HOW A TRADE WORKS (all decisions on a bar's CLOSE, all fills at the NEXT bar's open, like the trend engine)
  entry   one of three oversold measures, filtered by a long-term trend average:
            "rsi"   RSI(period) below a threshold        (a few days of selling)
            "nlow"  close below the lowest close of the previous N days   (a fresh short-term low)
            "ibs"   close near the bottom of the day's range               (closed on its lows)
          long trades need the close ABOVE its trend average (trend_sma bars), shorts BELOW it;
          trend_sma = 0 switches the filter off.
  exit    the next open after any of: price back above a short average ("sma"), RSI recovered ("rsi"),
          close above the previous day's high ("prev_extreme"), or max_hold bars have passed.
  stop    a hard stop stop_atr x ATR from the fill, active from the entry day; a gap through it fills
          at the open, like the trend engine. The stop distance also sizes the trade:
          shares = risk_pct of equity / stop distance.
  limits  one position per market, max_pos_pct of equity per position, max_gross_pct across all
          positions, at most max_positions open.

COSTS AND FILLS reuse turtle_dual_system's functions, so the broker profile (per-contract fees,
per-asset spread and slippage, holding cost, whole-contract steps) applies exactly as it does to the
trend system: td.entry_fill_price / exit_fill_price / commission_for / daily_holding_cost / SIZE_STEP.

NOT MODELLED here: margin calls (positions are short-lived and small relative to the account; the
gross cap stands in for a leverage limit), roll costs, whole-contract rounding of partial exits (there
are none: every exit closes the whole position).
"""
import numpy as np
import pandas as pd

import turtle_dual_system as td

DEFAULT_PARAMS = {
    "direction": "long", "entry_type": "rsi", "rsi_period": 2, "rsi_entry": 10.0, "nlow_days": 5,
    "ibs_entry": 0.2, "trend_sma": 200, "exit_type": "sma", "exit_sma": 5, "exit_rsi": 60.0,
    "max_hold": 10, "atr_period": 14, "stop_atr": 3.0,
    "risk_pct": 1.0, "max_pos_pct": 100.0, "max_gross_pct": 300.0, "max_positions": 15,
}

# what the optimizer may vary; the sizing/limit settings are pinned (they set the scale, which the
# blend with the trend system rescales anyway) so the search is about WHEN to trade, not how big
PARAM_SPACE = {
    "direction": ["long", "both"],
    "entry_type": ["rsi", "nlow", "ibs"],
    "rsi_period": [2, 3, 4, 5],
    "rsi_entry": [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    "nlow_days": [2, 3, 4, 5, 7, 10],
    "ibs_entry": [0.05, 0.1, 0.15, 0.2, 0.3],
    "trend_sma": [0, 50, 100, 150, 200, 300],
    "exit_type": ["sma", "rsi", "prev_extreme"],
    "exit_sma": [2, 3, 4, 5, 7, 10, 15],
    "exit_rsi": [40.0, 50.0, 60.0, 70.0, 80.0],
    "max_hold": [2, 3, 5, 7, 10, 15, 20],
    "atr_period": [10, 14, 20],
    "stop_atr": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0],
}
PINNED = {"risk_pct": 1.0, "max_pos_pct": 100.0, "max_gross_pct": 300.0, "max_positions": 15}

COST_SETTING_NAMES = ("SLIPPAGE_PCT", "SPREAD_PCT", "COMMISSION_PCT", "HOLDING_COST_PCT_PER_YEAR",
                      "USE_PER_CONTRACT_FEES", "FEE_SCHEDULE", "ASSET_COSTS", "SIZE_STEP")


def current_cost_config() -> dict:
    """The engine's cost settings, for shipping to worker processes (spawned workers re-import with defaults)."""
    return {name: getattr(td, name) for name in COST_SETTING_NAMES}


def apply_cost_config(cfg: dict):
    for name, value in cfg.items():
        setattr(td, name, value)


# ----------------------------------------------------------------------------
# data + indicator cache
# ----------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(~((loss == 0) & (gain > 0)), 100.0)   # all gains, no losses: RSI is 100


class MRData:
    """Price arrays for a set of markets on their shared date calendar (A markets x T dates, NaN where a
    market has no bar) plus a cache of every indicator, so many parameter sets can be tried cheaply."""

    def __init__(self, raw: dict):
        self.names = list(raw)
        self.frames = raw
        dates = None
        for df in raw.values():
            dates = df.index if dates is None else dates.union(df.index)
        self.dates = pd.DatetimeIndex(sorted(dates))
        self.A, self.T = len(self.names), len(self.dates)

        def grid(col):
            return np.vstack([raw[n][col].reindex(self.dates).values.astype(float) for n in self.names])
        self.O, self.H, self.L, self.C = grid("Open"), grid("High"), grid("Low"), grid("Close")
        self.has_bar = ~np.isnan(self.C)
        self._memo = {}

    def _series_grid(self, key, fn):
        hit = self._memo.get(key)
        if hit is None:
            hit = np.vstack([fn(self.frames[n]).reindex(self.dates).values.astype(float) for n in self.names])
            self._memo[key] = hit
        return hit

    def rsi(self, period):
        return self._series_grid(("rsi", period), lambda d: _rsi(d["Close"], period))

    def sma(self, length):
        return self._series_grid(("sma", length), lambda d: d["Close"].rolling(length).mean())

    def atr(self, period):
        return self._series_grid(("atr", period), lambda d: td.compute_atr(d, period))

    def prior_min_close(self, k):
        return self._series_grid(("pmin", k), lambda d: d["Close"].shift(1).rolling(k).min())

    def prior_max_close(self, k):
        return self._series_grid(("pmax", k), lambda d: d["Close"].shift(1).rolling(k).max())

    def ibs(self):
        return self._series_grid(("ibs",), lambda d: (d["Close"] - d["Low"]) / (d["High"] - d["Low"]).replace(0, np.nan))

    def prior_high(self):
        return self._series_grid(("phigh",), lambda d: d["High"].shift(1))

    def prior_low(self):
        return self._series_grid(("plow",), lambda d: d["Low"].shift(1))

    def signals(self, p: dict):
        """Boolean (A, T) arrays: entry and exit conditions evaluated on each bar's close."""
        C = self.C
        with np.errstate(invalid="ignore"):
            if p["entry_type"] == "rsi":
                r = self.rsi(p["rsi_period"])
                long_cond, short_cond = r < p["rsi_entry"], r > 100.0 - p["rsi_entry"]
            elif p["entry_type"] == "nlow":
                long_cond = C < self.prior_min_close(p["nlow_days"])
                short_cond = C > self.prior_max_close(p["nlow_days"])
            else:
                b = self.ibs()
                long_cond, short_cond = b < p["ibs_entry"], b > 1.0 - p["ibs_entry"]
            if p["trend_sma"]:
                trend = self.sma(p["trend_sma"])
                long_cond = long_cond & (C > trend)
                short_cond = short_cond & (C < trend)
            if p["exit_type"] == "sma":
                avg = self.sma(p["exit_sma"])
                long_exit, short_exit = C > avg, C < avg
            elif p["exit_type"] == "rsi":
                r2 = self.rsi(p["rsi_period"])
                long_exit, short_exit = r2 > p["exit_rsi"], r2 < 100.0 - p["exit_rsi"]
            else:
                long_exit, short_exit = C > self.prior_high(), C < self.prior_low()
        atr = self.atr(p["atr_period"])
        ok = self.has_bar & ~np.isnan(atr)
        long_in = long_cond & ok
        short_in = (short_cond & ok) if p["direction"] == "both" else np.zeros_like(long_in)
        return long_in, short_in, long_exit & self.has_bar, short_exit & self.has_bar, atr


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------
def simulate(data: MRData, p: dict, starting_capital: float = 100000.0, record_trades: bool = False) -> dict:
    """Runs the strategy over every date in `data`. Returns {"equity_curve", "trades", costs, counters}."""
    long_in, short_in, long_out, short_out, atr = data.signals(p)
    O, H, L, C, has_bar = data.O, data.H, data.L, data.C, data.has_bar
    names, dates, T = data.names, data.dates, data.T
    risk_frac, max_hold, stop_atr = p["risk_pct"] / 100.0, int(p["max_hold"]), float(p["stop_atr"])
    pos_cap, gross_cap, max_positions = p["max_pos_pct"] / 100.0, p["max_gross_pct"] / 100.0, int(p["max_positions"])
    has_holding_cost = td.HOLDING_COST_PCT_PER_YEAR > 0 or any(
        "holding_pct" in c and c["holding_pct"] > 0 for c in (td.ASSET_COSTS or {}).values())
    entry_any = long_in | short_in

    cash = float(starting_capital)
    equity_prev = cash
    positions, pending = {}, {}
    equity = np.empty(T)
    trades = []
    totals = {"slippage": 0.0, "commission": 0.0, "holding": 0.0, "skipped_room": 0, "skipped_rounding": 0,
              "skipped_max_positions": 0}
    n_assets = data.A

    def close_position(a, raw_price, reason, t):
        nonlocal cash
        pos = positions.pop(a)
        sign, shares, entry_fill = pos["sign"], pos["shares"], pos["entry"]
        fill = td.exit_fill_price(raw_price, names[a]) if sign > 0 else td.entry_fill_price(raw_price, names[a])
        gross = sign * shares * (fill - entry_fill)
        comm = td.commission_for(names[a], shares, shares * fill)
        cash += gross - comm
        totals["commission"] += comm
        totals["slippage"] += shares * abs(raw_price - fill)
        if record_trades:
            trades.append({"asset": names[a], "side": "long" if sign > 0 else "short",
                           "entry_date": dates[pos["t"]], "exit_date": dates[t], "entry_price": entry_fill,
                           "exit_price": fill, "shares": shares, "bars_held": pos["hold"], "reason": reason,
                           "pnl": gross - comm - pos["entry_comm"]})

    prev_date = None
    for t in range(T):
        date = dates[t]
        if has_holding_cost and positions and prev_date is not None:
            days = max((date - prev_date).days, 1)
            for a, pos in positions.items():
                cost = td.daily_holding_cost(pos["shares"] * pos["entry"], names[a], days)
                cash -= cost
                totals["holding"] += cost
        prev_date = date

        # 1. exits decided at the previous close fill at today's open
        if positions:
            for a in [a for a, pos in positions.items() if pos["exit_flag"] and has_bar[a, t]]:
                close_position(a, O[a, t], positions[a]["exit_reason"], t)

        # 2. entries decided at the previous close fill at today's open, one market at a time
        if pending:
            todays = [a for a in pending if has_bar[a, t]]
            if len(todays) > 1:
                shift = t % len(todays)
                todays = todays[shift:] + todays[:shift]       # nobody is permanently first in the queue
            for a in todays:
                sign, atr_value = pending.pop(a)
                if len(positions) >= max_positions:
                    totals["skipped_max_positions"] += 1
                    continue
                if equity_prev <= 0 or atr_value <= 0:
                    continue
                raw = O[a, t]
                fill = td.entry_fill_price(raw, names[a]) if sign > 0 else td.exit_fill_price(raw, names[a])
                distance = stop_atr * atr_value
                shares = equity_prev * risk_frac / distance
                gross_now = sum(pos["shares"] * pos["last"] for pos in positions.values())
                allowed_notional = min(pos_cap * equity_prev, gross_cap * equity_prev - gross_now)
                if allowed_notional <= 0:
                    totals["skipped_room"] += 1
                    continue
                shares = min(shares, allowed_notional / fill)
                step = td.SIZE_STEP.get(names[a]) if td.SIZE_STEP else None
                if step:
                    shares = float(np.floor(shares / step + 1e-9) * step)
                    if shares <= 0:
                        totals["skipped_rounding"] += 1
                        continue
                if shares <= 1e-9:
                    continue
                comm = td.commission_for(names[a], shares, shares * fill)
                cash -= comm
                totals["commission"] += comm
                totals["slippage"] += shares * abs(raw - fill)
                positions[a] = {"sign": sign, "shares": shares, "entry": fill, "t": t, "hold": 0,
                                "stop": fill - sign * distance, "exit_flag": False, "exit_reason": "",
                                "entry_comm": comm, "last": fill}

        # 3. hard stops, active from the entry day; a gap through the level fills at the open
        if positions:
            for a in [a for a in positions if has_bar[a, t]]:
                pos = positions[a]
                if pos["sign"] > 0 and L[a, t] <= pos["stop"]:
                    close_position(a, O[a, t] if O[a, t] < pos["stop"] else pos["stop"], "stop", t)
                elif pos["sign"] < 0 and H[a, t] >= pos["stop"]:
                    close_position(a, O[a, t] if O[a, t] > pos["stop"] else pos["stop"], "stop", t)

        # 4. at the close: age every position, flag exits for tomorrow's open
        for a, pos in positions.items():
            if has_bar[a, t]:
                pos["last"] = C[a, t]
                pos["hold"] += 1
                if not pos["exit_flag"]:
                    if (long_out[a, t] if pos["sign"] > 0 else short_out[a, t]):
                        pos["exit_flag"], pos["exit_reason"] = True, "signal"
                    elif pos["hold"] >= max_hold:
                        pos["exit_flag"], pos["exit_reason"] = True, "time limit"

        # 5. new signals at this close become tomorrow's entries
        for a in np.flatnonzero(entry_any[:, t]):
            a = int(a)
            if a in positions or a in pending:
                continue
            pending[a] = (1, atr[a, t]) if long_in[a, t] else (-1, atr[a, t])

        unrealized = 0.0
        for pos in positions.values():
            unrealized += pos["sign"] * pos["shares"] * (pos["last"] - pos["entry"])
        equity[t] = cash + unrealized
        equity_prev = equity[t]

    # whatever is still open at the very end is closed at the last close
    for a in list(positions):
        last_t = int(np.flatnonzero(has_bar[a])[-1])
        close_position(a, C[a, last_t], "end of data", T - 1)
    equity[-1] = cash
    return {"equity_curve": pd.Series(equity, index=dates), "trades": trades, "starting_capital": starting_capital,
            "total_slippage_spread_cost": totals["slippage"], "total_commission_cost": totals["commission"],
            "total_holding_cost": totals["holding"], "skipped_room": totals["skipped_room"],
            "skipped_rounding": totals["skipped_rounding"], "skipped_max_positions": totals["skipped_max_positions"]}


# ----------------------------------------------------------------------------
# scoring and search
# ----------------------------------------------------------------------------
def score(data: MRData, p: dict, starting_capital: float = 100000.0, score_start=None, score_end=None,
          min_trades: int = 30) -> dict | None:
    """One trial: simulate, then score the equity slice [score_start, score_end]. None if too few trades."""
    result = simulate(data, p, starting_capital, record_trades=True)
    curve = result["equity_curve"].loc[score_start:score_end].dropna()
    if len(curve) < 60 or curve.iloc[0] <= 0:
        return None
    n_trades = sum(1 for t in result["trades"]
                   if (score_start is None or t["exit_date"] >= score_start) and (score_end is None or t["exit_date"] <= score_end))
    if n_trades < min_trades:
        return None
    stats = td.performance_stats(curve, float(curve.iloc[0]))
    if not np.isfinite(stats["Sharpe"]):
        return None
    return {"params": dict(p), "sharpe": float(stats["Sharpe"]), "cagr": float(stats["CAGR"]),
            "max_dd": float(stats["Max Drawdown"]), "calmar": float(stats["Calmar"]) if np.isfinite(stats["Calmar"]) else 0.0,
            "vol": float(stats["Ann. Vol"]), "n_trades": n_trades, "curve": curve}


def suggest(trial, space: dict = None, pinned: dict = None) -> dict:
    """Optuna define-by-run sampling: a setting is only suggested when the chosen entry/exit rule uses it."""
    space, pinned = space or PARAM_SPACE, pinned or PINNED

    def pick(name):
        values = space[name]
        if len(values) == 1:
            return values[0]
        return values[trial.suggest_int(name, 0, len(values) - 1)] if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) \
            else values[trial.suggest_categorical(name, list(range(len(values))))]

    p = dict(DEFAULT_PARAMS)
    p.update(pinned)
    p["direction"], p["entry_type"], p["exit_type"] = pick("direction"), pick("entry_type"), pick("exit_type")
    if p["entry_type"] == "rsi" or p["exit_type"] == "rsi":
        p["rsi_period"] = pick("rsi_period")
    if p["entry_type"] == "rsi":
        p["rsi_entry"] = pick("rsi_entry")
    elif p["entry_type"] == "nlow":
        p["nlow_days"] = pick("nlow_days")
    else:
        p["ibs_entry"] = pick("ibs_entry")
    if p["exit_type"] == "sma":
        p["exit_sma"] = pick("exit_sma")
    elif p["exit_type"] == "rsi":
        p["exit_rsi"] = pick("exit_rsi")
    for name in ("trend_sma", "max_hold", "atr_period", "stop_atr"):
        p[name] = pick(name)
    return p


def params_key(p: dict) -> tuple:
    return tuple(sorted(p.items()))


def search(data: MRData, n_trials: int, seed: int = 42, starting_capital: float = 100000.0,
           score_start=None, score_end=None, metric: str = "sharpe", min_trades: int = 30,
           space: dict = None, pinned: dict = None) -> list:
    """Optuna TPE search maximizing `metric` on the scored slice. Returns every distinct scored trial
    (dicts from score(), best first)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(15, n_trials // 5))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    seen, results = {}, []
    sims = [0]

    def objective(trial):
        p = suggest(trial, space, pinned)
        key = params_key(p)
        if key in seen:
            if seen[key] is None:
                raise optuna.TrialPruned()
            return seen[key]
        sims[0] += 1
        r = score(data, p, starting_capital, score_start, score_end, min_trades)
        if r is None:
            seen[key] = None
            raise optuna.TrialPruned()
        seen[key] = r[metric]
        results.append(r)
        return r[metric]

    def stop_when_done(study_, trial_):
        if sims[0] >= n_trials:
            study_.stop()

    study.optimize(objective, n_trials=n_trials * 4, callbacks=[stop_when_done], catch=())
    return sorted(results, key=lambda r: r[metric], reverse=True)
