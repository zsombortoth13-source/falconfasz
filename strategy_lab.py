"""Mechanism lab: a small single-market simulator for trying new ways of entering, sizing, stopping, scaling and exiting a long-only trend position.

It uses the engine's own cost model (turtle_dual_system.entry_fill_price / exit_fill_price / commission_for) and carry model (CARRY_* settings), the same
next-open fills and the same gap handling for stops, so a design written here can be checked against the engine on the baselines. Unlike the engine's one big
bar loop, every mechanic is a small component with a fixed hook, so designs are assembled from parts:

    Design(entry=..., sizer=..., exits=[...], stops=[...], trims=[...], adds=[...])

Each bar, in this order (mirroring the engine): carry accrues on yesterday's close; orders decided at yesterday's close fill at today's open; stops fill
intraday (at the level, or at the open when the market gapped through it); the account is marked at the close; the per-unit trackers update; then the
components look at the close and queue orders for the next open.

Hooks (all optional; see the component classes below):
    entry.signal(ctx)            while flat: a dict {"frac", "tag", ...} to open a position at the next open (or None)
    sizer.notional(ctx, frac)    the notional to buy for an entry or an add
    exit.fires(ctx, unit)        a close-based exit: sell the unit at the next open
    stop.level(ctx, unit)        an intraday stop level for the next bar (None = none); the highest of a unit's stops applies
    trim.fraction(ctx)           the share of the whole position to sell at the next open
    add.order(ctx)              {"notional", "tag"} to buy more at the next open (or None)
Money is tracked like the engine's: realized cash equity, plus open profit against each unit's fill price.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import turtle_dual_system as td


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
class Features:
    """Arrays for one market, computed once and shared by every component of every design run on it."""

    def __init__(self, df: pd.DataFrame, asset: str = "MKT"):
        self.df = df
        self.asset = asset
        self.dates = df.index
        self.o = df["Open"].to_numpy(dtype=float)
        self.h = df["High"].to_numpy(dtype=float)
        self.l = df["Low"].to_numpy(dtype=float)
        self.c = df["Close"].to_numpy(dtype=float)
        self.n = len(df)
        self._cache = {}

    def _get(self, key, fn):
        hit = self._cache.get(key)
        if hit is None:
            hit = fn()
            self._cache[key] = hit
        return hit

    def sma(self, n):
        return self._get(("sma", n), lambda: self.df["Close"].rolling(int(n)).mean().to_numpy())

    def ema(self, n):
        return self._get(("ema", n), lambda: self.df["Close"].ewm(span=int(n), adjust=False).mean().to_numpy())

    def atr(self, n):
        return self._get(("atr", n), lambda: td.compute_atr(self.df, int(n)).to_numpy())

    def highest_close(self, n):
        """Highest close of the previous n bars (today's excluded)."""
        return self._get(("hc", n), lambda: self.df["Close"].rolling(int(n)).max().shift(1).to_numpy())

    def lowest_low(self, n):
        return self._get(("ll", n), lambda: self.df["Low"].rolling(int(n)).min().shift(1).to_numpy())

    def efficiency_ratio(self, n):
        def fn():
            c = self.df["Close"]
            return ((c - c.shift(int(n))).abs() / c.diff().abs().rolling(int(n)).sum()).to_numpy()
        return self._get(("er", n), fn)

    def series(self, key, fn):
        """Any other feature a component needs: fn(df) -> array-like, cached under `key`."""
        return self._get(("custom", key), lambda: np.asarray(fn(self.df), dtype=float))


# ----------------------------------------------------------------------------
# state objects
# ----------------------------------------------------------------------------
class Unit:
    """One tranche of the position, with its own entry price, stop and trackers."""
    __slots__ = ("shares", "entry", "entry_i", "sig_i", "tag", "stop", "peak_high", "trough_low", "peak_close", "init_dist",
                 "closing", "episode", "state", "comm_in", "notional_in")

    def __init__(self, shares, entry, entry_i, tag, episode, comm_in):
        self.shares = shares
        self.entry = entry
        self.entry_i = entry_i
        self.sig_i = entry_i - 1
        self.tag = tag
        self.stop = None
        self.peak_high = entry
        self.trough_low = entry
        self.peak_close = entry
        self.init_dist = None      # the distance this unit risked at entry (set by a stop component), the unit's "R" in price terms
        self.closing = False
        self.episode = episode
        self.state = {}
        self.comm_in = comm_in
        self.notional_in = shares * entry


class Ctx:
    """What a component sees at the close of bar i."""

    def __init__(self, f: Features, capital: float):
        self.f = f
        self.i = 0
        self.units = []
        self.cash = capital          # realized equity (start capital + closed P&L - commissions + carry)
        self.episode = 0
        self.exit_i = None           # bar of the latest full exit
        self.exit_reason = None
        self.exit_price = None
        self.episode_start_i = None
        self.pending_entry = False

    # money, marked at the close of bar i
    def open_profit(self):
        c = self.f.c[self.i]
        return sum(u.shares * (c - u.entry) for u in self.units)

    def equity(self):
        return self.cash + self.open_profit()

    def notional(self):
        c = self.f.c[self.i]
        return sum(u.shares * c for u in self.units)

    def cost_basis(self):
        return sum(u.shares * u.entry for u in self.units)

    def exposure(self):
        eq = self.equity()
        return self.notional() / eq if eq > 0 else 0.0

    @property
    def flat(self):
        return not self.units

    def bars_since_exit(self):
        return None if self.exit_i is None else self.i - self.exit_i

    def units_tagged(self, tag):
        return [u for u in self.units if u.tag == tag]


class Component:
    """Base of every mechanic. `tags`: the unit tags an exit or stop applies to (None = every unit)."""
    name = "component"
    tags = None

    def bind(self, f: Features):
        """Called once per run, before the first bar: fetch arrays from the Features here."""

    def attach(self, design):
        """Called once per run with the whole Design, for a component that needs another one (an add that sizes like the entry does)."""

    def on_open(self, ctx: Ctx, unit: Unit):
        """A unit was just opened (filled)."""

    def on_episode_end(self, ctx: Ctx, reason: str):
        """The whole position was closed."""

    def applies(self, unit: Unit) -> bool:
        return self.tags is None or unit.tag in self.tags


@dataclass
class Design:
    entry: Component
    sizer: Component
    exits: list = field(default_factory=list)
    stops: list = field(default_factory=list)
    trims: list = field(default_factory=list)
    adds: list = field(default_factory=list)
    cap: float = None          # max cost-basis notional as a multiple of equity (the engine's MAX_TOTAL_EXPOSURE_PCT / 100); None = no cap
    entry_cap: float = None    # the cap for the first entry alone (None = same as cap); adds may take the position up to `cap`
    stop_slip: float = 0.0     # extra slippage on stop fills at the stop level, as a fraction of the level (real stops fill worse than the level in fast markets)
    name: str = ""

    def components(self):
        return [self.entry, self.sizer, *self.exits, *self.stops, *self.trims, *self.adds]


# ----------------------------------------------------------------------------
# result
# ----------------------------------------------------------------------------
class LabResult:
    def __init__(self, f, equity, exposure, units, capital, total_carry):
        self.f = f
        self.equity = pd.Series(equity, index=f.dates)
        self.exposure = pd.Series(exposure, index=f.dates)
        self.units = pd.DataFrame(units)
        self.capital = capital
        self.total_carry = total_carry

    def stats(self, a=None, b=None) -> dict:
        eq = self.equity.loc[a:b] if a else self.equity
        eq = eq.dropna()
        if len(eq) < 30:
            return {"CAGR": np.nan, "MaxDD": np.nan, "Sharpe": np.nan, "Calmar": np.nan}
        s = td.performance_stats(eq, float(eq.iloc[0]))
        return {"CAGR": s["CAGR"] * 100, "MaxDD": s["Max Drawdown"] * 100, "Sharpe": s["Sharpe"], "Calmar": s["Calmar"]}

    def episodes(self) -> pd.DataFrame:
        """One row per trade (episode): first fill to final exit, money-weighted."""
        if self.units.empty:
            return pd.DataFrame()
        u = self.units
        rows = []
        for ep, g in u.groupby("episode"):
            first, last = g["entry_i"].min(), g["exit_i"].max()
            pnl = g["pnl"].sum()
            cost = (g["shares"] * g["entry"]).sum()
            rows.append({"episode": ep, "entry_i": first, "exit_i": last, "entry_date": self.f.dates[first], "exit_date": self.f.dates[min(last, self.f.n - 1)],
                         "bars": last - first, "pnl": pnl, "ret": pnl / cost * 100 if cost else 0.0,
                         "units": len(g), "reason": g.sort_values("exit_i")["reason"].iloc[-1],
                         "mfe": g["mfe"].max(), "mae": g["mae"].min()})
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        s = self.stats()
        ep = self.episodes()
        out = dict(s)
        out["in_mkt"] = float((self.exposure > 0).mean() * 100)
        out["avg_expo"] = float(self.exposure.mean() * 100)
        out["episodes"] = int(len(ep))
        if len(ep):
            win, los = ep[ep.ret > 0], ep[ep.ret <= 0]
            out["win_rate"] = float(len(win) / len(ep) * 100)
            out["avg_win"] = float(win.ret.mean()) if len(win) else 0.0
            out["avg_loss"] = float(los.ret.mean()) if len(los) else 0.0
            out["short_trades_pct"] = float((ep.bars < 10).mean() * 100)
            top = ep.sort_values("pnl", ascending=False).head(10)["pnl"].sum()
            out["top10_share"] = float(top / win.pnl.sum() * 100) if len(win) and win.pnl.sum() > 0 else np.nan
        return out


# ----------------------------------------------------------------------------
# the simulator
# ----------------------------------------------------------------------------
def run_design(design: Design, f: Features, capital: float = 100_000.0, carry_rates=None, carry_days=None) -> LabResult:
    """Simulate `design` on one market. Carry (td.CARRY_MODE == "futures") is applied exactly as the engine does it."""
    n = f.n
    o, h, l, c = f.o, f.h, f.l, f.c
    asset = f.asset
    for comp in design.components():
        comp.bind(f)
        comp.attach(design)
    ctx = Ctx(f, capital)
    units = ctx.units
    pend = []                      # orders decided at the previous close, filled at this bar's open: tuples
    stop_buys = []                 # entry orders resting for this bar: dicts with a level
    equity = np.empty(n)
    exposure = np.zeros(n)
    closed = []
    carry_on = td.CARRY_MODE == "futures"
    if carry_on:
        if carry_rates is None:
            carry_rates = td.carry_rates_for(f.dates)
            carry_days = np.diff(f.dates.values).astype("timedelta64[D]").astype(float)
    div = td.CARRY_DIVIDEND_YIELD.get(asset, 0.0)
    total_carry = 0.0
    cap = design.cap

    def new_unit(shares, fill, i, tag, comm):
        u = Unit(shares, fill, i, tag, ctx.episode, comm)
        units.append(u)
        for comp in design.components():
            comp.on_open(ctx, u)
        # the unit's initial stops are live from its own bar, like the engine's, and may only use what was known at the signal close: the
        # components are asked as of that bar (ctx.i is moved back one bar while they answer)
        best = None
        here = ctx.i
        ctx.i = max(i - 1, 0)
        for st in design.stops:
            if st.applies(u):
                lv = st.level(ctx, u)
                if lv is not None and (best is None or lv > best):
                    best = lv
        ctx.i = here
        u.stop = best
        return u

    def close_unit(u, raw, i, reason, frac=1.0):
        """Sell `frac` of the unit at the raw price (exit costs applied). Returns True if the unit is gone."""
        sh = u.shares * frac
        fill = td.exit_fill_price(raw, asset)
        comm = td.commission_for(asset, sh, sh * fill)
        pnl = sh * (fill - u.entry)
        ctx.cash += pnl - comm
        closed.append({"episode": u.episode, "tag": u.tag, "entry_i": u.entry_i, "exit_i": i, "shares": sh, "entry": u.entry, "exit": fill,
                       "pnl": pnl - comm - u.comm_in * frac, "reason": reason,
                       "mfe": (u.peak_high / u.entry - 1) * 100, "mae": (u.trough_low / u.entry - 1) * 100})
        u.shares -= sh
        u.comm_in *= (1 - frac)
        return u.shares <= 1e-12

    def end_episode_if_flat(i, reason, price):
        if not units:
            ctx.exit_i, ctx.exit_reason, ctx.exit_price = i, reason, price
            for comp in design.components():
                comp.on_episode_end(ctx, reason)
            ctx.episode += 1
            ctx.episode_start_i = None

    for i in range(n):
        ctx.i = i
        # 0. carry for the interval since the previous bar, on the account as it closed then
        if carry_on and i > 0:
            ctx.i = i - 1
            eq_prev = ctx.equity()
            c_prev = c[i - 1]
            years = carry_days[i - 1] / 365.0
            r = carry_rates[i]
            amount = eq_prev * r * years + sum(u.shares * c_prev for u in units) * (div - r) * years
            ctx.cash += amount
            total_carry += amount
            ctx.i = i
        # 1. orders queued at the previous close fill at this open
        if pend:
            last_sell_reason = None
            for order in pend:
                kind = order[0]
                if kind == "buy":
                    _, notional, tag = order
                    fill = td.entry_fill_price(o[i], asset)
                    shares = notional / fill
                    if shares > 1e-12:
                        comm = td.commission_for(asset, shares, shares * fill)
                        ctx.cash -= comm
                        if ctx.episode_start_i is None:
                            ctx.episode_start_i = i
                        new_unit(shares, fill, i, tag, comm)
                    ctx.pending_entry = False
                elif kind == "sell":
                    _, u, reason = order
                    if u in units:
                        last_sell_reason = reason
                        if close_unit(u, o[i], i, reason):
                            units.remove(u)
                elif kind == "sell_frac":
                    _, u, frac, reason = order
                    if u in units:
                        last_sell_reason = reason
                        if close_unit(u, o[i], i, reason, frac):
                            units.remove(u)
            if not units and last_sell_reason is not None:
                end_episode_if_flat(i, last_sell_reason, o[i])
            pend = []
        # 1b. resting stop-buy entry orders (filled intraday at the level, or at the open if it gapped above it)
        if stop_buys:
            for order in stop_buys:
                lvl = order["level"]
                if h[i] >= lvl and not units:
                    raw = o[i] if o[i] >= lvl else lvl
                    fill = td.entry_fill_price(raw, asset)
                    shares = order["notional"] / fill
                    comm = td.commission_for(asset, shares, shares * fill)
                    ctx.cash -= comm
                    ctx.episode_start_i = i
                    u = new_unit(shares, fill, i, order["tag"], comm)
                    u.sig_i = i - 1
            stop_buys = []
            ctx.pending_entry = False
        # 2. intraday stops
        for u in list(units):
            lvl = u.stop
            if lvl is not None and l[i] <= lvl:
                raw = o[i] if o[i] < lvl else lvl * (1.0 - design.stop_slip)
                if close_unit(u, raw, i, "stop"):
                    units.remove(u)
                    end_episode_if_flat(i, "stop", raw)
        # 3. mark at the close
        op = sum(u.shares * (c[i] - u.entry) for u in units)
        equity[i] = ctx.cash + op
        exposure[i] = (sum(u.shares * c[i] for u in units) / equity[i]) if units and equity[i] > 0 else 0.0
        # 4. trackers
        for u in units:
            if h[i] > u.peak_high:
                u.peak_high = h[i]
            if l[i] < u.trough_low:
                u.trough_low = l[i]
            if c[i] > u.peak_close:
                u.peak_close = c[i]
        # 5. signals at this close, for the next open
        if i >= n - 1:
            continue
        if units:
            for ex in design.exits:
                for u in units:
                    if not u.closing and ex.applies(u) and ex.fires(ctx, u):
                        u.closing = True
                        pend.append(("sell", u, getattr(ex, "name", "exit")))
            live = [u for u in units if not u.closing]
            for u in live:
                # the stop for the next bar is the highest level any of the unit's stop components asks for (None if none does); a
                # component that must never loosen keeps its own ratchet
                best = None
                for st in design.stops:
                    if st.applies(u):
                        lv = st.level(ctx, u)
                        if lv is not None and (best is None or lv > best):
                            best = lv
                u.stop = best
            if live:
                for tr in design.trims:
                    frac = tr.fraction(ctx)
                    if frac and frac > 1e-9:
                        for u in live:
                            if tr.applies(u):
                                pend.append(("sell_frac", u, min(frac, 1.0), getattr(tr, "name", "trim")))
                for ad in design.adds:
                    order = ad.order(ctx)
                    if order and order["notional"] > 1e-9:
                        notional = order["notional"]
                        if cap is not None:
                            room = cap * ctx.equity() - ctx.cost_basis() - sum(p[1] for p in pend if p[0] == "buy")
                            notional = min(notional, room)
                        if notional > 1e-9:
                            pend.append(("buy", notional, order.get("tag", "add")))
        elif not ctx.pending_entry and not (ctx.exit_i == i and ctx.exit_reason == "stop"):
            # (like the engine, no new entry is considered on the bar a stop fired on: the earliest is the next close)
            sig = design.entry.signal(ctx)
            if sig:
                eq = ctx.equity()
                notional = design.sizer.notional(ctx, sig.get("frac", 1.0))
                ecap = design.entry_cap if design.entry_cap is not None else cap
                if ecap is not None:
                    notional = min(notional, ecap * eq)
                if notional > 1e-9:
                    if sig.get("level") is not None:
                        stop_buys.append({"level": sig["level"], "notional": notional, "tag": sig.get("tag", "core")})
                    else:
                        pend.append(("buy", notional, sig.get("tag", "core")))
                    ctx.pending_entry = True
    # the final bar: mark any open units out at the last close, like the engine's end-of-data close
    for u in list(units):
        close_unit(u, c[n - 1], n - 1, "end_of_data")
        units.remove(u)
    equity[n - 1] = ctx.cash
    return LabResult(f, equity, exposure, closed, capital, total_carry)


# ----------------------------------------------------------------------------
# baseline components (enough to reproduce the engine's regime and channel systems)
# ----------------------------------------------------------------------------
class RegimeEntry(Component):
    """Enter while flat when the close is above its n-day simple average (+ band %), once the ATR exists (the engine's condition)."""
    name = "regime_entry"

    def __init__(self, n=250, band_pct=0.0, frac=1.0, tag="core", atr_n=20, confirm=1, slope_bars=0, wait_after_stop=0, above_stop_price=False):
        """confirm: the number of consecutive closes above the line required; slope_bars: if > 0 the average must also be higher than that many bars ago;
        wait_after_stop: bars to stay out after a stop exit; above_stop_price: after a stop exit, also wait for a close above the price the stop filled at."""
        self.n, self.band, self.frac, self.tag, self.atr_n = n, band_pct / 100.0, frac, tag, atr_n
        self.confirm, self.slope_bars = int(confirm), int(slope_bars)
        self.wait, self.above_stop = int(wait_after_stop), bool(above_stop_price)

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)

    def signal(self, ctx):
        i = ctx.i
        ma = self.ma[i]
        if np.isnan(ma) or np.isnan(self.atr[i]) or self.atr[i] <= 0:
            return None
        c = self.f.c
        if ctx.exit_reason == "stop" and ctx.exit_i is not None:
            if i - ctx.exit_i < self.wait:
                return None
            if self.above_stop and c[i] <= ctx.exit_price:
                return None
        if self.confirm > 1:
            if i < self.confirm:
                return None
            for j in range(self.confirm):
                m = self.ma[i - j]
                if np.isnan(m) or c[i - j] <= m * (1 + self.band):
                    return None
        elif c[i] <= ma * (1 + self.band):
            return None
        if self.slope_bars and (i < self.slope_bars or not ma > self.ma[i - self.slope_bars]):
            return None
        return {"frac": self.frac, "tag": self.tag}


class StopBuyRegimeEntry(RegimeEntry):
    """E4. The regime entry, but while the close is still under the line a stop-buy rests at the line for the next bar: an intraday cross is bought at
    the level (or at the open, if the market gapped above it) instead of waiting for a close above the line and the open after that."""
    name = "stopbuy_regime_entry"

    def signal(self, ctx):
        base = super().signal(ctx)
        if base:
            return base
        i = ctx.i
        ma = self.ma[i]
        if np.isnan(ma) or np.isnan(self.atr[i]) or self.atr[i] <= 0:
            return None
        if ctx.exit_reason == "stop" and ctx.exit_i is not None and (i - ctx.exit_i < self.wait or (self.above_stop and self.f.c[i] <= ctx.exit_price)):
            return None
        return {"frac": self.frac, "tag": self.tag, "level": ma * (1 + self.band)}


class TwoSpeedEntry(RegimeEntry):
    """E3. The regime entry, plus a FAST entry (unit tag "fast") on an n-day closing high while the close is still under the regime line. The fast
    unit is meant to carry its own tighter exit and stop (tags), so it can catch the start of a move without holding the regime's patience."""
    name = "two_speed_entry"

    def __init__(self, n=250, band_pct=0.0, fast_n=20, fast_frac=0.5, fast_buffer=0.0, **kw):
        super().__init__(n, band_pct, **kw)
        self.fast_n, self.fast_frac, self.fast_buffer = fast_n, fast_frac, fast_buffer

    def bind(self, f):
        super().bind(f)
        self.hc = f.highest_close(self.fast_n)

    def signal(self, ctx):
        base = super().signal(ctx)
        if base:
            return base
        i = ctx.i
        lvl, ma, a = self.hc[i], self.ma[i], self.atr[i]
        if np.isnan(lvl) or np.isnan(ma) or np.isnan(a) or a <= 0:
            return None
        if ctx.exit_reason == "stop" and ctx.exit_i is not None and i - ctx.exit_i < self.wait:
            return None
        if self.f.c[i] > lvl + self.fast_buffer * a:
            return {"frac": self.fast_frac, "tag": "fast"}
        return None


class BreakoutEntry(Component):
    """Enter while flat on a close above the highest close of the previous n bars (+ buffer x ATR)."""
    name = "breakout_entry"

    def __init__(self, n=20, buffer_atr=0.0, frac=1.0, tag="core", atr_n=20):
        self.n, self.buffer, self.frac, self.tag, self.atr_n = n, buffer_atr, frac, tag, atr_n

    def bind(self, f):
        self.f, self.hc, self.atr = f, f.highest_close(self.n), f.atr(self.atr_n)

    def signal(self, ctx):
        i = ctx.i
        lvl, a = self.hc[i], self.atr[i]
        if np.isnan(lvl) or np.isnan(a) or a <= 0:
            return None
        if self.f.c[i] > lvl + self.buffer * a:
            return {"frac": self.frac, "tag": self.tag}
        return None


class ExposureSizer(Component):
    """Buy `exposure` x equity (x the entry's fraction x every scaler), measured at the signal close."""
    name = "exposure_sizer"

    def __init__(self, exposure=1.5, scalers=()):
        self.exposure, self.scalers = exposure, list(scalers)

    def bind(self, f):
        for s in self.scalers:
            s.bind(f)

    def notional(self, ctx, frac=1.0):
        m = 1.0
        for s in self.scalers:
            m *= s.value(ctx)
        return self.exposure * frac * m * ctx.equity()


class AtrRiskSizer(Component):
    """The engine's sizing: risk_pct of equity at stake over a stop k x ATR away, so notional = risk_pct x equity x price / (k x ATR).
    Calm markets give big positions and turbulent ones small ones; the design's cap limits the result. (The engine reads the next open as
    the price; this reads the signal close, a difference of a fraction of a percent.)"""
    name = "atr_risk_sizer"

    def __init__(self, risk_pct=20.0, k=10.0, atr_n=20, scalers=()):
        self.risk_pct, self.k, self.atr_n, self.scalers = risk_pct, k, atr_n, list(scalers)

    def bind(self, f):
        self.f, self.atr = f, f.atr(self.atr_n)
        for s in self.scalers:
            s.bind(f)

    def notional(self, ctx, frac=1.0):
        a = self.atr[ctx.i]
        if np.isnan(a) or a <= 0:
            return 0.0
        m = 1.0
        for s in self.scalers:
            m *= s.value(ctx)
        return frac * m * self.risk_pct / 100.0 * ctx.equity() * self.f.c[ctx.i] / (self.k * a)


class RegimeExit(Component):
    """Sell when the close is below its n-day average (- band %)."""
    name = "exit_channel"

    def __init__(self, n=250, band_pct=0.0, tags=None):
        self.n, self.band, self.tags = n, band_pct / 100.0, tags

    def bind(self, f):
        self.f, self.ma = f, f.sma(self.n)

    def fires(self, ctx, unit):
        ma = self.ma[ctx.i]
        return (not np.isnan(ma)) and self.f.c[ctx.i] < ma * (1 - self.band)


class ChannelExit(Component):
    """Sell when the close is below the lowest low of the previous n bars."""
    name = "exit_channel"

    def __init__(self, n=100, tags=None):
        self.n, self.tags = n, tags

    def bind(self, f):
        self.f, self.ll = f, f.lowest_low(self.n)

    def fires(self, ctx, unit):
        lvl = self.ll[ctx.i]
        return (not np.isnan(lvl)) and self.f.c[ctx.i] < lvl


class AtrStop(Component):
    """A fixed stop k x ATR under the unit's fill (the ATR of the signal bar), the engine's initial stop."""
    name = "stop"

    def __init__(self, k=10.0, atr_n=20, tags=None):
        self.k, self.atr_n, self.tags = k, atr_n, tags

    def bind(self, f):
        self.atr = f.atr(self.atr_n)

    def level(self, ctx, unit):
        lv = unit.state.get("atr_stop")
        if lv is None:
            dist = self.k * self.atr[unit.sig_i]
            if unit.init_dist is None:
                unit.init_dist = dist
            lv = unit.entry - dist
            unit.state["atr_stop"] = lv
        return lv
