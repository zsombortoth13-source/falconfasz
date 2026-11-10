"""New mechanics for the mechanism lab (stops, exits, trims, adds, sizers, scalers). Each is a small component with the hooks described in
strategy_lab.py; lab_designs.py assembles them into designs. Names follow the research plan: S = stop, T = take-profit / trim, P = pyramiding,
R = risk model, E = entry and filters."""
import numpy as np

from strategy_lab import Component


# ----------------------------------------------------------------------------
# S: stops and exits
# ----------------------------------------------------------------------------
class ProbationStop(Component):
    """S1. A tight stop that exists only while a unit is on probation: it is dropped once the unit has gained `proof_pct`% over its fill or has
    lived `days` bars. After that only the unit's other stops (the wide one, the regime exit) apply: tight to prove, loose to run."""
    name = "probation_stop"

    def __init__(self, tight_pct=3.0, days=20, proof_pct=6.0, tags=None):
        self.tight, self.days, self.proof, self.tags = tight_pct, days, proof_pct, tags

    def level(self, ctx, unit):
        if unit.state.get("proven"):
            return None
        if (unit.peak_high / unit.entry - 1) * 100 >= self.proof or (ctx.i - unit.entry_i) >= self.days:
            unit.state["proven"] = True
            return None
        return unit.entry * (1 - self.tight / 100.0)


class MaStop(Component):
    """S2. An intraday stop at the thesis level: the n-day average (less k x ATR). It acts the moment price touches it instead of waiting for a close."""
    name = "stop"

    def __init__(self, n=250, k_atr=0.0, band_pct=0.0, atr_n=20, tags=None):
        self.n, self.k, self.band, self.atr_n, self.tags = n, k_atr, band_pct / 100.0, atr_n, tags

    def bind(self, f):
        self.ma, self.atr = f.sma(self.n), f.atr(self.atr_n)

    def level(self, ctx, unit):
        i = ctx.i
        ma, a = self.ma[i], self.atr[i]
        if np.isnan(ma) or np.isnan(a):
            return None
        return ma * (1 - self.band) - self.k * a


class InvalidationSizer(Component):
    """R1. Size so that being wrong costs `risk_pct` of equity: notional = risk_pct x equity x price / (distance to the n-day average less k x ATR),
    with the distance floored at `min_dist_atr` x ATR. Big when price hugs the average (cheap risk), small when it is extended."""
    name = "invalidation_sizer"

    def __init__(self, risk_pct=15.0, n=250, k_atr=0.0, min_dist_atr=2.0, atr_n=20, scalers=()):
        self.risk, self.n, self.k, self.min_dist, self.atr_n, self.scalers = risk_pct, n, k_atr, min_dist_atr, atr_n, list(scalers)

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)
        for s in self.scalers:
            s.bind(f)

    def notional(self, ctx, frac=1.0):
        i = ctx.i
        ma, a, px = self.ma[i], self.atr[i], self.f.c[i]
        if np.isnan(ma) or np.isnan(a) or a <= 0:
            return 0.0
        dist = max(px - (ma - self.k * a), self.min_dist * a)
        m = 1.0
        for s in self.scalers:
            m *= s.value(ctx)
        return frac * m * self.risk / 100.0 * ctx.equity() * px / dist


class BandExit(Component):
    """S3. Sell on a close below the n-day average less k x ATR: a band that widens in volatile markets and narrows in calm ones."""
    name = "exit_channel"

    def __init__(self, n=250, k_atr=1.0, atr_n=20, tags=None):
        self.n, self.k, self.atr_n, self.tags = n, k_atr, atr_n, tags

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)

    def fires(self, ctx, unit):
        i = ctx.i
        ma, a = self.ma[i], self.atr[i]
        return (not np.isnan(ma)) and (not np.isnan(a)) and self.f.c[i] < ma - self.k * a


class TieredTrail(Component):
    """S4 / T4. A trailing stop from the unit's highest high whose distance (in ATRs) shrinks as the unit's open profit grows. `tiers` is a list
    of (profit in ATRs of the unit's entry, trail distance in ATRs); the first tier is the starting distance. The level never falls."""
    name = "trail"

    def __init__(self, tiers=((0, 12.0), (6, 8.0), (12, 5.0), (24, 3.5)), atr_n=20, tags=None):
        self.tiers = sorted(tiers)
        self.atr_n, self.tags = atr_n, tags

    def bind(self, f):
        self.atr = f.atr(self.atr_n)

    def level(self, ctx, unit):
        i = ctx.i
        a = self.atr[i]
        a0 = self.atr[unit.sig_i]
        if np.isnan(a) or np.isnan(a0) or a0 <= 0:
            return None
        gain_atr = (unit.peak_high - unit.entry) / a0
        mult = self.tiers[0][1]
        for thr, m in self.tiers:
            if gain_atr >= thr:
                mult = m
        lv = unit.peak_high - mult * a
        prev = unit.state.get("trail")
        if prev is not None and prev > lv:
            lv = prev
        unit.state["trail"] = lv
        return lv


class AcceleratingTrail(Component):
    """T5. A trail from the highest high that tightens with the unit's age: starts `start_atr` ATRs away and loses `step_atr` per `every` bars, down to `floor_atr`."""
    name = "trail"

    def __init__(self, start_atr=12.0, step_atr=0.5, every=20, floor_atr=4.0, atr_n=20, tags=None):
        self.start, self.step, self.every, self.floor, self.atr_n, self.tags = start_atr, step_atr, every, floor_atr, atr_n, tags

    def bind(self, f):
        self.atr = f.atr(self.atr_n)

    def level(self, ctx, unit):
        a = self.atr[ctx.i]
        if np.isnan(a):
            return None
        mult = max(self.floor, self.start - self.step * ((ctx.i - unit.entry_i) // self.every))
        lv = unit.peak_high - mult * a
        prev = unit.state.get("atrail")
        if prev is not None and prev > lv:
            lv = prev
        unit.state["atrail"] = lv
        return lv


# ----------------------------------------------------------------------------
# T: exposure targeting, trims
# ----------------------------------------------------------------------------
class StaticTarget:
    """A constant target exposure (market value / equity)."""

    def __init__(self, target):
        self.target = target

    def bind(self, f):
        pass

    def value(self, ctx):
        return self.target


class VolTarget:
    """Target exposure = `base` x clip(vol_target / realized vol, floor, ceil), realized volatility being the annualized std of the last `window`
    daily returns of the market (yesterday's, no look-ahead). With ceil = 1 it only ever de-levers."""

    def __init__(self, base, vol_target=0.15, window=20, floor=0.2, ceil=1.0):
        self.base, self.vt, self.window, self.floor, self.ceil = base, vol_target, window, floor, ceil

    def bind(self, f):
        import pandas as pd
        r = pd.Series(f.c).pct_change()
        self.vol = (r.rolling(self.window).std() * np.sqrt(252)).to_numpy()

    def value(self, ctx):
        v = self.vol[ctx.i]
        if np.isnan(v) or v <= 0:
            return self.base
        return self.base * min(max(self.vt / v, self.floor), self.ceil)


class TargetTrim(Component):
    """T1 (sell half). When exposure drifts above `band` x the target, sell the excess at the next open."""
    name = "target_trim"

    def __init__(self, target, band=0.25):
        self.target, self.band = target, band

    def bind(self, f):
        self.target.bind(f)

    def fraction(self, ctx):
        exp = ctx.exposure()
        tgt = self.target.value(ctx)
        if tgt > 0 and exp > tgt * (1 + self.band):
            return 1.0 - tgt / exp
        return 0.0


class TargetRefill(Component):
    """T1 / P2 (buy half). When exposure falls below the target by more than `band`, buy back up to it (once `min_gap` bars have passed since the last refill)."""
    name = "target_refill"

    def __init__(self, target, band=0.25, min_gap=5, tag="refill"):
        self.target, self.band, self.min_gap, self.tag = target, band, min_gap, tag
        self.last_i = -10**9

    def bind(self, f):
        self.target.bind(f)
        self.last_i = -10**9

    def on_episode_end(self, ctx, reason):
        self.last_i = -10**9

    def order(self, ctx):
        if any(u.closing for u in ctx.units):
            return None
        exp = ctx.exposure()
        tgt = self.target.value(ctx)
        if tgt > 0 and exp < tgt * (1 - self.band) and ctx.i - self.last_i >= self.min_gap:
            self.last_i = ctx.i
            return {"notional": (tgt - exp) * ctx.equity(), "tag": self.tag}
        return None


class ExtensionState:
    """What the extension trim sold and not yet bought back, shared by its two halves."""

    def __init__(self):
        self.trimmed_notional = 0.0
        self.armed = True


class ExtensionTrim(Component):
    """T2 (sell half). When price is more than `k_hi` ATRs above the n-day average, sell `frac` of the position (once per extension)."""
    name = "extension_trim"

    def __init__(self, state, k_hi=5.0, frac=0.33, n=250, atr_n=20):
        self.s, self.k_hi, self.frac, self.n, self.atr_n = state, k_hi, frac, n, atr_n

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)

    def on_episode_end(self, ctx, reason):
        self.s.trimmed_notional, self.s.armed = 0.0, True

    def fraction(self, ctx):
        i = ctx.i
        ma, a = self.ma[i], self.atr[i]
        if np.isnan(ma) or np.isnan(a) or a <= 0:
            return 0.0
        ext = (self.f.c[i] - ma) / a
        if self.s.armed and ext >= self.k_hi:
            self.s.armed = False
            self.s.trimmed_notional += self.frac * ctx.notional()
            return self.frac
        return 0.0


class ExtensionReAdd(Component):
    """T2 (buy half). Buy back what the trim sold once price has come back within `k_lo` ATRs of the average (and is still above it)."""
    name = "extension_readd"

    def __init__(self, state, k_lo=2.0, n=250, atr_n=20):
        self.s, self.k_lo, self.n, self.atr_n = state, k_lo, n, atr_n

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)

    def order(self, ctx):
        i = ctx.i
        ma, a = self.ma[i], self.atr[i]
        if np.isnan(ma) or np.isnan(a) or a <= 0 or any(u.closing for u in ctx.units):
            return None
        ext = (self.f.c[i] - ma) / a
        if not self.s.armed and self.s.trimmed_notional > 0 and 0 < ext <= self.k_lo:
            amount, self.s.trimmed_notional, self.s.armed = self.s.trimmed_notional, 0.0, True
            return {"notional": amount, "tag": "readd"}
        if not self.s.armed and ext <= self.k_lo:
            self.s.armed = True
        return None


# ----------------------------------------------------------------------------
# P: pyramiding
# ----------------------------------------------------------------------------
class _AddBase(Component):
    """Adds that size themselves as a share of what the entry would buy."""

    def attach(self, design):
        self.design = design

    def base_notional(self, ctx):
        return self.design.sizer.notional(ctx, 1.0)

    def count(self, ctx, tag=None):
        return len(ctx.units) if tag is None else len(ctx.units_tagged(tag))


class HouseMoneyAdd(_AddBase):
    """P1. Add a unit (`unit_frac` of the base size) only when open profit already covers `cushion` x the risk of the new unit, so the account
    never has more at stake than it started with. The risk of a unit is its stop distance (`stop_k` ATRs)."""
    name = "house_money_add"

    def __init__(self, unit_frac=0.5, cushion=1.0, max_units=4, min_gap=10, stop_k=10.0, atr_n=20, tag="add"):
        self.unit_frac, self.cushion, self.max_units, self.min_gap, self.stop_k, self.atr_n, self.tag = unit_frac, cushion, max_units, min_gap, stop_k, atr_n, tag
        self.last_i = -10**9

    def bind(self, f):
        self.f, self.atr = f, f.atr(self.atr_n)
        self.last_i = -10**9

    def on_episode_end(self, ctx, reason):
        self.last_i = -10**9

    def order(self, ctx):
        if len(ctx.units) >= self.max_units or any(u.closing for u in ctx.units) or ctx.i - self.last_i < self.min_gap:
            return None
        i = ctx.i
        a, px = self.atr[i], self.f.c[i]
        if np.isnan(a) or a <= 0:
            return None
        notional = self.unit_frac * self.base_notional(ctx)
        risk_new = notional * self.stop_k * a / px
        if ctx.open_profit() >= self.cushion * risk_new:
            self.last_i = i
            return {"notional": notional, "tag": self.tag}
        return None


class PullbackAdd(_AddBase):
    """P3. After price has run `k_hi` ATRs above the n-day average, add a unit when it pulls back to within `k_lo` ATRs of it (still above): adds inside the trend, on weakness."""
    name = "pullback_add"

    def __init__(self, unit_frac=0.5, k_hi=3.0, k_lo=1.0, max_adds=3, n=250, atr_n=20, tag="add"):
        self.unit_frac, self.k_hi, self.k_lo, self.max_adds, self.n, self.atr_n, self.tag = unit_frac, k_hi, k_lo, max_adds, n, atr_n, tag
        self.armed, self.adds = False, 0

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)
        self.armed, self.adds = False, 0

    def on_episode_end(self, ctx, reason):
        self.armed, self.adds = False, 0

    def order(self, ctx):
        i = ctx.i
        ma, a = self.ma[i], self.atr[i]
        if np.isnan(ma) or np.isnan(a) or a <= 0 or any(u.closing for u in ctx.units):
            return None
        ext = (self.f.c[i] - ma) / a
        if ext >= self.k_hi:
            self.armed = True
        elif self.armed and 0 < ext <= self.k_lo and self.adds < self.max_adds:
            self.armed = False
            self.adds += 1
            return {"notional": self.unit_frac * self.base_notional(ctx), "tag": self.tag}
        return None


class BreakoutAdd(_AddBase):
    """P5. A satellite unit on a fresh `n`-day high while the core is held, at most `max_units` in all, `min_gap` bars apart."""
    name = "breakout_add"

    def __init__(self, n=20, unit_frac=0.5, max_units=3, min_gap=5, tag="sat"):
        self.n, self.unit_frac, self.max_units, self.min_gap, self.tag = n, unit_frac, max_units, min_gap, tag
        self.last_i = -10**9

    def bind(self, f):
        self.f, self.hc = f, f.highest_close(self.n)
        self.last_i = -10**9

    def on_episode_end(self, ctx, reason):
        self.last_i = -10**9

    def order(self, ctx):
        i = ctx.i
        lv = self.hc[i]
        if np.isnan(lv) or len(ctx.units) >= self.max_units or any(u.closing for u in ctx.units) or i - self.last_i < self.min_gap:
            return None
        if self.f.c[i] > lv:
            self.last_i = i
            return {"notional": self.unit_frac * self.base_notional(ctx), "tag": self.tag}
        return None


class ProofAdd(_AddBase):
    """P6 / E1. After a starter entry, buy the rest (`rest_frac` of the base size) once the position has earned `proof_pct`% over its fill or lived `days` bars."""
    name = "proof_add"

    def __init__(self, rest_frac=0.5, proof_pct=5.0, days=None, tag="rest"):
        self.rest, self.proof, self.days, self.tag = rest_frac, proof_pct, days, tag
        self.done = False

    def bind(self, f):
        self.done = False

    def on_episode_end(self, ctx, reason):
        self.done = False

    def order(self, ctx):
        if self.done or not ctx.units or any(u.closing for u in ctx.units):
            return None
        u = ctx.units[0]
        gained = (u.peak_high / u.entry - 1) * 100 >= self.proof
        aged = self.days is not None and (ctx.i - u.entry_i) >= self.days
        if gained or aged:
            self.done = True
            return {"notional": self.rest * self.base_notional(ctx), "tag": self.tag}
        return None


# ----------------------------------------------------------------------------
# E: size scalers (filters expressed as size, never as gates)
# ----------------------------------------------------------------------------
class ErScaler:
    """Scale new size by the efficiency ratio of the last n closes (|net change| / sum of |changes|): trendy tape gets full size, noisy tape `floor`."""

    def __init__(self, n=20, lo=0.1, hi=0.4, floor=0.3):
        self.n, self.lo, self.hi, self.floor = n, lo, hi, floor

    def bind(self, f):
        self.er = f.efficiency_ratio(self.n)

    def value(self, ctx):
        e = self.er[ctx.i]
        if np.isnan(e):
            return 1.0
        return min(1.0, max(self.floor, self.floor + (1 - self.floor) * (e - self.lo) / max(self.hi - self.lo, 1e-9)))


class DistanceScaler:
    """Scale new size by how far the close sits above the n-day average, in ATRs: right at the average (the chop zone) `floor`, `full_atr` or more full size."""

    def __init__(self, n=250, full_atr=3.0, floor=0.4, atr_n=20):
        self.n, self.full, self.floor, self.atr_n = n, full_atr, floor, atr_n

    def bind(self, f):
        self.f, self.ma, self.atr = f, f.sma(self.n), f.atr(self.atr_n)

    def value(self, ctx):
        i = ctx.i
        ma, a = self.ma[i], self.atr[i]
        if np.isnan(ma) or np.isnan(a) or a <= 0:
            return 1.0
        ext = (self.f.c[i] - ma) / a
        return min(1.0, max(self.floor, self.floor + (1 - self.floor) * ext / self.full))


class VolScaler:
    """Scale new size by vol_target / realized vol (clipped to [floor, 1]): turbulent markets get smaller entries."""

    def __init__(self, vol_target=0.15, window=20, floor=0.25):
        self.vt, self.window, self.floor = vol_target, window, floor

    def bind(self, f):
        import pandas as pd
        r = pd.Series(f.c).pct_change()
        self.vol = (r.rolling(self.window).std() * np.sqrt(252)).to_numpy()

    def value(self, ctx):
        v = self.vol[ctx.i]
        if np.isnan(v) or v <= 0:
            return 1.0
        return min(1.0, max(self.floor, self.vt / v))
