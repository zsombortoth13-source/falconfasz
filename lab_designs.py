"""Design factories for the mechanism lab: each is factory(L, **params) -> Design, where L scales the whole design's exposure (position sizes,
the cap, any exposure target) so lab_research.calibrate can find the leverage that fits a drawdown cap. Registered by name so jobs can run in
worker processes. Every design starts from the regime baseline (in above the n-day average, out below it, engine-style ATR risk sizing with a
cap, a wide ATR stop) and changes one mechanic, so its result can be read against the baseline."""
from lab_mechanics import (AcceleratingTrail, BandExit, BreakoutAdd, DistanceScaler, ErScaler, ExtensionReAdd, ExtensionState, ExtensionTrim,
                           HouseMoneyAdd, InvalidationSizer, MaStop, ProbationStop, ProofAdd, PullbackAdd, StaticTarget, TargetRefill, TargetTrim,
                           TieredTrail, VolScaler, VolTarget)
from lab_research import register
from strategy_lab import (AtrRiskSizer, AtrStop, BreakoutEntry, ChannelExit, Design, RegimeEntry, RegimeExit, StopBuyRegimeEntry,
                          TwoSpeedEntry)

BASE = dict(n=250, band=0.0, risk=20.0, k=10.0, cap=1.5, stop_k=10.0)


def _p(params):
    return {**BASE, **params}


# ---------------------------------------------------------------- baselines
@register("regime")
def regime(L, n=250, band=0.0, risk=20.0, k=10.0, cap=1.5, stop_k=10.0):
    """The baseline: in while the close is above its n-day average (+/- band %), engine-style ATR risk sizing capped at `cap` x equity, a wide
    stop `stop_k` x ATR away. At L = 1 this is the 250-day regime rule at a 150% cap that the engine produced."""
    return Design(entry=RegimeEntry(n, band), sizer=AtrRiskSizer(risk * L, k), exits=[RegimeExit(n, band)], stops=[AtrStop(stop_k)],
                  cap=cap * L, name="regime")


@register("channel")
def channel(L, entry_n=20, exit_n=100, buffer=0.5, risk=20.0, k=10.0, cap=2.0, stop_k=10.0):
    """The fast-in / slow-out channel rule."""
    return Design(entry=BreakoutEntry(entry_n, buffer_atr=buffer), sizer=AtrRiskSizer(risk * L, k), exits=[ChannelExit(exit_n)],
                  stops=[AtrStop(stop_k)], cap=cap * L, name="channel")


# ---------------------------------------------------------------- S: stops and exits
@register("S1_probation")
def s1_probation(L, tight_pct=3.0, days=20, proof_pct=6.0, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=[RegimeExit(b["n"], b["band"])],
                  stops=[AtrStop(b["stop_k"]), ProbationStop(tight_pct, days, proof_pct)], cap=b["cap"] * L, name="S1_probation")


@register("S2_mastop")
def s2_mastop(L, k_atr=0.0, **base):
    """An intraday stop at the average itself, instead of waiting for the close below it."""
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=[RegimeExit(b["n"], b["band"])],
                  stops=[AtrStop(b["stop_k"]), MaStop(b["n"], k_atr)], cap=b["cap"] * L, name="S2_mastop")


@register("R1_invalid")
def r1_invalid(L, risk=15.0, k_atr=0.0, min_dist_atr=2.0, **base):
    """Invalidation-based sizing: risk a fixed share of equity between the price and the average."""
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=InvalidationSizer(risk * L, b["n"], k_atr, min_dist_atr),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])], cap=b["cap"] * L, name="R1_invalid")


@register("S3_bandexit")
def s3_bandexit(L, k_atr=1.0, **base):
    """Out only when the close is k ATRs below the average (instead of any close below it); in on any close above it."""
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=[BandExit(b["n"], k_atr)],
                  stops=[AtrStop(b["stop_k"])], cap=b["cap"] * L, name="S3_bandexit")


@register("S4_tiered")
def s4_tiered(L, start=12.0, t1=8.0, t2=5.0, t3=3.5, g1=6, g2=12, g3=24, keep_regime_exit=True, **base):
    b = _p(base)
    exits = [RegimeExit(b["n"], b["band"])] if keep_regime_exit else []
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=exits,
                  stops=[AtrStop(b["stop_k"]), TieredTrail(((0, start), (g1, t1), (g2, t2), (g3, t3)))], cap=b["cap"] * L, name="S4_tiered")


@register("T5_acceltrail")
def t5_acceltrail(L, start=12.0, step=0.5, every=20, floor=4.0, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=[RegimeExit(b["n"], b["band"])],
                  stops=[AtrStop(b["stop_k"]), AcceleratingTrail(start, step, every, floor)], cap=b["cap"] * L, name="T5_acceltrail")


# ---------------------------------------------------------------- T / R: exposure targeting and trims
@register("T1_target")
def t1_target(L, tmult=1.0, band=0.3, min_gap=5, **base):
    """Hold the exposure near tmult x the cap: trim after run-ups, refill after drops (profit-taking and pyramiding as one rule)."""
    b = _p(base)
    tgt = StaticTarget(b["cap"] * L * tmult)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=[RegimeExit(b["n"], b["band"])],
                  stops=[AtrStop(b["stop_k"])], trims=[TargetTrim(tgt, band)], adds=[TargetRefill(tgt, band, min_gap)], cap=b["cap"] * L * 1.5,
                  name="T1_target")


@register("T2_exttrim")
def t2_exttrim(L, k_hi=5.0, frac=0.33, k_lo=2.0, **base):
    b = _p(base)
    st = ExtensionState()
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"]), exits=[RegimeExit(b["n"], b["band"])],
                  stops=[AtrStop(b["stop_k"])], trims=[ExtensionTrim(st, k_hi, frac, b["n"])], adds=[ExtensionReAdd(st, k_lo, b["n"])],
                  cap=b["cap"] * L, name="T2_exttrim")


@register("R2_voltarget")
def r2_voltarget(L, vol_target=0.15, window=20, floor=0.2, band=0.3, min_gap=5, **base):
    """Exposure held at cap x clip(vol target / realized vol): de-levers in turbulence (never levers above the cap)."""
    b = _p(base)
    tgt = VolTarget(b["cap"] * L, vol_target, window, floor, 1.0)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"], scalers=[VolScaler(vol_target, window, floor)]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])], trims=[TargetTrim(tgt, band)],
                  adds=[TargetRefill(tgt, band, min_gap)], cap=b["cap"] * L, name="R2_voltarget")


# ---------------------------------------------------------------- P: pyramiding
@register("P1_house")
def p1_house(L, entry_frac=0.5, unit_frac=0.5, cushion=1.0, max_units=3, min_gap=10, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"], frac=entry_frac), sizer=AtrRiskSizer(b["risk"] * L, b["k"]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])],
                  adds=[HouseMoneyAdd(unit_frac, cushion, max_units, min_gap, b["stop_k"])], cap=b["cap"] * L, name="P1_house")


@register("P3_pullback")
def p3_pullback(L, entry_frac=0.5, unit_frac=0.5, k_hi=3.0, k_lo=1.0, max_adds=3, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"], frac=entry_frac), sizer=AtrRiskSizer(b["risk"] * L, b["k"]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])],
                  adds=[PullbackAdd(unit_frac, k_hi, k_lo, max_adds, b["n"])], cap=b["cap"] * L, name="P3_pullback")


@register("P5_coresat")
def p5_coresat(L, core_frac=0.6, sat_frac=0.4, sat_n=20, sat_exit_n=10, max_units=3, min_gap=5, **base):
    """A patient regime core plus satellite units that join on fresh highs and leave on a short channel low."""
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"], frac=core_frac, tag="core"), sizer=AtrRiskSizer(b["risk"] * L, b["k"]),
                  exits=[RegimeExit(b["n"], b["band"]), ChannelExit(sat_exit_n, tags=["sat"])], stops=[AtrStop(b["stop_k"])],
                  adds=[BreakoutAdd(sat_n, sat_frac, max_units, min_gap, "sat")], cap=b["cap"] * L, name="P5_coresat")


@register("P6_starter")
def p6_starter(L, entry_frac=0.4, proof_pct=6.0, days=None, **base):
    """Open with a starter, buy the rest once the position has proved itself (+proof_pct% over its fill, or `days` bars)."""
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"], frac=entry_frac), sizer=AtrRiskSizer(b["risk"] * L, b["k"]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])],
                  adds=[ProofAdd(1.0 - entry_frac, proof_pct, days)], cap=b["cap"] * L, name="P6_starter")


# ---------------------------------------------------------------- composite: any combination of the mechanics above
@register("combo")
def combo(L, base=None, trail=None, const_trail=None, starter=None, house=None, pullback=None, coresat=None, probation=None, mastop=None,
          invalid=None, target=None, vol=None, exttrim=None, confirm=1, bandexit=None, regime_exit=True, stop_slip=0.0, reentry=None,
          entry_cap=None, total_cap=None, entry_mode=None, fast=None, **ignored):
    """The regime baseline plus any of the mechanics, each given as a dict of its own parameters (or None to leave it out):
    trail (S4 tiered trail: start, t1, t2, t3, g1, g2, g3), const_trail ({"dist": ATRs}: one fixed-distance trail, the control for the tiers),
    starter (P6: entry_frac, proof_pct, days), house (P1: entry_frac, unit_frac, cushion, max_units, min_gap), pullback (P3), coresat (P5),
    probation (S1), mastop (S2), invalid (R1), target (T1), vol (R2), exttrim (T2), bandexit (S3). `regime_exit` False leaves the trail as the only exit."""
    b = _p(base or {})
    n, band = b["n"], b["band"]
    entry_frac, tag = 1.0, "core"
    for part in (starter, house, pullback):
        if part:
            entry_frac = min(entry_frac, part.get("entry_frac", 0.5))
    if coresat:
        entry_frac = coresat.get("core_frac", 0.6)
    scalers = [VolScaler(vol.get("vol_target", 0.15), vol.get("window", 20), vol.get("floor", 0.2))] if vol else []
    sizer = (InvalidationSizer(invalid.get("risk", 15.0) * L, n, invalid.get("k_atr", 0.0), invalid.get("min_dist_atr", 2.0), scalers=scalers)
             if invalid else AtrRiskSizer(b["risk"] * L, b["k"], scalers=scalers))
    two_speed = entry_mode == "twospeed"
    core_tags = ["core"] if two_speed else None
    exits = [RegimeExit(n, band, tags=core_tags)] if regime_exit else []
    if bandexit:
        exits = [BandExit(n, bandexit.get("k_atr", 1.0))]
    stops = [AtrStop(b["stop_k"], tags=core_tags)]
    if two_speed:
        fa = fast or {}
        exits.append(ChannelExit(fa.get("fast_exit", 10), tags=["fast"]))
        stops.append(AtrStop(fa.get("fast_stop_k", 4.0), tags=["fast"]))
    if trail:
        stops.append(TieredTrail(((0, trail.get("start", 10.0)), (trail.get("g1", 6), trail.get("t1", 9.0)), (trail.get("g2", 12), trail.get("t2", 7.0)),
                                  (trail.get("g3", 24), trail.get("t3", 5.0)))))
    if const_trail:
        stops.append(TieredTrail(((0, const_trail["dist"]),)))
    if probation:
        stops.append(ProbationStop(probation.get("tight_pct", 4.0), probation.get("days", 20), probation.get("proof_pct", 8.0)))
    if mastop:
        stops.append(MaStop(n, mastop.get("k_atr", 1.0)))
    trims, adds = [], []
    cap = b["cap"] * L
    if target:
        tgt = StaticTarget(b["cap"] * L * target.get("tmult", 1.0))
        trims.append(TargetTrim(tgt, target.get("band", 0.3)))
        adds.append(TargetRefill(tgt, target.get("band", 0.3), target.get("min_gap", 5)))
        cap = b["cap"] * L * 1.5
    if vol:
        tgt = VolTarget(b["cap"] * L, vol.get("vol_target", 0.15), vol.get("window", 20), vol.get("floor", 0.2), 1.0)
        trims.append(TargetTrim(tgt, vol.get("band", 0.3)))
        adds.append(TargetRefill(tgt, vol.get("band", 0.3), vol.get("min_gap", 5)))
    if exttrim:
        st = ExtensionState()
        trims.append(ExtensionTrim(st, exttrim.get("k_hi", 6.0), exttrim.get("frac", 0.25), n))
        adds.append(ExtensionReAdd(st, exttrim.get("k_lo", 3.0), n))
    if starter:
        adds.append(ProofAdd(1.0 - entry_frac, starter.get("proof_pct", 3.0), starter.get("days")))
    if house:
        adds.append(HouseMoneyAdd(house.get("unit_frac", 0.5), house.get("cushion", 1.0), house.get("max_units", 3), house.get("min_gap", 10), b["stop_k"]))
    if pullback:
        adds.append(PullbackAdd(pullback.get("unit_frac", 0.5), pullback.get("k_hi", 3.0), pullback.get("k_lo", 1.0), pullback.get("max_adds", 3), n))
    if coresat:
        exits.append(ChannelExit(coresat.get("sat_exit_n", 10), tags=["sat"]))
        adds.append(BreakoutAdd(coresat.get("sat_n", 20), coresat.get("sat_frac", 0.4), coresat.get("max_units", 3), coresat.get("min_gap", 5), "sat"))
    re = reentry or {}
    if total_cap is not None:
        cap = total_cap * L
    ekw = dict(frac=entry_frac, tag=tag, confirm=confirm, wait_after_stop=re.get("wait", 0), above_stop_price=re.get("above_stop", False))
    if entry_mode == "stopbuy":
        entry = StopBuyRegimeEntry(n, band, **ekw)
    elif two_speed:
        fa = fast or {}
        entry = TwoSpeedEntry(n, band, fast_n=fa.get("fast_n", 20), fast_frac=fa.get("fast_frac", 0.5), **ekw)
    else:
        entry = RegimeEntry(n, band, **ekw)
    return Design(entry=entry, sizer=sizer, exits=exits, stops=stops, trims=trims,
                  adds=adds, cap=cap, entry_cap=(entry_cap * L if entry_cap is not None else None), stop_slip=stop_slip, name="combo")


# ---------------------------------------------------------------- E: entries and filters (as size, never gates)
@register("E_confirm")
def e_confirm(L, confirm=3, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"], confirm=confirm), sizer=AtrRiskSizer(b["risk"] * L, b["k"]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])], cap=b["cap"] * L, name="E_confirm")


@register("E_slope")
def e_slope(L, slope_bars=20, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"], slope_bars=slope_bars), sizer=AtrRiskSizer(b["risk"] * L, b["k"]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])], cap=b["cap"] * L, name="E_slope")


@register("E_er")
def e_er(L, n=20, lo=0.1, hi=0.4, floor=0.3, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"], scalers=[ErScaler(n, lo, hi, floor)]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])], cap=b["cap"] * L, name="E_er")


@register("E_dist")
def e_dist(L, full_atr=3.0, floor=0.4, **base):
    b = _p(base)
    return Design(entry=RegimeEntry(b["n"], b["band"]), sizer=AtrRiskSizer(b["risk"] * L, b["k"], scalers=[DistanceScaler(b["n"], full_atr, floor)]),
                  exits=[RegimeExit(b["n"], b["band"])], stops=[AtrStop(b["stop_k"])], cap=b["cap"] * L, name="E_dist")
