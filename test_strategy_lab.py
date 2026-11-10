"""Tests for the mechanism lab and the research harness.

Run:  .venv/Scripts/python.exe test_strategy_lab.py     (no pytest needed; it also collects under pytest)
The lab must agree with the engine on the baseline systems (same trades on the same bars, same money to within a fraction of a percent), because every
research conclusion rests on it; the harness pieces (leverage calibration, paired comparison, walk-forward leverage) are checked on small cases.
"""
import copy
import sys

import numpy as np
import pandas as pd

import lab_designs  # noqa: F401  (registers the factories)
import lab_research as lr
import turtle_dual_system as td
from strategy_lab import (AtrRiskSizer, AtrStop, Design, ExposureSizer, Features, RegimeEntry, RegimeExit, run_design)
from test_engine_realism import Settings, clean_engine, synthetic_market

KEYS = ("SYSTEMS", "MAX_TOTAL_EXPOSURE_PCT", "USE_PORTFOLIO_RISK_CAP", "STOP_SLIPPAGE_PCT", "CARRY_MODE", "CARRY_RATE_SERIES", "CARRY_RATE_FLAT",
        "CARRY_DIVIDEND_YIELD", "MARK_TO_MARKET", "EXECUTION_TIMING", "SLIPPAGE_PCT", "SPREAD_PCT", "COMMISSION_PCT", "ATH_MAX_DISTANCE_PCT")


def _wavy_market(n=1500, seed=5):
    """A drifting market with long swings, so a 50-day regime rule trades a handful of times."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ret = 0.0004 + 0.012 * rng.standard_normal(n) + 0.0035 * np.sin(2 * np.pi * t / 400)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + 0.002 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + np.abs(0.004 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.004 * rng.standard_normal(n)))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0}, index=pd.bdate_range("2010-01-01", periods=n))
    df.attrs["asset"] = "WAVY"
    return df


def _engine_vs_lab(raw, use_trail, slip_pct):
    """Run the same regime system in the engine and in the lab; return (engine entry dates, lab entry dates, engine final, lab final)."""
    with Settings():
        clean_engine()
        td.MAX_TOTAL_EXPOSURE_PCT = 150
        td.USE_PORTFOLIO_RISK_CAP = False
        td.USE_DRAWDOWN_SCALING = False   # the engine's default account adjustment is not a lab mechanic
        td.STOP_SLIPPAGE_PCT = slip_pct
        system = {**td.SYSTEMS[0], "name": "S2", "use_ma_regime": True, "ma_regime_period": 50, "ma_regime_band_pct": 0.0, "use_peak_trail": use_trail,
                  "peak_trail_atr_mult": 6.0, "n_atr_period": 20, "risk_per_trade_pct": 20.0, "stop_n_mult": 10.0, "max_units_per_trend": 1,
                  "use_scale_out": False, "use_time_stop": False, "use_cooldown": False, "use_ema_filter": False, "use_n_day_low_trailing_stop": False,
                  "use_chandelier_exit": False, "use_breakeven_stop": False, "use_giveback_stop": False}
        td.SYSTEMS = [system]
        feats = td.build_features(raw)
        feats.attrs["asset"] = "WAVY"
        eng = td.simulate_portfolio({"WAVY": feats}, starting_capital=100_000.0)
        design = Design(entry=RegimeEntry(50, 0.0), sizer=AtrRiskSizer(20.0, 10.0), exits=[RegimeExit(50, 0.0)],
                        stops=[AtrStop(10.0)] + ([lab_trail()] if use_trail else []), cap=1.5, stop_slip=slip_pct / 100.0)
        lab = run_design(design, Features(raw, "WAVY"), 100_000.0)
    idx = raw.index
    nxt = lambda d: idx[min(idx.get_loc(d) + 1, len(idx) - 1)]      # the engine dates a trade by its signal bar, the lab by its fill bar
    e_in = [nxt(t["entry_dates"][0]) for t in eng["trades_by_asset"]["WAVY"]]
    l_in = list(lab.episodes().entry_date)
    return e_in, l_in, eng["final_equity"], lab.equity.iloc[-1], eng, lab


def lab_trail():
    from lab_mechanics import TieredTrail
    return TieredTrail(((0, 6.0),))


def test_lab_matches_engine_on_regime_and_trail():
    raw = _wavy_market()
    for use_trail, slip in ((False, 0.0), (True, 0.0), (True, 0.5)):
        e_in, l_in, e_final, l_final, eng, lab = _engine_vs_lab(raw, use_trail, slip)
        assert len(e_in) >= 4, "the test market should produce several trades"
        assert e_in == l_in, (use_trail, slip, e_in[:5], l_in[:5])
        assert abs(l_final / e_final - 1) < 0.01, (use_trail, slip, l_final, e_final)


def test_lab_carry_matches_engine_when_idle():
    index = pd.bdate_range("2018-01-01", periods=300)
    flat = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0}, index=index)
    with Settings():
        clean_engine()
        td.CARRY_MODE, td.CARRY_RATE_SERIES, td.CARRY_RATE_FLAT, td.CARRY_DIVIDEND_YIELD = "futures", None, 0.05, {}
        design = Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0), exits=[RegimeExit(50, 0.0)], stops=[AtrStop(10.0)], cap=1.0)
        lab = run_design(design, Features(flat, "FLAT"), 100_000.0)
    expected = 100_000.0
    for gap in np.diff(index.values).astype("timedelta64[D]").astype(float):
        expected *= 1 + 0.05 * gap / 365.0
    assert abs(lab.equity.iloc[-1] - expected) < 1e-6 and abs(lab.total_carry - (expected - 100_000.0)) < 1e-6


def test_hooks_do_what_they_say():
    """A stop fills at its level (not the open) unless the market gapped; an entry fills at the next open; the exit and its price are as set."""
    from lab_mechanics import ProbationStop
    raw = _wavy_market()
    f = Features(raw, "WAVY")
    with Settings():
        clean_engine()
        base = Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0), exits=[RegimeExit(50, 0.0)], stops=[AtrStop(10.0)], cap=1.0)
        tight = Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0), exits=[RegimeExit(50, 0.0)],
                       stops=[AtrStop(10.0), ProbationStop(tight_pct=1.0, days=30, proof_pct=50.0)], cap=1.0)
        res_b, res_t = run_design(base, f, 100_000.0), run_design(tight, f, 100_000.0)
    u = res_t.units
    stops = u[u.reason == "stop"]
    assert len(stops) > 0 and len(res_t.episodes()) > len(res_b.episodes())    # a 1% probation stop fires and cuts attempts short
    for _, r in stops.head(20).iterrows():
        o, lo = f.o[r.exit_i], f.l[r.exit_i]
        assert lo <= r["exit"] / (1 - (td.SLIPPAGE_PCT + td.SPREAD_PCT / 2) / 100) + 1e-6       # the fill is within the bar's range
        assert r["exit"] <= o * (1 + 1e-9)                                                        # never above the open: a stop sells at the level or lower
    first = res_b.units.iloc[0]
    assert abs(first["entry"] - td.entry_fill_price(f.o[int(first.entry_i)], "WAVY")) < 1e-9      # fills at that bar's open, with the entry costs


def test_calibration_respects_the_cap_and_scales_leverage():
    raw = _wavy_market()
    f = Features(raw, "WAVY")
    with Settings():
        clean_engine()
        fac = lambda L: Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0 * L), exits=[RegimeExit(50, 0.0)], stops=[AtrStop(10.0)], cap=1.0 * L)
        L40, r40 = lr.calibrate(fac, f, 40.0)
        L20, r20 = lr.calibrate(fac, f, 20.0)
    assert -r40.stats()["MaxDD"] <= 40.0 + 1e-9 and -r20.stats()["MaxDD"] <= 20.0 + 1e-9
    assert L40 > L20 and r40.stats()["CAGR"] >= r20.stats()["CAGR"]                 # a looser cap allows more leverage and more return
    # the same market with its worst gap days opening at their close can only be worse for a long position held through them
    stressed = lr.stress_gaps(f, -0.01)
    with Settings():
        clean_engine()
        assert run_design(fac(1.0), stressed).stats()["MaxDD"] <= run_design(fac(1.0), f).stats()["MaxDD"] + 1e-9


def test_paired_comparison_and_thresholds():
    idx = pd.bdate_range("2000-01-01", "2020-12-31")
    rng = np.random.default_rng(2)
    base_ret = pd.Series(0.0003 + 0.01 * rng.standard_normal(len(idx)), index=idx)
    base = (1 + base_ret).cumprod() * 100_000
    better = (1 + base_ret + 0.0002).cumprod() * 100_000                              # +0.02% a day on every day, about +5% a year
    same = lr.paired_delta(base, base)
    assert abs(same["delta_cagr"]) < 1e-9
    up = lr.paired_delta(better, base)
    assert up["delta_cagr"] > 4.0 and up["t"] > 3.0 and up["p_pos"] > 0.99
    assert lr.t_threshold(10) < lr.t_threshold(1000) < 4.0 and abs(lr.t_threshold(2) - (2 * np.log(2)) ** 0.5) < 1e-12
    lv = lr.levered_buy_and_hold(Features(_wavy_market(), "WAVY"), 2.0, carry=False)
    unl = lr.levered_buy_and_hold(Features(_wavy_market(), "WAVY"), 1.0, carry=False)
    assert abs(lv.pct_change().iloc[5] / unl.pct_change().iloc[5] - 2.0) < 1e-9


def test_walk_forward_leverage_uses_only_earlier_years():
    raw = _wavy_market(n=3000)
    with Settings():
        clean_engine()
        fac = lambda L, **p: Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0 * L), exits=[RegimeExit(50, 0.0)], stops=[AtrStop(10.0)], cap=1.0 * L)
        out = lr.walk_forward_leverage(fac, {}, raw, "WAVY", 40.0, min_train_years=3)
        assert out["stats"]["years"] >= 5 and len(out["L"]) == out["stats"]["years"]
        # truncating the future leaves every earlier year's leverage unchanged: it was fitted without it
        cut = lr.walk_forward_leverage(fac, {}, raw.loc[:"2017-12-31"], "WAVY", 40.0, min_train_years=3)
    common = [y for y in cut["L"] if y in out["L"] and y < max(cut["L"])]
    assert common and all(abs(cut["L"][y] - out["L"][y]) < 1e-12 for y in common)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:   # noqa: BLE001 -- report every failure, not just the first
            failed += 1
            import traceback
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
