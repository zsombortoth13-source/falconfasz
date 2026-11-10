"""More tests for the mechanism lab: fills, trailing stops, the gap stress, stop-buy entries, and lab vs engine on a synthetic market.
Run:  .venv/Scripts/python.exe test_lab_mechanics.py   (test_strategy_lab.py holds the other lab tests: carry, calibration, paired comparison, walk-forward leverage)"""
import sys

import numpy as np

import lab_mechanics as lm
import lab_research as lr
import turtle_dual_system as td
from strategy_lab import AtrRiskSizer, AtrStop, Design, ExposureSizer, Features, RegimeEntry, RegimeExit, StopBuyRegimeEntry, run_design
from test_engine_realism import Settings, clean_engine, synthetic_market


def _market(crash=-0.3, seed=3):
    _, raw = synthetic_market(crash=crash, seed=seed)
    return raw


def test_lab_matches_engine_on_a_synthetic_market():
    raw = _market()
    with Settings():
        clean_engine()
        td.CARRY_MODE = "off"
        td.MAX_TOTAL_EXPOSURE_PCT = 150
        td.USE_PORTFOLIO_RISK_CAP = False
        system = {**td.SYSTEMS[0], "name": "S2", "use_ma_regime": True, "ma_regime_period": 50, "ma_regime_band_pct": 0.0, "risk_per_trade_pct": 20.0,
                  "stop_n_mult": 10.0, "max_units_per_trend": 1, "n_atr_period": 20, "use_scale_out": False, "use_time_stop": False, "use_cooldown": False,
                  "use_ema_filter": False, "use_n_day_low_trailing_stop": False, "use_breakeven_stop": False, "use_chandelier_exit": False,
                  "use_giveback_stop": False, "use_regime_scale_out": False}
        td.SYSTEMS = [system]
        feats = td.build_features(raw)
        feats.attrs["asset"] = "TEST"
        eng = td.simulate_portfolio({"TEST": feats}, starting_capital=100_000.0)
        design = Design(entry=RegimeEntry(50, 0.0), sizer=AtrRiskSizer(20.0, 10.0), exits=[RegimeExit(50, 0.0)], stops=[AtrStop(10)], cap=1.5)
        lab = run_design(design, Features(raw, "TEST"))
        trades = eng["trades_by_asset"]["TEST"]
        ep = lab.episodes()
        assert len(trades) == len(ep) >= 1
        idx = raw.index
        for t, (_, e) in zip(trades, ep.iterrows()):
            assert idx[idx.get_loc(t["entry_dates"][0]) + 1] == e["entry_date"]          # the engine dates a trade by its signal bar, the lab by its fill bar
        assert abs(lab.equity.iloc[-1] / eng["final_equity"] - 1) < 0.01


def test_stop_fills_at_level_or_gap_open():
    raw = _market(crash=-0.35)
    with Settings():
        clean_engine()
        td.CARRY_MODE = "off"
        f = Features(raw, "TEST")
        atr = f.atr(20)
        # an initial stop 3 ATR under the fill: filled at the level when the day traded down through it
        res = run_design(Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0), exits=[], stops=[AtrStop(3.0)], cap=1.0), f)
        stops = res.units[res.units.reason == "stop"]
        assert len(stops) >= 1
        for _, r in stops.iterrows():
            i, sig = int(r["exit_i"]), int(r["entry_i"]) - 1
            level = r["entry"] - 3.0 * atr[sig]
            raw_price = f.o[i] if f.o[i] < level else level
            assert abs(r["exit"] - td.exit_fill_price(raw_price, "TEST")) < 1e-9 * r["exit"]
        # a trailing stop just under the peak is gapped through by the crash: filled at the (much lower) open, not at the level
        res2 = run_design(Design(entry=RegimeEntry(50, 0.0), sizer=ExposureSizer(1.0), exits=[], stops=[lm.TieredTrail(((0, 3.0),))], cap=1.0), f)
        crash_bar = int(np.argmin(f.o[1:] / f.c[:-1])) + 1
        hit = res2.units[(res2.units.reason == "stop") & (res2.units.exit_i == crash_bar)]
        assert len(hit) == 1
        assert abs(hit["exit"].iloc[0] - td.exit_fill_price(f.o[crash_bar], "TEST")) < 1e-9 * hit["exit"].iloc[0]
        assert hit["exit"].iloc[0] < 0.7 * f.c[crash_bar - 1]


def test_tiered_trail_never_falls_and_tightens():
    raw = _market()
    f = Features(raw, "TEST")
    trail = lm.TieredTrail(((0, 10.0), (4, 6.0)))
    trail.bind(f)
    unit = type("U", (), {})()
    unit.entry, unit.sig_i, unit.state, unit.peak_high = 100.0, 40, {}, 100.0
    ctx = type("C", (), {})()
    levels = []
    for i, ph in zip(range(41, 60), np.linspace(100, 140, 19)):
        ctx.i, unit.peak_high = i, ph
        levels.append(trail.level(ctx, unit))
    assert all(b >= a - 1e-9 for a, b in zip(levels, levels[1:]))                          # a trailing stop only ever rises
    assert unit.peak_high - levels[-1] <= 6.0 * f.atr(20)[59] + 1e-6                        # after 4 ATRs of profit the distance is the tighter tier


def test_stress_gaps_opens_at_the_close():
    raw = _market(crash=-0.3)
    f = Features(raw, "TEST")
    s = lr.stress_gaps(f)
    bad = raw["Close"].pct_change() <= -0.04
    assert bad.sum() >= 1
    assert np.allclose(s.df.loc[bad, "Open"].values, raw.loc[bad, "Close"].values)
    assert (s.df["High"] >= s.df[["Open", "Close"]].max(axis=1) - 1e-12).all()


def test_stop_buy_entry_fills_at_the_level():
    raw = _market()
    with Settings():
        clean_engine()
        td.CARRY_MODE = "off"
        design = Design(entry=StopBuyRegimeEntry(50, 0.0), sizer=ExposureSizer(1.0), exits=[RegimeExit(50, 0.0)], stops=[], cap=1.0)
        res = run_design(design, Features(raw, "TEST"))
        ma = raw["Close"].rolling(50).mean()
        u = res.units
        assert len(u) >= 1
        stop_buy_fills = 0
        for _, r in u.iterrows():
            i = int(r["entry_i"])
            level = ma.iloc[i - 1]
            if raw["Close"].iloc[i - 1] <= level and raw["High"].iloc[i] >= level:           # an intraday cross: bought at the level or the gap-up open
                stop_buy_fills += 1
                assert abs(r["entry"] / td.entry_fill_price(max(raw["Open"].iloc[i], level), "TEST") - 1) < 1e-6
        assert stop_buy_fills >= 1


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    lr.configure(None, False)
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:   # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
