"""Hand-checkable tests for the account / broker realism features of turtle_dual_system.

Run:  .venv/Scripts/python.exe test_engine_realism.py     (no pytest needed; it also collects under pytest)

Each test builds a small synthetic market where the right answer can be worked out on paper, then
checks the engine against it: fee arithmetic, whole-contract rounding, next-open fills, margin calls,
and that the books balance (start capital + trade P&L - every cost = final equity).
"""
import copy
import sys

import numpy as np
import pandas as pd

import brokers
import turtle_dual_system as td

SETTINGS = ("SLIPPAGE_PCT", "SPREAD_PCT", "COMMISSION_PCT", "HOLDING_COST_PCT_PER_YEAR", "USE_PER_CONTRACT_FEES",
            "FEE_SCHEDULE", "ASSET_COSTS", "SIZE_STEP", "TAKE_MIN_ONE_UNIT", "USE_MARGIN_CALLS", "MARGIN_SPEC",
            "MARK_TO_MARKET", "EXECUTION_TIMING", "MAX_TOTAL_EXPOSURE_PCT", "PORTFOLIO_STARTING_CAPITAL",
            "USE_PORTFOLIO_RISK_CAP", "MAX_PORTFOLIO_RISK_PCT", "SYSTEMS", "MAX_GROSS_EXPOSURE_PCT", "VOL_TARGET_PCT",
            "VOL_TARGET_WINDOW", "VOL_TARGET_FLOOR", "VOL_TARGET_CAP", "HITRATE_WINDOW", "HITRATE_MIN_AVG_R", "HITRATE_SCALE",
            "ASSET_RISK_MULT", "ASSET_RISK_NORMALIZE", "ALLOW_EXIT_ABOVE_ENTRY", "CARRY_MODE", "CARRY_RATE_SERIES",
            "CARRY_RATE_FLAT", "CARRY_DIVIDEND_YIELD", "STOP_SLIPPAGE_PCT", "USE_DRAWDOWN_SCALING",
            "DRAWDOWN_SCALING_THRESHOLD", "DRAWDOWN_SCALING_FACTOR", "DRAWDOWN_SCALING_REFERENCE",
            "TAIL_WINDOW", "TAIL_MIN_BARS", "TAIL_FALLBACK", "GAP_SIZING_K", "STRESS_TOTAL_CAP_PCT", "STRESS_SINGLE_CAP_PCT",
            "STRESS_CLUSTER_CAP_PCT", "MARKET_CLUSTERS", "BOOK_VOL_CAP_PCT", "BOOK_VOL_WINDOW", "EQUITY_MA_WINDOW",
            "EQUITY_MA_SCALE", "SHOCK_PAUSE_LOSS_PCT", "SHOCK_PAUSE_DAYS", "EXTENSION_MA", "EXTENSION_N_MULT",
            "EXTENSION_RISK_MULT", "DAILY_LOSS_STOP_N_MULT", "DAILY_LOSS_STOP_PCT", "ENTRY_TOP_K", "ENTRY_STRENGTH_LOOKBACK",
            "ENTRY_MIN_STRENGTH", "ENTRY_RANK_FIRST_UNIT_ONLY", "VOL_RISK_GAMMA", "VOL_RISK_ATR", "VOL_RISK_REF", "VOL_RISK_MIN",
            "VOL_RISK_MAX", "ENTRY_EMA_PERIOD", "ENTRY_EMA_MODE", "ENTRY_EMA_EXIT", "MH_CLEAN_MODE")


class Settings:
    """Context manager: snapshot the engine's globals, restore them afterwards."""
    def __enter__(self):
        self.saved = {name: copy.deepcopy(getattr(td, name)) for name in SETTINGS}
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(td, name, value)


def synthetic_market(crash: float = 0.0, seed: int = 3, n_flat: int = 300, n_up: int = 250, n_after: int = 120):
    """Flat noisy market, a steady 0.4%/day climb, then optionally one overnight gap of `crash`
    (e.g. -0.35 = -35%) followed by a flat stretch. Returns a features DataFrame for asset 'TEST'."""
    rng = np.random.default_rng(seed)
    close = [100.0]
    for _ in range(n_flat):
        close.append(close[-1] * (1 + rng.normal(0, 0.002)))
    for _ in range(n_up):
        close.append(close[-1] * 1.004)
    peak = close[-1]
    gap = peak * (1 + crash)
    for k in range(n_after):
        close.append(gap * (1 + rng.normal(0, 0.002)))
    close = np.array(close)
    open_ = np.concatenate([[close[0]], close[:-1]])
    if crash:
        open_[n_flat + n_up + 1] = gap   # the crash happens overnight, between two closes
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    index = pd.bdate_range("2015-01-01", periods=len(close))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0}, index=index)
    df.attrs["asset"] = "TEST"
    return td.build_features(df), df


def clean_engine():
    td.USE_PER_CONTRACT_FEES = False
    td.FEE_SCHEDULE, td.ASSET_COSTS, td.SIZE_STEP, td.MARGIN_SPEC = {}, {}, {}, {}
    td.TAKE_MIN_ONE_UNIT = False
    td.USE_MARGIN_CALLS = False
    td.MARK_TO_MARKET = True
    td.EXECUTION_TIMING = "next_open"
    td.SLIPPAGE_PCT, td.SPREAD_PCT, td.COMMISSION_PCT, td.HOLDING_COST_PCT_PER_YEAR = 0.02, 0.02, 0.01, 0.0
    td.MAX_TOTAL_EXPOSURE_PCT = 300
    td.USE_PORTFOLIO_RISK_CAP = True
    td.MAX_PORTFOLIO_RISK_PCT = 20.0


def all_units(result):
    return [u for trades in result["trades_by_asset"].values() for t in trades for u in t["unit_details"]]


# ----------------------------------------------------------------------------
def test_fee_arithmetic():
    with Settings():
        clean_engine()
        td.USE_PER_CONTRACT_FEES = True
        td.FEE_SCHEDULE = {"TEST": {"multiplier": 5.0, "fee_per_contract": 0.62, "code": "MES"}}
        # 40 engine shares of a multiplier-5 contract = 8 contracts, at $0.62 each
        assert abs(td.commission_for("TEST", 40.0, 40 * 6000.0) - 8 * 0.62) < 1e-9
        # an asset with no schedule entry keeps the percentage commission
        assert abs(td.commission_for("OTHER", 40.0, 10_000.0) - 10_000 * 0.01 / 100) < 1e-9
        # per-asset spread and slippage replace the globals only for that asset
        td.ASSET_COSTS = {"TEST": {"slippage_pct": 0.1, "spread_pct": 0.2}}
        assert abs(td.entry_fill_price(1000.0, "TEST") - 1000 * (1 + (0.1 + 0.1) / 100)) < 1e-9
        assert abs(td.exit_fill_price(1000.0, "TEST") - 1000 * (1 - (0.1 + 0.1) / 100)) < 1e-9
        assert abs(td.entry_fill_price(1000.0, "OTHER") - 1000 * (1 + (0.02 + 0.01) / 100)) < 1e-9
        td.ASSET_COSTS = {"TEST": {"holding_pct": 2.52}}
        assert abs(td.daily_holding_cost(10_000.0, "TEST") - 10_000 * 2.52 / 100 / 252) < 1e-9


def test_whole_contract_sizing():
    # equity 100k, 1% risk = $1,000. N = 2.0 price points, 2N stop = 4.0 points => 250 shares.
    # A multiplier-50 contract: 250 / 50 = 5 contracts. A multiplier-100 contract: 2.5 -> 2 contracts.
    base = dict(equity=100_000.0, entry_price=100.0, n_value=2.0, stop_n_mult=2.0, current_notional=0.0, risk_pct=1.0)
    with Settings():
        clean_engine()
        td.MAX_TOTAL_EXPOSURE_PCT = 100_000
        fractional = td.size_unit(**base)
        assert abs(fractional["shares"] - 250.0) < 1e-9
        assert abs(td.size_unit(**base, step=50.0)["shares"] - 250.0) < 1e-9
        two = td.size_unit(**base, step=100.0)
        assert abs(two["shares"] - 200.0) < 1e-9 and abs(two["risk_amount"] - 200 * 4.0) < 1e-9
        # a 300-share step does not fit even once: skipped, or one step if asked (and the cap allows)
        assert td.size_unit(**base, step=300.0) is None
        one = td.size_unit(**base, step=300.0, take_min_one=True)
        assert abs(one["shares"] - 300.0) < 1e-9
        counters = td._EquityState(1.0)
        td.size_unit(**base, step=300.0, counters=counters)
        assert counters.n_rounding_skips == 1
        # margin room: 10% margin and $5,000 of free margin funds only $50,000 of notional = 500 shares at
        # price 100 -> not binding (250 wanted); $1,500 of free margin funds $15,000 = 150 shares
        assert abs(td.size_unit(**base, margin_room_notional=50_000.0)["shares"] - 250.0) < 1e-9
        assert abs(td.size_unit(**base, margin_room_notional=15_000.0)["shares"] - 150.0) < 1e-9
        assert td.size_unit(**base, margin_room_notional=0.0) is None
        # partial exits are whole steps, and a trim never falls short of what the cap needs
        td.SIZE_STEP = {"TEST": 5.0}
        assert td._round_partial("TEST", 12.0, 50.0) == 10.0
        assert td._round_partial("TEST", 12.0, 50.0, up=True) == 15.0
        assert td._round_partial("TEST", 12.0, 11.0, up=True) == 11.0


def test_next_open_fill():
    features, df = synthetic_market()
    with Settings():
        clean_engine()
        td.ASSET_COSTS = {"TEST": {"slippage_pct": 0.05, "spread_pct": 0.04}}
        results = {}
        for timing in ("same_close", "next_open"):
            td.EXECUTION_TIMING = timing
            results[timing] = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        for timing, res in results.items():
            trade = res["trades_by_asset"]["TEST"][0]
            signal_date = trade["entry_dates"][0]
            i = df.index.get_loc(signal_date)
            slip = (0.05 + 0.04 / 2) / 100
            raw = df["Close"].iloc[i] if timing == "same_close" else df["Open"].iloc[i + 1]
            expected = raw * (1 + slip)
            got = trade["entry_prices"][0]
            assert abs(got - expected) < 1e-6 * expected, (timing, got, expected)


def test_books_balance():
    for crash in (0.0, -0.35):
        features, _ = synthetic_market(crash=crash)
        for mode in ("plain", "margin_and_contracts"):
            with Settings():
                clean_engine()
                td.HOLDING_COST_PCT_PER_YEAR = 3.0
                td.USE_PER_CONTRACT_FEES = True
                td.FEE_SCHEDULE = {"TEST": {"multiplier": 5.0, "fee_per_contract": 0.62, "code": "MES"}}
                if mode != "plain":
                    td.SIZE_STEP = {"TEST": 5.0}
                    td.USE_MARGIN_CALLS = True
                    td.MARGIN_SPEC = {"TEST": (0.10, 0.09)}
                res = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
                units = all_units(res)
                assert units, "the synthetic market should produce trades"
                trade_pnl = sum(u["pnl"] for u in units)   # net of BOTH commissions on each unit
                expected_final = 100_000.0 + trade_pnl - res["total_holding_cost"]
                # the same figure from the trade records: every trade's P&L adds up to the account change
                assert abs(sum(t["pnl"] for ts in res["trades_by_asset"].values() for t in ts) - trade_pnl) < 1e-6
                if res["wiped_out"]:
                    continue   # the floor at zero breaks the identity by design
                assert abs(res["final_equity"] - expected_final) < 1e-4 * max(1.0, abs(expected_final)), \
                    (crash, mode, res["final_equity"], expected_final)


def test_whole_contracts_everywhere():
    features, _ = synthetic_market()
    with Settings():
        clean_engine()
        td.SIZE_STEP = {"TEST": 5.0}
        td.MAX_TOTAL_EXPOSURE_PCT = 600
        res = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        units = all_units(res)
        assert units
        for u in units:
            contracts = u["shares"] / 5.0
            assert abs(contracts - round(contracts)) < 1e-6, f"fractional contracts: {u['shares']}"


def _margin_fixture(equity_cash: float, rate: tuple):
    """One asset at a price of 70 holding two units bought at 100: A = 100 shares, B = 50 shares.
    Slippage is zero so the arithmetic is exact. Returns (state, contexts, unit_fns)."""
    td.SLIPPAGE_PCT = td.SPREAD_PCT = 0.0
    td.USE_MARGIN_CALLS = True
    td.MARGIN_SPEC = {"TEST": rate}
    unit_a = {"shares": 100.0, "entry_price": 100.0, "entry_iloc": -1}
    unit_b = {"shares": 50.0, "entry_price": 100.0, "entry_iloc": -1}
    system = {"units": [unit_a, unit_b], "episode_closed_units": [], "exit_markers": [], "trades": [],
              "in_position": True}
    ctx = {"cur_i": 0, "close": [70.0], "systems": {"S1": system}, "sys_params": {"S1": {"use_cooldown": False}}}
    state = td._EquityState(equity_cash)
    state.contexts = {"TEST": ctx}

    def close_unit(u, price, reason, date, shares_override=None):
        state.equity += u["shares"] * (price - u["entry_price"])
        return {"exit_price": price, "reason": reason}

    def finalize_episode(sys_state, equity_after):
        sys_state["trades"].append({"pnl": 0.0})

    return state, {"TEST": ctx}, {"TEST": (None, close_unit, finalize_episode)}, system


def test_margin_call_liquidates_biggest_first():
    with Settings():
        clean_engine()
        # cash 10,000; open loss = 150 shares x -30 = -4,500 -> account 5,500. Open notional at 70 is
        # 10,500; a 60% maintenance rate needs 6,300 > 5,500: margin call.
        state, contexts, fns, system = _margin_fixture(10_000.0, (0.7, 0.6))
        td._enforce_margin(contexts, fns, state, "2020-01-01")
        assert len(state.margin_events) == 1
        event = state.margin_events[0]
        assert abs(event["equity"] - 5_500.0) < 1e-9 and abs(event["required"] - 6_300.0) < 1e-9
        # closing the 100-share unit (largest margin) realizes -3,000 and leaves 50 shares:
        # account 5,500 against 50 x 70 x 0.6 = 2,100 -- covered, so only one unit is closed
        assert event["units_closed"] == 1
        assert [u["shares"] for u in system["units"]] == [50.0]
        assert abs(state.equity - 7_000.0) < 1e-9
        assert system["in_position"] is True and system["exit_markers"][0][2] == "margin_call"


def test_margin_call_not_triggered_when_covered_and_wipeout_floors_at_zero():
    with Settings():
        clean_engine()
        state, contexts, fns, system = _margin_fixture(10_000.0, (0.3, 0.2))   # needs 2,100 < 5,500
        td._enforce_margin(contexts, fns, state, "2020-01-01")
        assert state.margin_events == [] and len(system["units"]) == 2
        assert abs(state.max_margin_use - 3_150.0 / 5_500.0) < 1e-9
        # cash 1,000 and the same loss: account -3,500 -> everything is closed, equity floors at zero
        state, contexts, fns, system = _margin_fixture(1_000.0, (0.7, 0.6))
        td._enforce_margin(contexts, fns, state, "2020-01-01")
        assert system["units"] == [] and system["in_position"] is False
        assert state.wiped_out and state.equity == 0.0


def test_margin_limits_new_positions_on_synthetic_run():
    features, _ = synthetic_market()
    with Settings():
        clean_engine()
        td.MAX_TOTAL_EXPOSURE_PCT = 5000
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        for system in td.SYSTEMS:
            system["risk_per_trade_pct"] = 6.0
        free = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        td.USE_MARGIN_CALLS = True
        td.MARGIN_SPEC = {"TEST": (0.5, 0.45)}   # 2:1 leverage at most
        tight = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        assert tight["n_margin_blocked"] > 0, "a 2:1 limit must hold back the 6%-risk sizing"
        # initial margin can drift above equity after entry, but never past 1 / (maintenance / initial) = 1.11,
        # where the call fires and positions are closed
        assert tight["max_margin_use"] <= 1 / 0.9 + 1e-9, tight["max_margin_use"]
        assert tight["final_equity"] < free["final_equity"], "less leverage on a winning trend earns less"
        assert free["margin_calls"] == []


def test_mark_to_market_shows_open_losses():
    features, _ = synthetic_market(crash=-0.35)
    with Settings():
        clean_engine()
        td.MARK_TO_MARKET = False
        realized = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)["equity_curve"]
        td.MARK_TO_MARKET = True
        marked = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)["equity_curve"]
        dd = lambda s: float((s / s.cummax() - 1).min())
        assert dd(marked) <= dd(realized) + 1e-12, "counting open positions can only deepen the drawdown"


def test_replay_history_matches_the_account():
    features, _ = synthetic_market(crash=-0.2)
    with Settings():
        clean_engine()
        td.MARGIN_SPEC = {"TEST": (0.1, 0.09)}
        td.RECORD_HISTORY = False
        assert td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)["history"] is None
        td.RECORD_HISTORY = True
        try:
            res = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        finally:
            td.RECORD_HISTORY = False
        hist, positions = res["history"], res["positions_history"]
        assert len(hist) == len(res["equity_curve"])
        # every day: equity = cash + unrealized, and it equals the equity curve (all but the very last
        # value, which the engine overwrites with the post-liquidation figure)
        assert np.allclose(hist["equity"], hist["realized_balance"] + hist["unrealized"])
        assert np.allclose(hist["equity"].iloc[:-1], res["equity_curve"].iloc[:-1])
        # the per-asset rows add up to the account-level figures on a day with a position open
        day = positions["date"].iloc[len(positions) // 2]
        rows = positions[positions["date"] == day]
        assert abs(rows["unrealized"].sum() - hist.loc[day, "unrealized"]) < 1e-6
        assert abs(rows["notional"].sum() - hist.loc[day, "gross_notional"]) < 1e-6
        assert abs(rows["margin"].sum() - hist.loc[day, "margin_initial"]) < 1e-6
        assert res["margin_modelled"] is True


def test_holding_cost_is_per_calendar_day():
    with Settings():
        clean_engine()
        td.HOLDING_COST_PCT_PER_YEAR = 3.65   # 0.01% per calendar day
        # a Friday-to-Monday step is 3 calendar days, a crypto-style daily step is 1
        assert abs(td.daily_holding_cost(100_000.0, None, 3) - 100_000 * 0.0001 * 3) < 1e-9
        assert abs(td.daily_holding_cost(100_000.0, None, 1) - 100_000 * 0.0001) < 1e-9
        state = td._EquityState(100_000.0)
        ctx = {"asset": "TEST", "cur_i": 0, "dates": [pd.Timestamp("2024-01-05")],   # a Friday
               "systems": {"S1": {"units": [{"capital_deployed": 50_000.0}]}}}
        td._apply_holding_cost(ctx, state, pd.Timestamp("2024-01-08"))               # the Monday
        assert abs(state.total_holding_cost - 50_000 * 0.0001 * 3) < 1e-9


def test_highest_touched_stop_fills():
    """With the file's own default exits, every stop that fills does so at the highest level the day's low
    reached, never lower than another stop that was also touched."""
    features, df = synthetic_market(crash=-0.25)
    for mode, lower_ok in (("highest_level", False), ("legacy", True)):
        with Settings():
            clean_engine()
            td.SLIPPAGE_PCT = td.SPREAD_PCT = 0.0
            td.STOP_PRIORITY = mode
            for system in td.SYSTEMS:
                system["stop_n_mult"] = 8.0   # a wide initial stop, well under the trailing stop
            td.STOP_PRIORITY = mode
            res = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
            exits = [u for u in all_units(res) if u["exit_reason"] in ("SL", "trailing_stop_low", "chandelier_exit")]
            assert exits
    # the two modes can only differ when more than one stop is touched; the new one is never lower
    results = {}
    for mode in ("legacy", "highest_level"):
        with Settings():
            clean_engine()
            td.STOP_PRIORITY = mode
            for system in td.SYSTEMS:
                system["stop_n_mult"] = 8.0
            results[mode] = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)["final_equity"]
    assert results["highest_level"] >= results["legacy"] - 1e-6


def test_asset_order_rotates():
    """Two assets with identical signals on a tight risk budget: 'listed' always favors the first,
    'rotate' alternates, so the first asset's share of trades falls."""
    features, _ = synthetic_market()
    other = features.copy()
    other.attrs["asset"] = "OTHER"
    shares = {}
    for mode in ("listed", "rotate"):
        with Settings():
            clean_engine()
            td.ASSET_PRIORITY = mode
            td.MAX_PORTFOLIO_RISK_PCT = 1.0
            res = td.simulate_portfolio({"TEST": features, "OTHER": other}, starting_capital=100_000.0)
            n_first = len(res["trades_by_asset"]["TEST"])
            n_second = len(res["trades_by_asset"]["OTHER"])
            shares[mode] = n_first / max(n_first + n_second, 1)
    assert shares["listed"] >= shares["rotate"] - 1e-9


def test_statistics_definitions():
    idx = pd.bdate_range("2020-01-01", periods=253)
    # 252 daily steps: +1% on even days, -1% on odd days -> known downside deviation
    steps = np.array([0.01 if k % 2 == 0 else -0.01 for k in range(252)])
    equity = pd.Series(np.concatenate([[100_000.0], 100_000.0 * np.cumprod(1 + steps)]), index=idx)
    st = td.performance_stats(equity, 100_000.0)
    ppy = 252 / ((idx[-1] - idx[0]).days / 365.25)
    expected_dd = np.sqrt(np.mean(np.minimum(np.concatenate([[0.0], steps]), 0) ** 2)) * np.sqrt(ppy)
    expected_sortino = np.concatenate([[0.0], steps]).mean() * ppy / expected_dd
    assert abs(st["Sortino"] - expected_sortino) < 1e-9
    # a straight-line fall of 20 days then recovery: longest underwater stretch in calendar days
    eq2 = pd.Series([100, 110, 105, 100, 95, 100, 111, 112.0], index=pd.to_datetime(
        ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-11", "2021-01-12", "2021-01-13"]))
    st2 = td.performance_stats(eq2, 100.0)
    # peak 2021-01-05, last underwater bar 2021-01-11 (100 < 110) -> 6 calendar days
    assert st2["Max DD Duration (days)"] == 6, st2["Max DD Duration (days)"]
    assert abs(st2["Max Drawdown"] - (95 / 110 - 1)) < 1e-12


def test_walk_forward_stitching_keeps_the_boundary_day():
    import walk_forward_turtle as wf
    idx1 = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    idx2 = pd.to_datetime(["2020-01-07", "2020-01-08"])
    w1 = pd.Series([105.0, 108.0, 110.0], index=idx1)      # prior bar was 100: +5% on its first day
    w2 = pd.Series([121.0, 130.0], index=idx2)             # prior bar was 110: +10% on its first day
    stitched = wf.stitch_oos_windows([{"X": w1}, {"X": w2}], 1000.0, [{"X": 100.0}, {"X": 110.0}])["X"]
    assert abs(stitched.iloc[-1] - 1000.0 * (110 / 100) * (130 / 110)) < 1e-9
    assert abs(stitched.iloc[0] - 1050.0) < 1e-9
    # without priors the first-day returns are dropped (the old behavior), so the end value is lower
    old = wf.stitch_oos_windows([{"X": w1}, {"X": w2}], 1000.0)["X"]
    assert old.iloc[-1] < stitched.iloc[-1]
    # prior value lookup: the bar before the window start, None when there is none
    curve = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    assert wf._prior_value(curve, pd.Timestamp("2020-01-03")) == 1.0
    assert wf._prior_value(curve, pd.Timestamp("2020-01-02")) is None


def test_risk_overlays():
    features, _ = synthetic_market()
    with Settings():
        clean_engine()
        td.MAX_TOTAL_EXPOSURE_PCT = 10_000
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        for system in td.SYSTEMS:
            system["risk_per_trade_pct"] = 5.0
        free = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        first = free["trades_by_asset"]["TEST"][0]["unit_details"][0]
        assert first["capital_deployed"] > 60_000.0                      # uncapped, the first unit is big
        # a total-notional cap of 50% of equity limits it (and counts what it blocked)
        td.MAX_GROSS_EXPOSURE_PCT = 50.0
        capped = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        for t in capped["trades_by_asset"]["TEST"]:
            for u in t["unit_details"]:
                assert u["capital_deployed"] <= 0.5 * u["equity_at_entry"] * 1.0001 + 1e-6 or u["exit_reason"] == "exposure_trim"
        assert capped["n_gross_blocked"] > 0
        td.MAX_GROSS_EXPOSURE_PCT = None
    # the multipliers themselves
    state = td._EquityState(100_000.0)
    with Settings():
        td.VOL_TARGET_PCT, td.HITRATE_WINDOW, td.HITRATE_MIN_AVG_R, td.HITRATE_SCALE = None, None, -0.1, 0.5
        assert state.risk_overlay_scale() == 1.0
        td.VOL_TARGET_PCT = 20.0
        state.vol_scale = 0.4
        assert state.risk_overlay_scale() == 0.4
        td.HITRATE_WINDOW = 4
        state.recent_r = [-0.5, -0.4, -0.2]                              # fewer than the window: no effect yet
        assert state.risk_overlay_scale() == 0.4
        state.recent_r = [-0.5, -0.4, -0.2, 0.1]                         # average -0.25 < -0.1: halve again
        assert abs(state.risk_overlay_scale() - 0.2) < 1e-12
        state.recent_r = [0.5, -0.4, 0.2, 0.1]                           # average +0.1: fine
        assert state.risk_overlay_scale() == 0.4
    # volatility targeting shrinks new risk after a volatile stretch, never above the cap
    features2, _ = synthetic_market(crash=-0.3)
    with Settings():
        clean_engine()
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        base = td.simulate_portfolio({"TEST": features2}, starting_capital=100_000.0)
        td.VOL_TARGET_PCT, td.VOL_TARGET_FLOOR, td.VOL_TARGET_CAP = 1.0, 0.25, 1.0   # a 1% target: always de-risking
        cut = td.simulate_portfolio({"TEST": features2}, starting_capital=100_000.0)
        def sizes(result):   # risk taken per ORIGINAL entry (scale-out fragments each carry only a share of it)
            out = {}
            for trade in result["trades_by_asset"]["TEST"]:
                for u in trade["unit_details"]:
                    key = u["entry_iloc"]
                    out.setdefault(key, [0.0, u["equity_at_entry"]])[0] += u["risk_amount"]
            return [risk / equity for risk, equity in out.values()]
        assert np.mean(sizes(cut)) < np.mean(sizes(base))
        assert min(sizes(cut)) >= 0.25 * min(sizes(base)) * 0.99


def test_turtle_account_adjustment():
    """The Turtles' rule: new risk is sized from a notional equity 20% of the reference lower for every full 10%
    the account is below it, measured from the start of the year (or the running peak), and undone on recovery."""
    day = pd.Timestamp
    with Settings():
        td.USE_DRAWDOWN_SCALING, td.DRAWDOWN_SCALING_THRESHOLD, td.DRAWDOWN_SCALING_FACTOR = True, 0.10, 0.20
        td.DRAWDOWN_SCALING_REFERENCE = "year_start"
        state = td._EquityState(100_000.0)
        state.mark_close(100_000.0, day("2020-12-30"))                  # 2021 begins from this
        for equity, expected in [(110_000, 1.0), (100_000, 1.0), (95_000, 1.0),   # nothing until a full 10% is lost
                                 (90_000, 0.8 * 100_000 / 90_000), (85_000, 0.8 * 100_000 / 85_000),
                                 (80_000, 0.6 * 100_000 / 80_000), (70_000, 0.4 * 100_000 / 70_000),
                                 (60_000, 0.2 * 100_000 / 60_000), (50_000, 0.0), (30_000, 0.0)]:
            assert abs(state.drawdown_scale(equity, day("2021-03-01")) - expected) < 1e-12, equity
        state.mark_close(90_000.0, day("2021-12-30"))                   # a losing year: January starts from here
        assert state.drawdown_scale(90_000.0, day("2022-01-03")) == 1.0
        assert abs(state.drawdown_scale(81_000.0, day("2022-01-04")) - 0.8 * 90_000 / 81_000) < 1e-12
        assert state.drawdown_scale(95_000.0, day("2022-02-01")) == 1.0   # recovered above the reference: full size again
        td.DRAWDOWN_SCALING_REFERENCE = "peak"                          # the running high never resets
        state = td._EquityState(100_000.0)
        state.mark_close(120_000.0, day("2020-06-01"))
        state.mark_close(100_000.0, day("2020-12-30"))
        assert abs(state.drawdown_scale(100_000.0, day("2023-05-01")) - 0.8 * 120_000 / 100_000) < 1e-12
        state.mark_close(125_000.0, day("2023-06-01"))
        assert state.drawdown_scale(124_000.0, day("2023-06-02")) == 1.0
        td.USE_DRAWDOWN_SCALING = False
        assert state.drawdown_scale(50_000.0, day("2023-06-03")) == 1.0
    import optimize_turtle_dual_system as opt
    assert "DRAWDOWN_SCALING_REFERENCE" in opt._PORTFOLIO_CONFIG_ATTRS   # shipped to the optimizer's worker processes
    # in a run: a -30% overnight gap, then a new climb inside the same year. Entries before the gap are untouched;
    # entries after it risk the Turtle fraction (80% of the year's starting equity, over today's equity) of the base 3%
    rng = np.random.default_rng(3)
    close = [100.0]
    for _ in range(300):
        close.append(close[-1] * (1 + rng.normal(0, 0.002)))
    for _ in range(250):
        close.append(close[-1] * 1.004)
    gap = close[-1] * 0.7
    for _ in range(40):
        close.append(gap * (1 + rng.normal(0, 0.002)))
    for _ in range(150):
        close.append(close[-1] * 1.004)
    close = np.array(close)
    open_ = np.concatenate([[close[0]], close[:-1]])
    open_[551] = gap
    index = pd.bdate_range("2015-01-01", periods=len(close))
    frame = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                          "Close": close, "Volume": 1.0}, index=index)
    frame.attrs["asset"] = "TEST"
    features = td.build_features(frame)
    calls, real = [], td.size_unit
    def spy(*args, **kwargs):
        calls.append((args[0], args[5]))                                # equity and risk % handed to the sizer
        return real(*args, **kwargs)
    td.size_unit = spy
    try:
        with Settings():
            clean_engine()
            td.MAX_PORTFOLIO_RISK_PCT = 90.0
            td.USE_DRAWDOWN_SCALING = False
            td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
            assert {round(r, 9) for _, r in calls} == {3.0}
            calls.clear()
            td.USE_DRAWDOWN_SCALING, td.DRAWDOWN_SCALING_REFERENCE = True, "year_start"
            result = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
    finally:
        td.size_unit = real
    curve = result["equity_curve"]
    year_start = float(curve[curve.index.year < 2017].iloc[-1])         # the account as 2016 closed
    cut = [(equity, risk) for equity, risk in calls if risk < 3.0 - 1e-9]
    assert cut and all(risk <= 3.0 + 1e-9 for _, risk in calls)         # it bit after the gap, and never sized up
    for equity, risk in cut:
        assert year_start * 0.9 > equity                                # only below the first step, which is -10% of the year start
        steps = int((year_start - equity) / year_start / 0.10 + 1e-9)
        assert abs(risk - 3.0 * (1 - 0.2 * steps) * year_start / equity) < 1e-6


def test_daily_loss_stop():
    """The intraday exit sells at k x N (or p %) below YESTERDAY'S close when the market trades there, at the open if it
    opened lower, and not at all when it is off. The ordinary stop is pushed far away so only this exit can act."""
    rng = np.random.default_rng(5)
    close = [100.0]
    for _ in range(300):
        close.append(close[-1] * (1 + rng.normal(0, 0.002)))
    for _ in range(250):
        close.append(close[-1] * 1.004)
    crash_i = len(close)
    index = pd.bdate_range("2015-01-01", periods=crash_i + 45)

    def frame_with(open_gap):
        c = list(close)
        prev = c[-1]
        c.append(prev * 0.80)                                             # falls 20% during the day and closes at the low
        for _ in range(44):
            c.append(c[-1] * (1 + rng.normal(0, 0.002)))
        c = np.array(c)
        o = np.concatenate([[c[0]], c[:-1]])
        o[crash_i] = prev * (1 + open_gap)
        h = np.maximum(o, c) * 1.001
        low = np.minimum(o, c) * 0.999
        low[crash_i] = c[crash_i]
        f = pd.DataFrame({"Open": o, "High": h, "Low": low, "Close": c, "Volume": 1.0}, index=index)
        f.attrs["asset"] = "TEST"
        return td.build_features(f), prev

    def crash_trades(result):
        return [t for t in result["trades_by_asset"]["TEST"] if t["exit_date"] == index[crash_i]]

    with Settings():
        clean_engine()
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        td.USE_DRAWDOWN_SCALING = False
        for system in td.SYSTEMS:
            system["stop_n_mult"] = 60.0                                  # the fixed stop is nowhere near, and no other exit acts intraday
            system.update(use_chandelier_exit=False, use_n_day_low_trailing_stop=False, use_breakeven_stop=False,
                          use_scale_out=False, use_time_stop=False, use_giveback_stop=False)
        features, prev = frame_with(0.0)
        plain = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        assert crash_trades(plain) and all(t["exit_reason"] == "exit_channel" for t in crash_trades(plain))   # runs to the close
        td.DAILY_LOSS_STOP_PCT = 5.0
        cut = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        trades = crash_trades(cut)
        assert trades and all(t["exit_reason"] == "daily_loss_stop" for t in trades)
        assert all(abs(t["exit_price"] - td.exit_fill_price(prev * 0.95, "TEST")) < 1e-6 for t in trades)
        assert cut["final_equity"] > plain["final_equity"]                # 5% off the prior close beats the 20% fall
        td.DAILY_LOSS_STOP_PCT = None
        td.DAILY_LOSS_STOP_N_MULT = 2.0                                   # 2 x yesterday's N below yesterday's close
        by_n = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        for t in crash_trades(by_n):
            n_prev = float(features[f"n_{t['system']}"].iloc[crash_i - 1])
            assert t["exit_reason"] == "daily_loss_stop"
            assert abs(t["exit_price"] - td.exit_fill_price(prev - 2.0 * n_prev, "TEST")) < 1e-6
        assert crash_trades(by_n)
        # both set: the tighter level (the higher price) is the one that acts
        td.DAILY_LOSS_STOP_PCT = 0.5
        both = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        assert all(abs(t["exit_price"] - td.exit_fill_price(prev * 0.995, "TEST")) < 1e-6 for t in crash_trades(both))
        # it opened 10% lower: a gap fills at the open, not at the level it never traded at
        td.DAILY_LOSS_STOP_N_MULT, td.DAILY_LOSS_STOP_PCT = None, 5.0
        features_gap, prev_gap = frame_with(-0.10)
        gapped = td.simulate_portfolio({"TEST": features_gap}, starting_capital=100_000.0)
        trades = crash_trades(gapped)
        assert trades and all(abs(t["exit_price"] - td.exit_fill_price(prev_gap * 0.90, "TEST")) < 1e-6 for t in trades)
        # off: nothing changes
        td.DAILY_LOSS_STOP_PCT = None
        off = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        assert off["final_equity"] == plain["final_equity"]
    import optimize_turtle_dual_system as opt
    assert "DAILY_LOSS_STOP_PCT" in opt._PORTFOLIO_CONFIG_ATTRS and "DAILY_LOSS_STOP_N_MULT" in opt._PORTFOLIO_CONFIG_ATTRS


def test_entry_top_k_selection():
    """With ENTRY_TOP_K a new trend may only start in a market ranked 1..K that day by trend strength (log return over the lookback divided
    by its volatility), above ENTRY_MIN_STRENGTH, using closes up to the signal bar; open positions and pyramid adds are not filtered."""
    n = 700
    index = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(11)

    def market(name, drift, vol):
        close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        frame = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                              "Close": close, "Volume": 1.0}, index=index)
        frame.attrs["asset"] = name
        return td.build_features(frame)

    markets = {"A": market("A", 0.0030, 0.003), "B": market("B", 0.0012, 0.012), "C": market("C", -0.002, 0.004)}
    span = 126
    with Settings():
        clean_engine()
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        td.USE_DRAWDOWN_SCALING = False
        td.ENTRY_STRENGTH_LOOKBACK, td.ENTRY_MIN_STRENGTH, td.ENTRY_RANK_FIRST_UNIT_ONLY = span, 0.0, True
        td.ENTRY_TOP_K = None
        free = td.simulate_portfolio(markets, starting_capital=100_000.0)
        n_trades = {k: len(v) for k, v in free["trades_by_asset"].items()}
        assert n_trades["B"] >= 1 and free["n_rank_blocked"] == 0
        # the strength and rank, by hand
        logc = np.log(pd.DataFrame({k: v["Close"] for k, v in markets.items()}))
        strength = (logc - logc.shift(span)) / (logc.diff().rolling(span, min_periods=span).std() * np.sqrt(span))
        rank = strength.rank(axis=1, ascending=False, method="first")
        assert (rank["A"].dropna() == 1).all() and (rank["C"].dropna() == 3).all()      # steady climb first, steady fall last
        td.ENTRY_TOP_K = 1
        top1 = td.simulate_portfolio(markets, starting_capital=100_000.0)
        assert top1["n_rank_blocked"] > 0
        assert len(top1["trades_by_asset"]["B"]) == 0 and len(top1["trades_by_asset"]["C"]) == 0   # only the strongest may start trends
        starts = [t["entry_dates"][0] for t in top1["trades_by_asset"]["A"]]
        assert starts and all(d >= index[span] for d in starts)                          # no strength exists before one lookback of data
        td.ENTRY_TOP_K = 2
        top2 = td.simulate_portfolio(markets, starting_capital=100_000.0)
        for name, trades in top2["trades_by_asset"].items():
            for t in trades:
                d = t["entry_dates"][0]
                assert rank.loc[d, name] <= 2 and strength.loc[d, name] > 0, (name, d)   # every new trend passed the test on its signal day
        assert len(top2["trades_by_asset"]["C"]) == 0                                    # a falling market never has positive strength
        td.ENTRY_TOP_K = None
        again = td.simulate_portfolio(markets, starting_capital=100_000.0)
        assert again["final_equity"] == free["final_equity"]                              # off: identical
    import optimize_turtle_dual_system as opt
    assert "ENTRY_TOP_K" in opt._PORTFOLIO_CONFIG_ATTRS


def test_vol_risk_scale():
    """VOL_RISK_GAMMA scales a new unit's risk by clip(ratio ** -gamma, MIN, MAX), ratio = ATR(20) / its average over the previous 120 bars
    (bars up to the signal bar only; 1.0 until 140 bars exist). Off = identical to a plain run."""
    n = 600
    index = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(5)
    vol = np.where(np.arange(n) < 300, 0.004, 0.02)           # calm for 300 bars, then 5x as loud
    close = 100 * np.exp(np.cumsum(rng.normal(0.0025, vol, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    frame = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * (1 + vol / 2), "Low": np.minimum(open_, close) * (1 - vol / 2),
                          "Close": close, "Volume": 1.0}, index=index)
    frame.attrs["asset"] = "T"
    features = td.build_features(frame)
    with Settings():
        clean_engine()
        td.USE_DRAWDOWN_SCALING = False
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        td.MAX_TOTAL_EXPOSURE_PCT = 100_000
        td.VOL_RISK_GAMMA = 0.0
        assert "vol_risk_mult" not in td._build_asset_context(features)                      # off: nothing built
        plain = td.simulate_portfolio({"T": features}, starting_capital=100_000.0)
        td.VOL_RISK_GAMMA, td.VOL_RISK_ATR, td.VOL_RISK_REF, td.VOL_RISK_MIN, td.VOL_RISK_MAX = 1.0, 20, 120, 0.5, 1.5
        mult = td._build_asset_context(features)["vol_risk_mult"]
        h, l, c = frame["High"], frame["Low"], frame["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        ratio = atr / atr.shift(1).rolling(120).mean()
        want = (1 / ratio).clip(0.5, 1.5).fillna(1.0).to_numpy()
        assert np.allclose(mult, want) and (mult[:139] == 1.0).all()
        assert (mult[310:340] == 0.5).all() and mult.min() >= 0.5 and mult.max() <= 1.5       # just after the switch: ATR far above its 120-bar average -> the floor
        assert np.ptp(mult[200:290]) > 0                                                       # and it moves while the market is calm
        # causal: rewriting every bar after bar 400 must not change the multipliers up to bar 400
        wild = frame.copy()
        wild.iloc[401:, :4] = wild.iloc[401:, :4].to_numpy() * 3.0
        wild.attrs["asset"] = "T"
        assert np.allclose(td._build_asset_context(td.build_features(wild))["vol_risk_mult"][:400], mult[:400])
        # end to end: a fixed multiplier of 0.5 halves the risk of every unit opened after the warm-up
        td.VOL_RISK_MIN = td.VOL_RISK_MAX = 0.5
        half = td.simulate_portfolio({"T": features}, starting_capital=100_000.0)

        def unit_risk_share(res):
            out = []
            for t in res["trades_by_asset"]["T"]:
                for u in t["unit_details"]:
                    if u["entry_iloc"] >= 150 and u["equity_at_entry"] > 0:
                        out.append(u["risk_amount"] / u["equity_at_entry"])
            return np.median(out)
        assert abs(unit_risk_share(half) / unit_risk_share(plain) - 0.5) < 0.05, (unit_risk_share(half), unit_risk_share(plain))
        td.VOL_RISK_GAMMA = 0.0
        again = td.simulate_portfolio({"T": features}, starting_capital=100_000.0)
        assert again["final_equity"] == plain["final_equity"]                                # off again: identical
    import optimize_turtle_dual_system as opt
    assert "VOL_RISK_GAMMA" in opt._PORTFOLIO_CONFIG_ATTRS


def test_ema_entry():
    """ENTRY_EMA_*: the entry level is an EMA of the close (today's bar included). 'cross' fires only on the day the close moves from at-or-below the EMA to
    above it; 'above' whenever it is above; per-system periods; optional EMA exit; causal; the engine and the fast walk-forward cache agree; off = unchanged."""
    import wf_fast
    n = 700
    index = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(11)
    drift = np.where((np.arange(n) // 140) % 2 == 0, 0.003, -0.003)          # up and down legs so the close crosses the EMA several times
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    frame = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.004, "Low": np.minimum(open_, close) * 0.996, "Close": close, "Volume": 1.0}, index=index)
    frame.attrs["asset"] = "T"
    with Settings():
        clean_engine()
        td.ENTRY_EMA_PERIOD, td.ENTRY_EMA_MODE, td.ENTRY_EMA_EXIT = None, "cross", False
        plain = td.build_features(frame)
        assert td.entry_ema_levels(frame["Close"], "S1") is None
        td.ENTRY_EMA_PERIOD = {"S1": 50, "S2": 120}
        f = td.build_features(frame)
        c = frame["Close"]
        ema50 = c.ewm(span=50, adjust=False).mean()
        fired = (c > f["breakout_level_S1"]).to_numpy()
        want = ((c > ema50) & (c.shift(1) <= ema50.shift(1))).to_numpy()
        want[:50] = False
        assert fired.sum() >= 3 and (fired == want).all()                                       # a signal only on crossing days, none during warm-up
        assert f["breakout_level_S1"].iloc[:50].isna().all() and np.isinf(f["breakout_level_S1"].iloc[60:]).any()
        ema120 = c.ewm(span=120, adjust=False).mean()
        assert np.allclose(f["breakout_level_S2"].iloc[120:][np.isfinite(f["breakout_level_S2"].iloc[120:])], ema120.iloc[120:][np.isfinite(f["breakout_level_S2"].iloc[120:])])
        assert (f["exit_low_level_S1"].dropna() == plain["exit_low_level_S1"].dropna()).all()  # the channel exit is untouched by default
        td.ENTRY_EMA_MODE = "above"
        a = td.build_features(frame)
        assert (a["breakout_level_S1"].iloc[50:] == ema50.iloc[50:]).all()
        td.ENTRY_EMA_MODE, td.ENTRY_EMA_EXIT = "cross", True
        e = td.build_features(frame)
        assert (e["exit_low_level_S1"].iloc[50:] == ema50.iloc[50:]).all() and e["exit_low_level_S1"].iloc[:50].isna().all()
        # causal: rewriting every bar after bar 400 must not change the levels up to bar 400
        wild = frame.copy()
        wild.iloc[401:, :4] = wild.iloc[401:, :4].to_numpy() * 3.0
        wild.attrs["asset"] = "T"
        g = td.build_features(wild)
        assert np.array_equal(g["breakout_level_S1"].iloc[:400].to_numpy(), e["breakout_level_S1"].iloc[:400].to_numpy(), equal_nan=True)
        # the fast walk-forward cache builds the same columns
        cache = wf_fast.FeatureCache({"T": frame}, {})
        cached = cache.build("T")
        for col in ("breakout_level_S1", "breakout_level_S2", "exit_low_level_S1", "exit_low_level_S2"):
            assert np.array_equal(cached[col].to_numpy(), e[col].to_numpy(), equal_nan=True), col
        # end to end: runs, trades only after a cross, and switching it off restores the plain result
        td.ENTRY_EMA_EXIT = False
        td.USE_DRAWDOWN_SCALING = False
        td.MAX_TOTAL_EXPOSURE_PCT = 100_000
        res = td.simulate_portfolio({"T": td.build_features(frame)}, starting_capital=100_000.0)
        trades = res["trades_by_asset"]["T"]
        assert trades, "the EMA entry produced no trades on a series with several up-legs"
        first_signal_days = set(index[np.flatnonzero(fired)])
        assert all(pd.Timestamp(t["entry_dates"][0]) in first_signal_days for t in trades if t["system"] == "S1")
        td.ENTRY_EMA_PERIOD = None
        again = td.simulate_portfolio({"T": td.build_features(frame)}, starting_capital=100_000.0)
        base = td.simulate_portfolio({"T": plain}, starting_capital=100_000.0)
        assert again["final_equity"] == base["final_equity"]
        # the same thing set PER SYSTEM (what the UI's S1 / S2 editors write): S1 on a 50-day cross, S2 left on its breakout
        td.ENTRY_EMA_PERIOD, td.ENTRY_EMA_MODE, td.ENTRY_EMA_EXIT = None, "cross", False
        assert all(not s.get("use_ema_entry") and "ema_entry_period" in s for s in td.SYSTEMS)    # defaults exist and are off
        td.SYSTEMS = [{**s, "use_ema_entry": s["name"] == "S1", "ema_entry_period": 50, "ema_entry_mode": "cross"} for s in td.SYSTEMS]
        ps = td.build_features(frame)
        assert np.array_equal(ps["breakout_level_S1"].to_numpy(), f["breakout_level_S1"].to_numpy(), equal_nan=True)     # S1: the 50-day cross level
        assert np.array_equal(ps["breakout_level_S2"].to_numpy(), plain["breakout_level_S2"].to_numpy(), equal_nan=True)  # S2: still the N-day-high channel
        assert td.ema_entry_settings("S1", td.SYSTEMS[0]) == (50, "cross") and td.ema_entry_settings("S2", td.SYSTEMS[1]) is None
        td.ENTRY_EMA_PERIOD = {"S1": 120}                                                  # a system's own setting wins over the module-level switch
        assert td.ema_entry_settings("S1", td.SYSTEMS[0])[0] == 50 and td.ema_entry_settings("S2", td.SYSTEMS[1]) is None
        td.SYSTEMS = [{**s, "use_ema_entry": False} for s in td.SYSTEMS]
        assert td.ema_entry_settings("S1", td.SYSTEMS[0])[0] == 120
        td.ENTRY_EMA_PERIOD = None
        sys_cached = wf_fast.FeatureCache({"T": frame}, {})
        td.SYSTEMS = [{**s, "use_ema_entry": s["name"] == "S1", "ema_entry_period": 50} for s in td.SYSTEMS]
        assert np.array_equal(sys_cached.build("T")["breakout_level_S1"].to_numpy(), ps["breakout_level_S1"].to_numpy(), equal_nan=True)
        res_ps = td.simulate_portfolio({"T": ps}, starting_capital=100_000.0)
        assert res_ps["trades_by_asset"]["T"]
        td.SYSTEMS = [{**s, "use_ema_entry": False} for s in td.SYSTEMS]
        assert td.simulate_portfolio({"T": td.build_features(frame)}, starting_capital=100_000.0)["final_equity"] == base["final_equity"]
        # an EMA EXIT on its own (independent of the entry): S1 leaves on a close below its 30-day EMA instead of the exit channel, S2 keeps the channel
        td.SYSTEMS = [{**s, "use_ema_exit": s["name"] == "S1", "ema_exit_period": 30} for s in td.SYSTEMS]
        assert td.ema_exit_period("S1", td.SYSTEMS[0]) == 30 and td.ema_exit_period("S2", td.SYSTEMS[1]) is None
        xs = td.build_features(frame)
        ema30 = c.ewm(span=30, adjust=False).mean()
        assert (xs["exit_low_level_S1"].iloc[30:] == ema30.iloc[30:]).all() and xs["exit_low_level_S1"].iloc[:30].isna().all()
        assert np.array_equal(xs["exit_low_level_S2"].to_numpy(), plain["exit_low_level_S2"].to_numpy(), equal_nan=True)       # S2: the channel
        assert np.array_equal(xs["breakout_level_S1"].to_numpy(), plain["breakout_level_S1"].to_numpy(), equal_nan=True)       # the entry is untouched
        cached_x = wf_fast.FeatureCache({"T": frame}, {}).build("T")
        for col in ("breakout_level_S1", "breakout_level_S2", "exit_low_level_S1", "exit_low_level_S2"):        # the fast cache builds every column the engine does
            assert np.array_equal(cached_x[col].to_numpy(), xs[col].to_numpy(), equal_nan=True), col
            assert cached_x[col].dtype == np.float64, col
        assert td.simulate_portfolio({"T": cached_x}, starting_capital=100_000.0)["final_equity"] == td.simulate_portfolio({"T": xs}, starting_capital=100_000.0)["final_equity"]
        # every exit_channel exit of S1 happens on a close below the EMA (signal bar = the bar before the fill)
        res_x = td.simulate_portfolio({"T": xs}, starting_capital=100_000.0)
        s1_exits = [t for t in res_x["trades_by_asset"]["T"] if t["system"] == "S1" and t["exit_reason"] == "exit_channel"]
        assert s1_exits and all(c.loc[pd.Timestamp(t["exit_date"])] < ema30.loc[pd.Timestamp(t["exit_date"])] for t in s1_exits)
        # both together (EMA entry 50 + EMA exit 30 on S1)
        td.SYSTEMS = [{**s, "use_ema_entry": s["name"] == "S1", "ema_entry_period": 50, "ema_entry_mode": "cross"} for s in td.SYSTEMS]
        both = td.build_features(frame)
        assert np.array_equal(both["breakout_level_S1"].to_numpy(), f["breakout_level_S1"].to_numpy(), equal_nan=True)
        assert np.array_equal(both["exit_low_level_S1"].to_numpy(), xs["exit_low_level_S1"].to_numpy(), equal_nan=True)
        # the research switch ENTRY_EMA_EXIT exits at the ENTRY EMA of a system that has one, and a system's own exit EMA wins over it
        td.SYSTEMS = [{**s, "use_ema_entry": False, "use_ema_exit": False} for s in td.SYSTEMS]
        td.ENTRY_EMA_PERIOD, td.ENTRY_EMA_EXIT = {"S1": 50}, True
        assert td.ema_exit_period("S1", td.SYSTEMS[0]) == 50 and td.ema_exit_period("S2", td.SYSTEMS[1]) is None
        td.SYSTEMS = [{**s, "use_ema_exit": s["name"] == "S1", "ema_exit_period": 30} for s in td.SYSTEMS]
        assert td.ema_exit_period("S1", td.SYSTEMS[0]) == 30
        td.ENTRY_EMA_PERIOD, td.ENTRY_EMA_EXIT = None, False
        td.SYSTEMS = [{**s, "use_ema_exit": False} for s in td.SYSTEMS]
        assert td.simulate_portfolio({"T": td.build_features(frame)}, starting_capital=100_000.0)["final_equity"] == base["final_equity"]
    import optimize_turtle_dual_system as opt
    assert "ENTRY_EMA_PERIOD" in opt._PORTFOLIO_CONFIG_ATTRS


def _ref_monthly_high_events(index, high, close, n_series, buffer_mult, min_age, max_age, ma_period, ma_type="ema"):
    """Brute-force reference of td.monthly_high_entry_level: bar by bar, month by month, with an explicit set of used-up months."""
    n = len(high)
    h, c = np.asarray(high, float), np.asarray(close, float)
    periods = index.to_period("M")
    groups = {}
    for pos in range(n):
        groups.setdefault(periods[pos], []).append(pos)
    months = []
    for per in sorted(groups):
        poss = groups[per]
        vals = h[poss]
        top = vals.max()
        months.append((top, poss[int(np.flatnonzero(vals == top)[-1])], poss[-1] + 1))      # high, the bar that set it (most recent on a tie), first bar after the month
    ma = None
    if ma_period:
        if ma_type == "sma":
            ma = pd.Series(c).rolling(ma_period).mean().to_numpy()
        else:
            ma = pd.Series(c).ewm(span=ma_period, adjust=False).mean().to_numpy()
            ma[:ma_period] = np.nan
    used = set()
    out = np.full(n, np.nan)
    for i in range(n):
        event = False
        if not np.isnan(n_series[i]):
            for g, (top, pos_top, after) in enumerate(months):
                if g in used or i < after:
                    continue
                if c[i] > top + buffer_mult * n_series[i]:                               # the first breakout of this monthly high: it is used up now
                    used.add(g)
                    if min_age <= i - pos_top <= max_age:
                        event = True
        if i >= max_age and not np.isnan(n_series[i]):
            out[i] = 0.0 if event else np.inf
            if ma_period and not (c[i] > ma[i]):
                out[i] = np.nan
    return out


def test_monthly_high_entry():
    """Monthly-high breakout: every completed calendar month has a monthly high; each breaks out ONCE (the first close above high + buffer x N after its month ended) and is then used up;
    that breakout is a signal only if the high is min_age..max_age bars old then and the close is above its EMA (or SMA). Checked against a brute-force loop, by hand (a re-cross is not a signal, a
    high broken while too young is used up without one, one gap through two highs is one signal), causal, identical through the fast walk-forward cache, and end to end."""
    import wf_fast
    n = 1600
    index = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(5)
    drift = np.where((np.arange(n) // 120) % 2 == 0, 0.003, -0.0015)               # up and down legs with a rising trend, so monthly highs get broken
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n)), "Close": close, "Volume": 1.0}, index=index)
    frame.attrs["asset"] = "T"
    n_series = pd.Series(1.0 + 0.5 * np.abs(np.sin(np.arange(n) / 50.0)), index=index)
    months = td.monthly_high_months(index, frame["High"])
    per = index.to_period("M")
    for g, m in enumerate(sorted(set(per))):                                              # the month table: high, bar that set it, first bar after the month
        idx = np.flatnonzero(per == m)
        top = high[idx].max()
        assert months[0][g] == top and months[1][g] == idx[np.flatnonzero(high[idx] == top)[-1]] and months[2][g] == idx[-1] + 1, g
    for min_age, max_age, buf, ma, mt in ((20, 365, 0.5, 200, "ema"), (20, 365, 0.5, 200, "sma"), (20, 365, 0.0, 0, "ema"), (60, 250, 0.25, 100, "ema"),
                                          (5, 60, 0.0, 0, "sma"), (30, 120, 1.0, 50, "sma"), (30, 120, 1.0, 50, "ema")):
        got = td.monthly_high_entry_level(frame["Close"], months, n_series, buf, min_age, max_age, ma, mt).to_numpy()
        want = _ref_monthly_high_events(index, high, close, n_series.to_numpy(), buf, min_age, max_age, ma, mt)
        assert np.array_equal(got, want, equal_nan=True), (min_age, max_age, buf, ma, mt)
        assert np.isfinite(got).sum() > 5 if (min_age, max_age) == (20, 365) else True
    # by hand: monthly highs Jan 110, Feb 120, Mar 105 (low months before so that 120 bars of history exist by April)
    idx2 = pd.bdate_range("2012-10-01", "2013-05-31")
    hi2 = pd.Series(np.where(idx2 < pd.Timestamp("2013-01-01"), 90.0, 100.0), index=idx2)
    cl2 = hi2.copy()
    hi2.loc["2013-01-10"] = 110.0
    hi2.loc["2013-02-12"] = 120.0
    hi2.loc["2013-03-12"] = 105.0
    nn2 = pd.Series(2.0, index=idx2)
    months2 = td.monthly_high_months(idx2, hi2)

    def fired(closes, min_age, max_age=120):
        lv = td.monthly_high_entry_level(closes, months2, nn2, 0.5, min_age, max_age, 0)
        return [str(d.date()) for d in idx2[(closes > lv).to_numpy() & ~lv.isna().to_numpy()]]

    cl2.loc["2013-04-02":"2013-04-09"] = 104.0
    cl2.loc["2013-04-10"] = 108.0                                                      # above the March high 105 (+0.5 N = 106): its first breakout
    cl2.loc["2013-04-11"] = 104.0                                                      # back below it ...
    cl2.loc["2013-04-12"] = 108.0                                                      # ... and above again: a RE-CROSS of an already used-up high is not a signal
    cl2.loc["2013-04-15":"2013-04-19"] = 108.0
    cl2.loc["2013-04-22"] = 112.0                                                      # above the January high 110 (+1 = 111): its first breakout
    cl2.loc["2013-04-23":"2013-04-30"] = 112.0
    cl2.loc["2013-05-01"] = 125.0                                                      # above the February high 120 (+1 = 121): its first breakout
    assert fired(cl2, 5) == ["2013-04-10", "2013-04-22", "2013-05-01"], fired(cl2, 5)
    # the March high is only about 20 bars old on April 10: with a minimum age of 30 it is broken while too young, so it is used up WITHOUT a signal (and the April 12 re-cross is none either)
    assert fired(cl2, 30) == ["2013-04-22", "2013-05-01"], fired(cl2, 30)
    # one day through two highs (January 110 and February 120) is one signal, and neither can fire again
    cl3 = cl2.copy()
    cl3.loc["2013-04-22":"2013-04-30"] = 125.0
    assert fired(cl3, 5) == ["2013-04-10", "2013-04-22"], fired(cl3, 5)
    # causal: rewriting every bar after bar 700 cannot change the level up to bar 700
    lv0 = td.monthly_high_entry_level(frame["Close"], months, n_series, 0.5, 20, 365, 200)
    wild = frame.copy()
    wild.iloc[701:, :4] = wild.iloc[701:, :4].to_numpy() * 2.5
    lv1 = td.monthly_high_entry_level(wild["Close"], td.monthly_high_months(index, wild["High"]), n_series, 0.5, 20, 365, 200)
    assert np.array_equal(lv0.iloc[:701].to_numpy(), lv1.iloc[:701].to_numpy(), equal_nan=True)
    with Settings():
        clean_engine()
        td.USE_DRAWDOWN_SCALING = False
        td.MAX_TOTAL_EXPOSURE_PCT = 100_000
        plain = td.build_features(frame)
        assert td.monthly_high_settings("S1", td.SYSTEMS[0]) is None
        orig_buffer = td.SYSTEMS[0]["entry_buffer_n_mult"]
        td.SYSTEMS = [{**s, "use_monthly_high_entry": s["name"] == "S1", "mh_min_age": 20, "mh_max_age": 365, "mh_sma_period": 200, "entry_buffer_n_mult": 0.5} for s in td.SYSTEMS]
        assert td.monthly_high_settings("S1", td.SYSTEMS[0]) == (20, 365, 200, "ema") and td.monthly_high_settings("S2", td.SYSTEMS[1]) is None
        f = td.build_features(frame)
        want = _ref_monthly_high_events(index, high, close, f["n_S1"].to_numpy(), 0.5, 20, 365, 200)
        assert np.array_equal(f["breakout_level_S1"].to_numpy(), want, equal_nan=True)
        assert np.array_equal(f["breakout_level_S2"].to_numpy(), plain["breakout_level_S2"].to_numpy(), equal_nan=True)     # S2 keeps its channel
        assert np.array_equal(f["exit_low_level_S1"].to_numpy(), plain["exit_low_level_S1"].to_numpy(), equal_nan=True)     # the exit is untouched
        cached = wf_fast.FeatureCache({"T": frame}, {}).build("T")
        for col in ("breakout_level_S1", "breakout_level_S2", "exit_low_level_S1", "exit_low_level_S2"):
            assert np.array_equal(cached[col].to_numpy(), f[col].to_numpy(), equal_nan=True) and cached[col].dtype == np.float64, col
        # it beats the other entry types when several are on
        td.SYSTEMS = [{**s, "use_ema_entry": True, "use_ma_regime": True} for s in td.SYSTEMS]
        both = td.build_features(frame)
        assert np.array_equal(both["breakout_level_S1"].to_numpy(), f["breakout_level_S1"].to_numpy(), equal_nan=True)
        assert not np.array_equal(both["breakout_level_S2"].to_numpy(), plain["breakout_level_S2"].to_numpy(), equal_nan=True)     # S2: the EMA entry
        td.SYSTEMS = [{**s, "use_ema_entry": False, "use_ma_regime": False} for s in td.SYSTEMS]
        # end to end: every S1 entry day is a first-breakout day of a usable monthly high, with the close above its 200-day average
        res = td.simulate_portfolio({"T": f}, starting_capital=100_000.0)
        trades = [t_ for t_ in res["trades_by_asset"]["T"] if t_["system"] == "S1"]
        assert len(trades) >= 3, len(trades)
        ema200 = frame["Close"].ewm(span=200, adjust=False).mean()
        for t_ in trades:
            d = pd.Timestamp(t_["entry_dates"][0])
            assert f["breakout_level_S1"].loc[d] == 0.0 and frame["Close"].loc[d] > ema200.loc[d], d
        # the entry buffer is part of the event: a huge buffer means no trades
        td.SYSTEMS = [{**s, "entry_buffer_n_mult": 1000.0} for s in td.SYSTEMS]
        assert not td.simulate_portfolio({"T": td.build_features(frame)}, starting_capital=100_000.0)["trades_by_asset"]["T"]
        td.SYSTEMS = [{**s, "use_monthly_high_entry": False, "entry_buffer_n_mult": orig_buffer} for s in td.SYSTEMS]
        base = td.simulate_portfolio({"T": plain}, starting_capital=100_000.0)
        assert td.simulate_portfolio({"T": td.build_features(frame)}, starting_capital=100_000.0)["final_equity"] == base["final_equity"]       # off again: identical


def test_monthly_high_pyramid():
    """Adds follow the entry: with the monthly-high ENTRY on, a unit is added on EVERY new signal of that entry (a monthly high 20..max bars old breaks out for the first time, close above its
    EMA), whatever pyramid_method says, and nothing else adds (no spacing rule); pyramid_method "monthly_high" gives an ordinary entry the same adds; the cache builds the same columns."""
    import wf_fast
    n = 2200
    index = pd.bdate_range("2008-01-02", periods=n)
    rng = np.random.default_rng(8)
    drift = np.where((np.arange(n) // 90) % 2 == 0, 0.0035, -0.0015)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.013, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n)), "Close": close, "Volume": 1.0}, index=index)
    frame.attrs["asset"] = "T"
    mh = {"mh_min_age": 20, "mh_max_age": 100, "mh_sma_period": 50, "entry_buffer_n_mult": 0.0, "stop_n_mult": 8.0, "exit_lookback": 60,
          "max_units_per_trend": 6, "use_cooldown": False}
    ema50 = frame["Close"].ewm(span=50, adjust=False).mean()

    def opened_units(res, system="S1"):
        out = []
        for tr in res["trades_by_asset"]["T"]:
            if tr["system"] != system:
                continue
            assert tr["n_units"] <= 6
            seen = {}
            for u in tr["unit_details"]:
                seen.setdefault(u["entry_iloc"], u["entry_price"])                       # one record per unit OPENED (partial closes repeat it)
            out.append(sorted(seen))
        return out

    with Settings():
        clean_engine()
        td.USE_DRAWDOWN_SCALING = False
        td.MAX_TOTAL_EXPOSURE_PCT = 100_000
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        base_systems = [dict(s) for s in td.SYSTEMS]
        assert td.monthly_high_add_settings(td.SYSTEMS[0]) is None and "mh_add_level_S1" not in td.build_features(frame)      # off by default: nothing built
        # 1) the monthly-high ENTRY on, pyramid_method left at its default: the adds still follow the entry
        td.SYSTEMS = [{**s, **mh, "use_monthly_high_entry": s["name"] == "S1", "pyramid_method": "spacing"} for s in base_systems]
        assert td.monthly_high_add_settings(td.SYSTEMS[0]) == (20, 100, 50, "ema") and td.monthly_high_add_settings(td.SYSTEMS[1]) is None
        f = td.build_features(frame)
        want = _ref_monthly_high_events(index, high, close, f["n_S1"].to_numpy(), 0.0, 20, 100, 50, "ema")
        assert np.array_equal(f["mh_add_level_S1"].to_numpy(), want, equal_nan=True)
        assert np.array_equal(f["mh_add_level_S1"].to_numpy(), f["breakout_level_S1"].to_numpy(), equal_nan=True)               # the add level IS the entry level
        assert "mh_add_level_S2" not in f
        cached = wf_fast.FeatureCache({"T": frame}, {}).build("T")
        assert np.array_equal(cached["mh_add_level_S1"].to_numpy(), f["mh_add_level_S1"].to_numpy(), equal_nan=True) and cached["mh_add_level_S1"].dtype == np.float64
        res = td.simulate_portfolio({"T": f}, starting_capital=100_000.0)
        adds = 0
        for units in opened_units(res):
            for k, iloc in enumerate(units):
                assert f["breakout_level_S1"].iloc[iloc] == 0.0, (k, index[iloc])               # the first unit AND every add: a first breakout of a usable monthly high ...
                assert close[iloc] > ema50.iloc[iloc], (k, index[iloc])                           # ... with the close (the price of the add) above the EMA
                adds += k > 0
        assert adds >= 2, adds
        # every signal while in the position adds a unit (up to the max): count the signal days inside each trade and compare
        events = set(np.flatnonzero(f["breakout_level_S1"].to_numpy() == 0.0))
        for units in opened_units(res):
            assert set(units) <= events
        # 2) an ordinary entry with pyramid_method "monthly_high": only the ADDS follow the monthly highs
        td.SYSTEMS = [{**s, **mh, "use_monthly_high_entry": False, "pyramid_method": "monthly_high" if s["name"] == "S1" else "spacing"} for s in base_systems]
        g = td.build_features(frame)
        assert "mh_add_level_S1" in g and "mh_add_level_S2" not in g
        assert np.array_equal(g["breakout_level_S1"].to_numpy(), td.build_features(frame.assign())["breakout_level_S1"].to_numpy(), equal_nan=True)
        res2 = td.simulate_portfolio({"T": g}, starting_capital=100_000.0)
        events2 = set(np.flatnonzero(g["mh_add_level_S1"].to_numpy() == 0.0))
        adds2 = 0
        for units in opened_units(res2):
            for k, iloc in enumerate(units):
                if k >= 1:
                    assert iloc in events2 and close[iloc] > ema50.iloc[iloc], (k, index[iloc])
                    adds2 += 1
        assert adds2 >= 1, adds2
        # 3) everything off: identical to the plain run and no extra columns
        td.SYSTEMS = [{**s, **mh, "use_monthly_high_entry": False, "pyramid_method": "spacing"} for s in base_systems]
        assert "mh_add_level_S1" not in td.build_features(frame)
    import optimize_turtle_dual_system as opt
    assert "monthly_high" in opt.PER_SYSTEM_ALWAYS_SAMPLED["pyramid_method"]


def test_monthly_high_clean_mode():
    """The clean monthly-high mode: ONE system, built by params_to_systems from a minimal params dict, that has only (1) the monthly-high entry (close above its EMA, first breakout of a high aged
    min..max bars), (2) pyramiding on every further signal, (3) the fixed stop with the ratchet to the newest unit's stop, (4) a take-profit at entry + tp_n_mult x ATR(tp period). Every unit must end
    by "SL" or "scale_out" (the take-profit) only; nothing else (channel exit, trailing, chandelier, break-even, cooldown, ...) may fire, and the worker config carries the flag."""
    import wf_fast
    import optimize_turtle_dual_system as opt
    n = 2600
    index = pd.bdate_range("2006-01-02", periods=n)
    rng = np.random.default_rng(21)
    drift = np.where((np.arange(n) // 120) % 2 == 0, 0.0032, -0.0020)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.014, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, n))
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0}, index=index)
    frame.attrs["asset"] = "T"
    params = {"MAX_TOTAL_EXPOSURE_PCT": 100_000, "n_atr_period": 30, "stop_n_mult": 3.0, "tp_atr_period": 20, "tp_n_mult": 4.0, "risk_per_trade_pct": 1.0,
              "max_units_per_trend": 50, "entry_buffer_n_mult": 0.0, "use_monthly_high_entry": True, "mh_min_age": 20, "mh_max_age": 365, "mh_sma_period": 200}
    assert "MH_CLEAN_MODE" in opt._PORTFOLIO_CONFIG_ATTRS
    with Settings():
        clean_engine()
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        # the flag is off by default: nothing about the normal engine changes
        assert td.MH_CLEAN_MODE is False and td.DEFAULT_SYSTEM_PARAMS["use_channel_exit"] is True
        td.MH_CLEAN_MODE = True
        saved_enable = (opt.ENABLE_S1, opt.ENABLE_S2)
        opt.ENABLE_S1, opt.ENABLE_S2 = True, True               # clean mode ignores S2 and the lookbacks
        try:
            assert opt.is_valid_params(params)
            systems = opt.params_to_systems(params)
            assert [s_["name"] for s_ in systems] == ["S1"]
            sd = systems[0]
            for key, value in td.MH_CLEAN_SYSTEM_PARAMS.items():
                assert sd[key] == value, key
            assert sd["n_atr_period"] == 30 and sd["stop_n_mult"] == 3.0 and sd["tp_atr_period"] == 20 and sd["tp_n_mult"] == 4.0 and sd["mh_sma_period"] == 200
            # a stray key from a bigger params dict can never switch a mechanism back on
            stray = opt.params_to_systems({**params, "use_cooldown": True, "use_chandelier_exit": True, "use_channel_exit": True, "use_breakeven_stop": True})[0]
            assert not (stray["use_cooldown"] or stray["use_chandelier_exit"] or stray["use_channel_exit"] or stray["use_breakeven_stop"])
            td.USE_DRAWDOWN_SCALING = True
            opt.apply_params(params)
            assert td.USE_DRAWDOWN_SCALING is False and [s_["name"] for s_ in td.SYSTEMS] == ["S1"]
            # features: the channel exit is gone, the entry level is the monthly-high event, the cache builds the same
            f = td.build_features(frame)
            assert f["exit_low_level_S1"].isna().all()
            want = _ref_monthly_high_events(index, high, close, f["n_S1"].to_numpy(), 0.0, 20, 365, 200)
            assert np.array_equal(f["breakout_level_S1"].to_numpy(), want, equal_nan=True)
            cached = wf_fast.FeatureCache({"T": frame}, {}).build("T")
            for col in ("breakout_level_S1", "exit_low_level_S1", "mh_add_level_S1", "n_S1", "tp_n_S1"):
                assert np.array_equal(cached[col].to_numpy(), f[col].to_numpy(), equal_nan=True), col
            res = td.simulate_portfolio({"T": f}, starting_capital=100_000.0)
            trades = res["trades_by_asset"]["T"]
            assert len(trades) >= 5, len(trades)
            ema200 = frame["Close"].ewm(span=200, adjust=False).mean()
            events = set(np.flatnonzero(f["breakout_level_S1"].to_numpy() == 0.0))
            reasons, adds, tp_hits = set(), 0, 0
            for t_ in trades:
                seen = {}
                for u in t_["unit_details"]:
                    seen.setdefault(u["entry_iloc"], u)
                    reasons.add(u["exit_reason"])
                units = sorted(seen.values(), key=lambda x: x["entry_iloc"])
                for k, u in enumerate(units):
                    assert u["entry_iloc"] in events and close[u["entry_iloc"]] > ema200.iloc[u["entry_iloc"]], (k, index[u["entry_iloc"]])
                    adds += k > 0
                for u in t_["unit_details"]:
                    if u["exit_reason"] == "scale_out":
                        # the take-profit: the whole unit closes at entry + tp_n_mult x ATR(tp period) (the open, if the bar gapped over it)
                        tp_hits += 1
                        target = u["entry_price"] + 4.0 * u["tp_n_value"]
                        assert u["exit_price"] >= target * (1 - 0.01), (u["exit_price"], target)
            assert reasons <= {"SL", "scale_out", "end_of_data"}, reasons   # end_of_data = still open on the last bar, closed for the report only
            assert adds >= 3 and tp_hits >= 3, (adds, tp_hits)
            # the ratchet: at least one trade with an add and an SL exit; the exits sit at/above the stop of the newest unit minus gap slack
            checked = 0
            for t_ in trades:
                seen = {}
                for u in t_["unit_details"]:
                    seen.setdefault(u["entry_iloc"], u)
                units = sorted(seen.values(), key=lambda x: x["entry_iloc"])
                sl_units = [u for u in t_["unit_details"] if u["exit_reason"] == "SL"]
                if len(units) >= 2 and sl_units:
                    last = units[-1]
                    newest_stop = last["entry_price"] - last["stop_n_mult"] * last["n_value"]
                    first_stop = units[0]["entry_price"] - units[0]["stop_n_mult"] * units[0]["n_value"]
                    for u in sl_units:
                        assert u["exit_price"] <= max(newest_stop, first_stop) * 1.0001 or u["exit_price"] >= newest_stop * 0.9
                    checked += 1
            assert checked >= 1
            # the ENTRY BUFFER is a setting of the clean entry: absent = 0 (not the 0.25 of the standard defaults), present = used by the entry level, the cache and the trades
            assert opt.params_to_systems({k: v for k, v in params.items() if k != "entry_buffer_n_mult"})[0]["entry_buffer_n_mult"] == 0.0
            buffered = {**params, "entry_buffer_n_mult": 0.75}
            assert opt.params_to_systems(buffered)[0]["entry_buffer_n_mult"] == 0.75 and "entry_buffer_n_mult" not in td.MH_CLEAN_SYSTEM_PARAMS
            opt.apply_params(buffered)
            fb = td.build_features(frame)
            want_b = _ref_monthly_high_events(index, high, close, fb["n_S1"].to_numpy(), 0.75, 20, 365, 200)
            assert np.array_equal(fb["breakout_level_S1"].to_numpy(), want_b, equal_nan=True)
            assert not np.array_equal(fb["breakout_level_S1"].to_numpy(), f["breakout_level_S1"].to_numpy(), equal_nan=True)        # a buffer changes the signal days
            cached_b = wf_fast.FeatureCache({"T": frame}, {}).build("T")
            assert np.array_equal(cached_b["breakout_level_S1"].to_numpy(), fb["breakout_level_S1"].to_numpy(), equal_nan=True)
            assert np.array_equal(cached_b["mh_add_level_S1"].to_numpy(), fb["mh_add_level_S1"].to_numpy(), equal_nan=True)
            res_b = td.simulate_portfolio({"T": fb}, starting_capital=100_000.0)
            events_b = set(np.flatnonzero(fb["breakout_level_S1"].to_numpy() == 0.0))
            for t_ in res_b["trades_by_asset"]["T"]:
                for u in t_["unit_details"]:
                    assert u["entry_iloc"] in events_b and close[u["entry_iloc"]] > ema200.iloc[u["entry_iloc"]]
            assert len(res_b["trades_by_asset"]["T"]) >= 3
            opt.apply_params(params)
        finally:
            td.MH_CLEAN_MODE = False
            opt.ENABLE_S1, opt.ENABLE_S2 = saved_enable
    # flag off again: the full two-system set comes back from the same function
    with Settings():
        td.MH_CLEAN_MODE = False
        saved_enable = (opt.ENABLE_S1, opt.ENABLE_S2)
        opt.ENABLE_S1, opt.ENABLE_S2 = True, True
        try:
            both = opt.params_to_systems({**params, "S1_BREAKOUT_LOOKBACK": 20, "S1_EXIT_LOOKBACK": 10, "S2_BREAKOUT_LOOKBACK": 200, "S2_EXIT_LOOKBACK": 60})
            assert [s_["name"] for s_ in both] == ["S1", "S2"] and both[0]["use_channel_exit"] is True and both[0]["use_cooldown"] is True
        finally:
            opt.ENABLE_S1, opt.ENABLE_S2 = saved_enable


def test_ema_entry_exit_in_the_search():
    """The EMA entry and EMA exit are searchable toggles: the random sampler and the Optuna sampler (walk-forward) draw their lengths / type only when the toggle is on,
    values from the declared lists reach td.SYSTEMS through apply_params, and a search that does not include them leaves the engine at its defaults (off)."""
    import random
    import optuna
    import optimize_turtle_dual_system as opt
    import wf_fast
    toggles = opt.PER_SYSTEM_CONDITIONAL_TOGGLES
    assert set(toggles["use_ema_entry"]) == {"ema_entry_period", "ema_entry_mode"} and set(toggles["use_ema_exit"]) == {"ema_exit_period"}
    assert toggles["use_ema_entry"]["ema_entry_mode"] == ["cross", "above"]
    assert set(toggles["use_monthly_high_entry"]) == {"mh_min_age", "mh_max_age", "mh_sma_period", "mh_ma_type"}
    assert max(toggles["use_monthly_high_entry"]["mh_min_age"]) < min(toggles["use_monthly_high_entry"]["mh_max_age"])
    names = {"use_ema_entry", "ema_entry_period", "ema_entry_mode", "use_ema_exit", "ema_exit_period",
             "use_monthly_high_entry", "mh_min_age", "mh_max_age", "mh_sma_period", "mh_ma_type"}
    everything = set(opt.INDEPENDENT_PARAM_SPACE) | set(opt.PER_SYSTEM_ALWAYS_SAMPLED)
    for spaces in (opt.PER_SYSTEM_CONDITIONAL_TOGGLES, opt.GLOBAL_CONDITIONAL_TOGGLES):
        for toggle, subs in spaces.items():
            everything |= {toggle, *subs}
    fixed = {n: td.DEFAULT_SYSTEM_PARAMS[n] if n in td.DEFAULT_SYSTEM_PARAMS else getattr(td, n) for n in everything
             if n in td.DEFAULT_SYSTEM_PARAMS or hasattr(td, n)}                           # every parameter held at the engine default: the EMA switches are off
    fixed.update({"S1_BREAKOUT_LOOKBACK": 20, "S1_EXIT_LOOKBACK": 15, "S2_BREAKOUT_LOOKBACK": 300, "S2_EXIT_LOOKBACK": 20})
    assert fixed["use_ema_entry"] is False and fixed["use_ema_exit"] is False and fixed["use_monthly_high_entry"] is False
    with Settings():
        # not searched -> held at the pinned value (off), no EMA sub-parameter is drawn
        pinned = opt.sample_params_raw(random.Random(1), only_optimize=everything - names, fixed_values=fixed)
        assert pinned["use_ema_entry"] is False and pinned["use_ema_exit"] is False and "ema_exit_period" not in pinned
        # searched -> both appear, with values from the lists
        seen = {"entry": set(), "exit": set(), "mode": set()}
        for seed in range(60):
            p = opt.sample_params_raw(random.Random(seed), only_optimize=everything, fixed_values=fixed)
            if p["use_ema_entry"]:
                seen["entry"].add(p["ema_entry_period"]); seen["mode"].add(p["ema_entry_mode"])
            if p["use_ema_exit"]:
                seen["exit"].add(p["ema_exit_period"])
        assert seen["mode"] == {"cross", "above"} and len(seen["entry"]) > 2 and len(seen["exit"]) > 2
        assert seen["entry"] <= set(toggles["use_ema_entry"]["ema_entry_period"]) and seen["exit"] <= set(toggles["use_ema_exit"]["ema_exit_period"])
        # the Optuna sampler (walk-forward) follows the same rules
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
        on = {"entry": 0, "exit": 0}
        for _ in range(40):
            p = wf_fast.suggest_params(study.ask(), everything, fixed)
            on["entry"] += p["use_ema_entry"]; on["exit"] += p["use_ema_exit"]
            if p["use_ema_exit"]:
                assert p["ema_exit_period"] in toggles["use_ema_exit"]["ema_exit_period"]
        assert on["entry"] > 5 and on["exit"] > 5
        # the entry types replace one another: a trial may have at most one on, and the monthly high must be older than min_age but no older than max_age
        base_ok = dict(fixed)
        assert opt.is_valid_params(base_ok)
        assert not opt.is_valid_params(dict(base_ok, use_ema_entry=True, use_monthly_high_entry=True))
        assert not opt.is_valid_params(dict(base_ok, use_ma_regime=True, use_ema_entry=True))
        assert opt.is_valid_params(dict(base_ok, use_monthly_high_entry=True, mh_min_age=20, mh_max_age=365))
        assert not opt.is_valid_params(dict(base_ok, use_monthly_high_entry=True, mh_min_age=60, mh_max_age=60))
        mh_seen = set()
        for seed in range(80):
            p = opt.sample_params_raw(random.Random(seed), only_optimize=everything, fixed_values=fixed)
            if p["use_monthly_high_entry"]:
                mh_seen.add((p["mh_min_age"], p["mh_max_age"], p["mh_sma_period"]))
                assert p["mh_min_age"] in toggles["use_monthly_high_entry"]["mh_min_age"] and p["mh_max_age"] in toggles["use_monthly_high_entry"]["mh_max_age"]
        assert len(mh_seen) > 5
        # a drawn parameter set reaches the systems (shared by S1 and S2, like every non-lookback parameter)
        opt.ENABLE_S1 = opt.ENABLE_S2 = True
        params = dict(fixed, use_ema_entry=True, ema_entry_period=75, ema_entry_mode="above", use_ema_exit=True, ema_exit_period=30)
        opt.apply_params(params)
        assert all(s["use_ema_entry"] and s["ema_entry_period"] == 75 and s["ema_entry_mode"] == "above"
                   and s["use_ema_exit"] and s["ema_exit_period"] == 30 for s in td.SYSTEMS)
        opt.apply_params(dict(fixed, use_monthly_high_entry=True, mh_min_age=30, mh_max_age=250, mh_sma_period=150))
        assert all(s["use_monthly_high_entry"] and (s["mh_min_age"], s["mh_max_age"], s["mh_sma_period"]) == (30, 250, 150) for s in td.SYSTEMS)
        opt.apply_params(fixed)
        assert all(not s["use_ema_entry"] and not s["use_ema_exit"] and not s["use_monthly_high_entry"] for s in td.SYSTEMS)


def _fake_ctx(price, shares, tail):
    """Just enough of a market context for the state's open-position loops: one unit, one bar."""
    return {"cur_i": 0, "close": [price], "tail": np.array([tail]),
            "systems": {"S": {"units": [{"shares": shares, "entry_iloc": -1, "entry_price": price}]}}}


def test_tail_aware_risk():
    """Tail = a market's worst one-day drop over the last TAIL_WINDOW bars (causal). GAP_SIZING_K sizes a unit as if its
    stop were at least K tails away; the STRESS_* caps limit sum(notional x tail) in total, per market and per cluster;
    BOOK_VOL_CAP_PCT limits the open book's covariance-based volatility."""
    # --- the tail itself: causal, floored, with a fallback until enough days exist
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.004, 700))
    close[400:] *= 0.9                                                  # one -10% day at bar 400
    frame = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1.0},
                         index=pd.bdate_range("2012-01-02", periods=700))
    frame.attrs["asset"] = "TEST"
    with Settings():
        td.GAP_SIZING_K, td.TAIL_WINDOW, td.TAIL_MIN_BARS, td.TAIL_FALLBACK = 1.0, 300, 250, 0.15
        ctx = td._build_asset_context(td.build_features(frame))
        assert np.isnan(ctx["tail"][:250]).all() and np.isfinite(ctx["tail"][250:]).all()
        assert abs(td._tail_at(ctx, 100) - 0.15) < 1e-12                # too little history: the fallback
        assert td._tail_at(ctx, 399) < 0.05 and abs(td._tail_at(ctx, 400) - 0.10) < 0.01   # appears on the day, not before
        assert abs(td._tail_at(ctx, 699) - 0.10) < 0.01                 # still inside the 300-bar window
        td.TAIL_WINDOW = 200
        ctx = td._build_asset_context(td.build_features(frame))
        assert td._tail_at(ctx, 699) < 0.05                             # it rolled out of a shorter window
        quiet = {"tail": np.array([0.003])}
        assert td._tail_at(quiet, 0) == 0.02                            # never below 2%
    # --- sizing: the stop distance is only a floor for the size, and the recorded risk stays the nominal one
    plain = td.size_unit(100_000.0, 100.0, 1.0, 2.0, 0.0, risk_pct=1.0)                      # 1000 / 2 = 500 shares
    gap = td.size_unit(100_000.0, 100.0, 1.0, 2.0, 0.0, risk_pct=1.0, min_risk_distance=5.0)  # 1000 / 5 = 200 shares
    assert abs(plain["shares"] - 500.0) < 1e-9 and abs(gap["shares"] - 200.0) < 1e-9
    assert abs(gap["risk_amount"] - 200.0 * 2.0) < 1e-9
    near = td.size_unit(100_000.0, 100.0, 1.0, 2.0, 0.0, risk_pct=1.0, min_risk_distance=1.0)  # below the stop: no effect
    assert abs(near["shares"] - 500.0) < 1e-9
    state = td._EquityState(100_000.0)
    capped = td.size_unit(100_000.0, 100.0, 1.0, 2.0, 0.0, risk_pct=1.0, stress_room_notional=5_000.0, counters=state)
    assert abs(capped["capital_deployed"] - 5_000.0) < 1e-9
    assert td.size_unit(100_000.0, 100.0, 1.0, 2.0, 0.0, risk_pct=1.0, stress_room_notional=0.0, counters=state) is None
    assert state.n_stress_blocked == 1
    # --- the caps, by hand: A 5,000 notional x 10% = 500 stress; B 2,000 x 20% = 400; C is a different cluster
    with Settings():
        td.MARKET_CLUSTERS = {"A": "g", "B": "g", "C": "h"}
        state = td._EquityState(10_000.0)
        state.contexts = {"A": _fake_ctx(50.0, 100.0, 0.10), "B": _fake_ctx(10.0, 200.0, 0.20), "C": _fake_ctx(1.0, 0.0, 0.10)}
        state.contexts["C"]["systems"]["S"]["units"] = []
        td.STRESS_TOTAL_CAP_PCT = td.STRESS_SINGLE_CAP_PCT = td.STRESS_CLUSTER_CAP_PCT = None
        assert state.stress_room("C", 0.10, 10_000.0) is None          # no cap set
        td.STRESS_TOTAL_CAP_PCT = 20.0                                  # 2000 - 900 = 1100 of stress left
        assert abs(state.stress_room("C", 0.10, 10_000.0) - 11_000.0) < 1e-9
        td.STRESS_SINGLE_CAP_PCT = 10.0                                 # A: 1000 - 500 = 500 -> 5,000 notional; C: untouched
        assert abs(state.stress_room("A", 0.10, 10_000.0) - 5_000.0) < 1e-9
        assert abs(state.stress_room("C", 0.10, 10_000.0) - 10_000.0) < 1e-9
        td.STRESS_CLUSTER_CAP_PCT = 12.0                                # cluster g holds 900 of 1200: B has 300 -> 1,500 notional
        assert abs(state.stress_room("B", 0.20, 10_000.0) - 1_500.0) < 1e-9
        assert abs(state.stress_room("C", 0.10, 10_000.0) - 10_000.0) < 1e-9   # C is alone in its cluster
        td.STRESS_TOTAL_CAP_PCT = 8.0                                   # already over in total: nothing more for anyone
        assert state.stress_room("C", 0.10, 10_000.0) < 0
    # --- book volatility: after adding the room, the book's annualised vol is exactly the cap
    with Settings():
        rets = np.random.default_rng(7).normal(0, 0.01, (200, 2))
        rets[:, 1] = 0.6 * rets[:, 0] + 0.8 * rets[:, 1]                # two correlated markets
        state = td._EquityState(10_000.0)
        state.contexts = {"A": _fake_ctx(50.0, 100.0, 0.1), "B": _fake_ctx(10.0, 0.0, 0.1)}
        state.contexts["B"]["systems"]["S"]["units"] = []
        state.ret_matrix, state.asset_col, state.cur_row = rets, {"A": 0, "B": 1}, 150
        td.BOOK_VOL_WINDOW = 60
        td.BOOK_VOL_CAP_PCT = None
        assert state.book_vol_room("B", 10_000.0) is None
        td.BOOK_VOL_CAP_PCT = 20.0
        room = state.book_vol_room("B", 10_000.0)
        cov = np.cov(rets[91:151], rowvar=False)
        w = np.array([0.5, room / 10_000.0])
        assert abs(np.sqrt(w @ cov @ w * 252) - 0.20) < 1e-9, room
        td.BOOK_VOL_CAP_PCT = 1.0                                       # the book alone is already above 1%: no room
        assert state.book_vol_room("B", 10_000.0) == 0.0
        state.cur_row = 30                                              # not enough history yet: no opinion
        assert state.book_vol_room("B", 10_000.0) is None
    # --- account-level pieces: equity average, shock pause
    day = pd.Timestamp
    with Settings():
        td.EQUITY_MA_WINDOW, td.EQUITY_MA_SCALE = 3, 0.5
        td.VOL_TARGET_PCT, td.HITRATE_WINDOW = None, None
        state = td._EquityState(100.0)
        for value in (100.0, 110.0):
            state.mark_close(value, day("2021-01-04"))
        assert state.risk_overlay_scale() == 1.0                        # fewer closes than the window
        state.mark_close(120.0, day("2021-01-05"))
        assert state.risk_overlay_scale() == 1.0                        # 120 is above its 3-day average of 110
        state.mark_close(100.0, day("2021-01-06"))                      # average of 110, 120, 100 = 110: below it
        assert state.risk_overlay_scale() == 0.5
        td.SHOCK_PAUSE_LOSS_PCT, td.SHOCK_PAUSE_DAYS = 10.0, 5
        state = td._EquityState(100.0)
        state.mark_close(95.0, day("2021-02-01"))
        assert state.pause_until is None                                # -5% is not a shock
        state.mark_close(80.0, day("2021-02-02"))                       # -15.8%
        assert state.pause_until == day("2021-02-07")
    # --- in a run: a stress cap and the extension rule bite, and everything off changes nothing
    features, _ = synthetic_market()
    with Settings():
        clean_engine()
        td.MAX_TOTAL_EXPOSURE_PCT = 10_000
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        td.USE_DRAWDOWN_SCALING = False
        for system in td.SYSTEMS:
            system["risk_per_trade_pct"] = 5.0
        base = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        first = base["trades_by_asset"]["TEST"][0]["unit_details"][0]
        assert first["capital_deployed"] > 0.6 * first["equity_at_entry"]            # uncapped it is a big unit
        td.STRESS_SINGLE_CAP_PCT = 0.5                                  # 0.5% of equity lost on a repeat of the worst (2% floor) day
        capped = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        u = capped["trades_by_asset"]["TEST"][0]["unit_details"][0]
        assert u["capital_deployed"] <= 0.25 * u["equity_at_entry"] * 1.0001 + 1e-6   # 0.5% / 2% floor = 25% of equity
        td.STRESS_SINGLE_CAP_PCT = None
        td.GAP_SIZING_K = 1.0                                           # sized for a 2% floor tail: 2% x price beyond the 3N stop?
        gap_run = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        td.GAP_SIZING_K = 0.0
        assert gap_run["trades_by_asset"]["TEST"][0]["unit_details"][0]["capital_deployed"] <= first["capital_deployed"] + 1e-6
        calls, real = [], td.size_unit
        def spy(*args, **kwargs):
            calls.append(args[5])
            return real(*args, **kwargs)
        td.size_unit = spy
        try:
            td.EXTENSION_MA, td.EXTENSION_N_MULT, td.EXTENSION_RISK_MULT = 20, 0.0, 0.5   # any close above its average counts as extended
            td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        finally:
            td.size_unit = real
        assert calls and set(round(r, 9) for r in calls) == {2.5}                       # 5% x 0.5 on the entries made in the climb
        td.EXTENSION_MA = None
        again = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        assert again["final_equity"] == base["final_equity"]            # everything off: identical to the plain run
    import optimize_turtle_dual_system as opt
    for name in ("GAP_SIZING_K", "STRESS_TOTAL_CAP_PCT", "BOOK_VOL_CAP_PCT", "EQUITY_MA_WINDOW", "SHOCK_PAUSE_LOSS_PCT", "EXTENSION_MA"):
        assert name in opt._PORTFOLIO_CONFIG_ATTRS


def _first_entry_risk(result, asset):
    """Risk of the market's first unit as a fraction of the equity it was sized from."""
    u = result["trades_by_asset"][asset][0]["unit_details"][0]
    return u["risk_amount"] / u["equity_at_entry"]


def test_per_market_risk_multipliers():
    features, _ = synthetic_market()
    two = {"A": features, "B": features.copy()}
    with Settings():
        clean_engine()
        td.MAX_TOTAL_EXPOSURE_PCT = 10_000
        td.MAX_PORTFOLIO_RISK_PCT = 90.0
        for system in td.SYSTEMS:
            system["risk_per_trade_pct"] = 1.0
        td.ASSET_RISK_MULT, td.ASSET_RISK_NORMALIZE = {}, False
        base = td.simulate_portfolio(two, starting_capital=100_000.0)
        rb_a, rb_b = _first_entry_risk(base, "A"), _first_entry_risk(base, "B")
        assert abs(rb_a / rb_b - 1.0) < 0.01 and abs(rb_a - 0.01) < 0.0015   # same market, same 1% risk
        # a market's multiplier scales ITS risk only
        td.ASSET_RISK_MULT = {"A": 2.0}
        tilted = td.simulate_portfolio(two, starting_capital=100_000.0)
        assert abs(_first_entry_risk(tilted, "A") / rb_a - 2.0) < 0.02
        assert abs(_first_entry_risk(tilted, "B") / rb_b - 1.0) < 0.02
        # rescaled to an average of 1.0: A 4/3, B 2/3 -- the same total as before, A at twice B
        td.ASSET_RISK_NORMALIZE = True
        tilted = td.simulate_portfolio(two, starting_capital=100_000.0)
        assert abs(_first_entry_risk(tilted, "A") / rb_a - 4 / 3) < 0.02
        assert abs(_first_entry_risk(tilted, "B") / rb_b - 2 / 3) < 0.02
        # rescaling is over the markets actually in the run: a lone market ends up at 1.0 whatever it was set to
        solo = td.simulate_portfolio({"A": features}, starting_capital=100_000.0)
        assert abs(_first_entry_risk(solo, "A") / rb_a - 1.0) < 0.02
        assert td.asset_risk_multipliers(["A"], {"A": 2.0}, True) == {"A": 1.0}
    m = td.asset_risk_multipliers(["A", "B", "C"], {"A": 3.0, "B": 0.0}, True)
    assert abs(sum(m.values()) / 3 - 1.0) < 1e-12 and m["B"] == 0.0 and m["C"] == m["A"] / 3
    assert td.asset_risk_multipliers(["A", "B"], {}, True) == {"A": 1.0, "B": 1.0}
    assert td.asset_risk_multipliers(["A"], {"A": -1.0}, False) == {"A": 0.0}


def test_market_multipliers_are_searchable():
    import random
    import optuna
    import optimize_turtle_dual_system as opt
    import wf_fast
    prefix = td.ASSET_RISK_PARAM_PREFIX
    saved_space = opt.get_search_space()
    with Settings():
        try:
            space = opt.get_search_space()
            space["asset_risk"] = {"A": [0.5, 1.0, 2.0], "B": [0.5, 1.0, 2.0]}
            opt.set_search_space(space)
            assert opt.get_search_space()["asset_risk"] == space["asset_risk"]
            # A is searched, B is held at the manual value 1.5
            everything = set(opt.INDEPENDENT_PARAM_SPACE) | set(opt.PER_SYSTEM_ALWAYS_SAMPLED)
            for toggles in (opt.PER_SYSTEM_CONDITIONAL_TOGGLES, opt.GLOBAL_CONDITIONAL_TOGGLES):
                for toggle, subs in toggles.items():
                    everything |= {toggle, *subs}
            only = everything | {prefix + "A"}
            fixed = {prefix + "B": 1.5}
            seen_a = set()
            for seed in range(30):
                params = opt.sample_params_raw(random.Random(seed), only_optimize=only, fixed_values=fixed)
                assert params[prefix + "B"] == 1.5 and params[prefix + "A"] in (0.5, 1.0, 2.0)
                seen_a.add(params[prefix + "A"])
            assert seen_a == {0.5, 1.0, 2.0}
            # the Optuna sampler follows the same rules
            study = optuna.create_study()
            trial = study.ask()
            p = wf_fast.suggest_params(trial, only, fixed)
            assert p[prefix + "B"] == 1.5 and p[prefix + "A"] in (0.5, 1.0, 2.0)
            # a trial's multipliers become the engine's, and only then
            td.ASSET_RISK_MULT = {"Z": 9.0}
            opt.apply_params({**p, prefix + "A": 2.0, prefix + "B": 0.5})
            assert td.ASSET_RISK_MULT == {"A": 2.0, "B": 0.5}
            td.ASSET_RISK_MULT = {"Z": 9.0}
            no_mults = {k: v for k, v in p.items() if not k.startswith(prefix)}
            opt.apply_params(no_mults)
            assert td.ASSET_RISK_MULT == {"Z": 9.0}                       # no multiplier parameters: the manual ones stay
            assert not any(hasattr(td, k) for k in (prefix + "A", prefix + "B"))
            # the nudge test sees them as ordinary numeric parameters
            import monte_carlo
            cands = monte_carlo.candidate_lists(opt.get_search_space())
            assert cands[prefix + "A"] == [0.5, 1.0, 2.0]
        finally:
            opt.set_search_space(saved_space)
    assert opt.get_search_space()["asset_risk"] == saved_space["asset_risk"]
    # a space saved before this section existed still loads
    legacy = {k: v for k, v in opt.get_search_space().items() if k != "asset_risk"}
    opt.set_search_space(legacy)
    assert opt.get_search_space()["asset_risk"] == {}
    assert "ASSET_RISK_NORMALIZE" in opt._PORTFOLIO_CONFIG_ATTRS and "ASSET_RISK_MULT" in opt._PORTFOLIO_CONFIG_ATTRS


def test_carry_arithmetic():
    with Settings():
        clean_engine()
        td.CARRY_DIVIDEND_YIELD = {"X": 0.03}
        # an account in cash earns the T-bill rate on its equity
        idle = td._EquityState(100_000.0)
        idle.contexts = {}
        idle.accrue_carry(365, 0.05)
        assert abs(idle.equity - 105_000.0) < 1e-6 and abs(idle.total_carry - 5_000.0) < 1e-6

        def account(shares):
            es = td._EquityState(100_000.0)
            unit = {"shares": shares, "entry_iloc": -1, "entry_price": 100.0}
            es.contexts = {"X": {"cur_i": 0, "close": np.array([100.0]), "systems": {"S": {"units": [unit]}}}}
            return es
        # a position worth 100% of equity earns just its dividend yield: 5% on the equity - (5% - 3%) on the position
        full = account(1_000.0)
        full.accrue_carry(365, 0.05)
        assert abs(full.total_carry - 3_000.0) < 1e-6
        # worth 200% of equity: pays the T-bill rate on the extra 100%, earns the dividend on all of it
        double = account(2_000.0)
        double.accrue_carry(365, 0.05)
        assert abs(double.total_carry - (2 * 3_000.0 - 5_000.0)) < 1e-6
        # accrual scales with the days elapsed
        week = account(1_000.0)
        week.accrue_carry(7, 0.05)
        assert abs(week.total_carry - 3_000.0 * 7 / 365) < 1e-6
        # the rate on each date: the series carried forward, the flat rate before it starts
        td.CARRY_RATE_FLAT = 0.01
        td.CARRY_RATE_SERIES = pd.Series([0.04, 0.06], index=pd.to_datetime(["2020-01-02", "2020-01-10"]))
        dates = pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-08", "2020-01-10", "2020-01-20"])
        assert list(td.carry_rates_for(dates)) == [0.01, 0.04, 0.04, 0.06, 0.06]
        # a whole run with no trades: the account compounds at the T-bill rate, day by day
        td.CARRY_MODE, td.CARRY_RATE_SERIES, td.CARRY_RATE_FLAT = "futures", None, 0.05
        index = pd.bdate_range("2018-01-01", periods=300)
        df = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0}, index=index)
        feats = td.build_features(df)
        feats.attrs["asset"] = "FLAT"
        res = td.simulate_portfolio({"FLAT": feats}, starting_capital=100_000.0)
        expected = 100_000.0
        for gap in np.diff(index.values).astype("timedelta64[D]").astype(float):
            expected *= 1 + 0.05 * gap / 365.0
        assert sum(len(v) for v in res["trades_by_asset"].values()) == 0
        assert abs(res["final_equity"] - expected) < 1e-6 and abs(res["total_carry"] - (expected - 100_000.0)) < 1e-6
        # off by default
        td.CARRY_MODE = "off"
        assert td.simulate_portfolio({"FLAT": feats}, starting_capital=100_000.0)["final_equity"] == 100_000.0
    import carry_data
    assert carry_data.dividend_yields(2.0, ["SP500", "Gold"]) == {"SP500": 2.0 * carry_data.DEFAULT_DIVIDEND_YIELDS["SP500"]}


def test_ma_regime_system():
    import wf_fast
    close = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
    up, down = td.ma_regime_levels(close, 3, 2.0)
    assert abs(up.iloc[2] - 101.0 * 1.02) < 1e-9 and abs(down.iloc[2] - 101.0 * 0.98) < 1e-9
    assert np.isnan(up.iloc[1])                                                   # not enough history yet
    for band in (0.0, 3.0):
        _, raw = synthetic_market(crash=-0.3)
        with Settings():
            clean_engine()
            td.MAX_TOTAL_EXPOSURE_PCT = 100
            td.USE_PORTFOLIO_RISK_CAP = False
            system = {**td.SYSTEMS[0], "name": "S2", "use_ma_regime": True, "ma_regime_period": 50, "ma_regime_band_pct": band,
                      "entry_buffer_n_mult": 3.0, "risk_per_trade_pct": 20.0, "stop_n_mult": 10.0, "max_units_per_trend": 1,
                      "use_scale_out": False, "use_time_stop": False, "use_cooldown": False, "use_ema_filter": False}
            td.SYSTEMS = [system]
            features = td.build_features(raw)
            features.attrs["asset"] = "TEST"
            ma = raw["Close"].rolling(50).mean()
            res = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
            trades = res["trades_by_asset"]["TEST"]
            assert len(trades) >= 1
            for t in trades:
                for u in t["unit_details"]:
                    sig = raw.index[u["entry_iloc"]]                               # the bar whose close gave the signal
                    assert raw["Close"].loc[sig] > ma.loc[sig] * (1 + band / 100) - 1e-9
                    # filled at the NEXT bar's open, not at the signal close
                    nxt = raw.index[u["entry_iloc"] + 1]
                    assert abs(u["entry_price"] - raw["Open"].loc[nxt] * (1 + (td.SLIPPAGE_PCT + td.SPREAD_PCT / 2) / 100)) < 1e-6 * raw["Open"].loc[nxt]
                if t["exit_reason"] == "exit_channel":
                    assert raw["Close"].loc[t["exit_date"]] < ma.loc[t["exit_date"]] * (1 - band / 100) + 1e-9
            # the entry buffer (3 N here) is ignored: the first entry comes right when the close clears the average
            first_signal = raw.index[trades[0]["unit_details"][0]["entry_iloc"]]
            earlier = (raw["Close"] > ma * (1 + band / 100)).loc[:first_signal].iloc[:-1]
            assert not earlier.any()
            # the walk-forward feature cache builds the same levels
            cache = wf_fast.FeatureCache({"TEST": raw})
            cached = cache.build("TEST")
            assert np.allclose(cached["breakout_level_S2"].values, features["breakout_level_S2"].values, equal_nan=True)
            assert np.allclose(cached["exit_low_level_S2"].values, features["exit_low_level_S2"].values, equal_nan=True)
    # off by default, and the optimizer knows about it
    assert td.DEFAULT_SYSTEM_PARAMS["use_ma_regime"] is False and td.s2_SYSTEM_PARAMS["use_ma_regime"] is False
    import optimize_turtle_dual_system as opt
    assert "use_ma_regime" in opt.PER_SYSTEM_CONDITIONAL_TOGGLES


def _steady_climb_then_fall(gap_open=None):
    """100 bars climbing exactly 1 a day (range 2 a bar, so ATR is 2.0), a bar that falls through the trail, then a steady decline."""
    close = [100.0 + k for k in range(100)]                       # bars 0..99: 100..199
    open_ = [100.0] + close[:-1]
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    open_.append(gap_open if gap_open is not None else 199.0)   # bar 100: the fall
    high.append(open_[-1])
    low.append(open_[-1] - 9.0)
    close.append(open_[-1] - 8.0)
    for k in range(1, 40):                                        # bars 101..: keeps falling, no re-entry
        c = close[-1] - 1.0
        open_.append(close[-1]); high.append(close[-1]); low.append(c - 1.0); close.append(c)
    index = pd.bdate_range("2018-01-01", periods=len(close))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0}, index=index)
    df.attrs["asset"] = "TEST"
    return df


def test_repair_ohlc():
    rng = np.random.default_rng(11)
    n = 600
    close = 100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    index = pd.bdate_range("2015-01-01", periods=n)
    clean = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=index)
    # clean data comes back unchanged
    fixed, rep = td.repair_ohlc(clean)
    assert rep == {"rows_dropped": 0, "unit_rescales": 0, "spikes_removed": 0, "widened": 0, "flat_estimated": 0}
    assert np.allclose(fixed.to_numpy(), clean.to_numpy())
    bad = clean.copy()
    bad.iloc[200] =[-37.0, -36.0, -38.0, -37.6]                                        # a negative price
    bad.iloc[:300, :] = bad.iloc[:300, :] / 10.0                                        # the first 300 bars quoted in tenth units
    bad.iloc[400, 3] = bad.iloc[400, 3] * 1.6                                           # a bad tick on the close ...
    bad.iloc[400, 1] = bad.iloc[400, 1] * 1.6                                           # ... that is gone the next day
    bad.iloc[450, 1] = bad.iloc[450, 3] * 0.9                                           # a high below its own close
    for k in range(500, 560):                                                           # settlement-only bars
        bad.iloc[k, 0:3] = bad.iloc[k, 3]
    fixed, rep = td.repair_ohlc(bad)
    assert rep["rows_dropped"] == 1 and rep["unit_rescales"] == 1 and rep["spikes_removed"] >= 1 and rep["flat_estimated"] == 60 and rep["widened"] >= 1
    assert (fixed > 0).all().all() and fixed["Close"].pct_change().abs().max() < 0.2          # no bad prints left
    assert (fixed["High"] >= fixed[["Open", "Close"]].max(axis=1) - 1e-12).all() and (fixed["Low"] <= fixed[["Open", "Close"]].min(axis=1) + 1e-12).all()
    # the first stretch was brought to the later scale: it now sits where the untouched data does
    assert abs(fixed["Close"].iloc[10] / clean["Close"].iloc[10] - 1) < 1e-9
    real_range = ((clean["High"] - clean["Low"]) / clean["Close"]).iloc[380:500].median()       # what the real bars before them looked like
    est_range = ((fixed["High"] - fixed["Low"]) / fixed["Close"]).iloc[520:560].median()
    assert abs(est_range / real_range - 1) < 0.3, (est_range, real_range)
    # settlement-only bars get an estimated range from PAST bars only: changing the future cannot change an earlier estimate
    later = bad.copy()
    later.iloc[570:, :] = later.iloc[570:, :] * 1.0
    later.iloc[580, 1] = later.iloc[580, 3] * 1.5
    f2, _ = td.repair_ohlc(later)
    assert np.allclose(f2.iloc[500:560].to_numpy(), fixed.iloc[500:560].to_numpy())
    # the switch and the data-quality report
    assert td.REPAIR_BAD_DATA is True
    fixed.attrs["repair"] = rep
    rows = td.data_quality({"X": fixed})
    assert "ESTIMATED" in rows[0]["Verdict"] or rep["flat_estimated"] / len(fixed) <= 0.02
    assert "repaired" in rows[0]["Verdict"]


def test_peak_trail_stop():
    for gap_open, slip, expected_raw in ((None, 0.0, 196.0), (None, 1.0, 196.0 * 0.99), (185.0, 0.0, 185.0)):
        raw = _steady_climb_then_fall(gap_open)
        with Settings():
            clean_engine()
            td.MAX_TOTAL_EXPOSURE_PCT = 100
            td.USE_PORTFOLIO_RISK_CAP = False
            td.STOP_SLIPPAGE_PCT = slip
            system = {**td.SYSTEMS[0], "name": "S2", "use_ma_regime": True, "ma_regime_period": 5, "ma_regime_band_pct": 0.0,
                      "use_peak_trail": True, "peak_trail_atr_mult": 2.0, "n_atr_period": 20, "risk_per_trade_pct": 20.0, "stop_n_mult": 50.0,
                      "max_units_per_trend": 1, "use_scale_out": False, "use_time_stop": False, "use_cooldown": False, "use_ema_filter": False,
                      "use_n_day_low_trailing_stop": False, "use_chandelier_exit": False, "use_breakeven_stop": False, "use_giveback_stop": False}
            td.SYSTEMS = [system]
            feats = td.build_features(raw)
            feats.attrs["asset"] = "TEST"
            assert abs(td.compute_atr(raw, 20).iloc[50] - 2.0) < 1e-9
            res = td.simulate_portfolio({"TEST": feats}, starting_capital=100_000.0)
            trades = res["trades_by_asset"]["TEST"]
            assert len(trades) == 1, len(trades)                      # the climb never touches the trail; the fall does, once
            t = trades[0]
            assert t["exit_reason"] == "peak_trail"
            # the level was set at bar 99's close: highest high 200 - 2 x ATR 2.0 = 196; a gap below it fills at the open
            fill = td.exit_fill_price(expected_raw, "TEST")
            assert abs(t["exit_price"] - fill) < 1e-9, (t["exit_price"], fill)
    # the trail is off by default, and the optimizer knows it
    assert td.DEFAULT_SYSTEM_PARAMS["use_peak_trail"] is False and td.STOP_SLIPPAGE_PCT == 0.0
    import optimize_turtle_dual_system as opt
    assert "use_peak_trail" in opt.PER_SYSTEM_CONDITIONAL_TOGGLES and "STOP_SLIPPAGE_PCT" in opt._PORTFOLIO_CONFIG_ATTRS


def test_slow_exit_is_opt_in():
    import optimize_turtle_dual_system as opt
    import robustness
    p = {"S1_BREAKOUT_LOOKBACK": 20, "S1_EXIT_LOOKBACK": 100, "S2_BREAKOUT_LOOKBACK": 200, "S2_EXIT_LOOKBACK": 100}
    systems = [{"breakout_lookback": 20, "exit_lookback": 100}, {"breakout_lookback": 200, "exit_lookback": 100}]
    saved = (td.ALLOW_EXIT_ABOVE_ENTRY, opt.ENABLE_S1, opt.ENABLE_S2)
    try:
        opt.ENABLE_S1 = opt.ENABLE_S2 = True
        td.ALLOW_EXIT_ABOVE_ENTRY = False
        assert not opt.is_valid_params(p) and not robustness._valid(systems)       # S1 leaves on a longer channel than it enters on
        td.ALLOW_EXIT_ABOVE_ENTRY = True
        assert opt.is_valid_params(p) and robustness._valid(systems)
        assert not opt.is_valid_params({**p, "S1_BREAKOUT_LOOKBACK": 250})          # S1 slower than S2 is still refused
        assert "ALLOW_EXIT_ABOVE_ENTRY" in opt._PORTFOLIO_CONFIG_ATTRS and "ALLOW_EXIT_ABOVE_ENTRY" in robustness.TD_CONFIG_NAMES
    finally:
        td.ALLOW_EXIT_ABOVE_ENTRY, opt.ENABLE_S1, opt.ENABLE_S2 = saved
    assert td.ALLOW_EXIT_ABOVE_ENTRY is False                                       # the engine default is the Turtle pairing


def test_defaults_unchanged_when_everything_is_off():
    """Realistic features all off must reproduce the original same-close, realized-equity engine."""
    features, _ = synthetic_market()
    with Settings():
        clean_engine()
        td.MARK_TO_MARKET = False
        td.EXECUTION_TIMING = "same_close"
        a = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        td.ASSET_COSTS = {}
        td.SIZE_STEP = {}
        b = td.simulate_portfolio({"TEST": features}, starting_capital=100_000.0)
        assert a["final_equity"] == b["final_equity"]
        assert a["margin_calls"] == [] and a["n_rounding_skips"] == 0 and a["n_margin_blocked"] == 0


def test_profiles_compile():
    paths = brokers.list_profiles()
    assert "Interactive Brokers (futures)" in paths and "TMGM Edge (CFD, swap-free)" in paths
    assets = ["SP500", "Gold", "Nasdaq", "Cocoa"]
    ibkr = brokers.load(paths["Interactive Brokers (futures)"])
    micro = brokers.compile_profile(ibkr, assets, brokers.SIZING_MICRO, margin_calls=True)["engine"]
    mini = brokers.compile_profile(ibkr, assets, brokers.SIZING_MINI, margin_calls=True)["engine"]
    assert micro["SIZE_STEP"]["SP500"] == 5.0 and mini["SIZE_STEP"]["SP500"] == 50.0      # MES vs ES
    assert micro["SIZE_STEP"]["Cocoa"] == 10.0                                           # no micro: standard
    assert micro["USE_PER_CONTRACT_FEES"] and micro["MARGIN_SPEC"]["SP500"][1] < micro["MARGIN_SPEC"]["SP500"][0]
    stressed = brokers.compile_profile(ibkr, assets, brokers.SIZING_MICRO, margin_calls=True, margin_stress=2.0)["engine"]
    assert abs(stressed["MARGIN_SPEC"]["SP500"][0] - 2 * micro["MARGIN_SPEC"]["SP500"][0]) < 1e-12
    tmgm = brokers.load(paths["TMGM Edge (CFD, swap-free)"])
    cfd = brokers.compile_profile(tmgm, assets, brokers.SIZING_CFD, margin_calls=True)["engine"]
    assert not cfd["USE_PER_CONTRACT_FEES"] and cfd["SIZE_STEP"]["SP500"] == 0.01
    assert abs(cfd["MARGIN_SPEC"]["SP500"][0] - 1 / 20) < 1e-12 and abs(cfd["MARGIN_SPEC"]["SP500"][1] - 0.025) < 1e-12
    # table edits flow through and mark the number unverified
    rows = brokers.table_rows(ibkr, assets, brokers.SIZING_MICRO)
    idx = [r["Asset"] for r in rows].index("SP500")
    edited = brokers.apply_edits(ibkr, assets, brokers.SIZING_MICRO, {str(idx): {"Fee per side ($)": 1.5}})
    assert abs(brokers.compile_profile(edited, assets, brokers.SIZING_MICRO)["engine"]["FEE_SCHEDULE"]["SP500"]["fee_per_contract"] - 1.5) < 1e-12
    assert brokers.load(paths["Interactive Brokers (futures)"])["instruments"]["SP500"]["micro"]["verified"] is True


def test_ibkr_disclosure_grid_fees():
    """The IBKR profile reproduces the totals printed in IBKR's Costs and Charges grids (Jan 2024 futures,
    FX CFD; Sep 2024 Paxos crypto): the entry-and-exit cost of one contract, converted with the grid's own
    exchange rate, equals the grid's total in EUR."""
    import ibkr_fees
    paths = brokers.list_profiles()
    ibkr = brokers.load(paths["Interactive Brokers (futures)"])
    # (asset, grid fx per EUR, grid total in EUR for entry and exit, multiplier, margin in the contract's currency, order value)
    grid = [("CAC40", 1.0, 4.00, 10, 4812, 76290), ("FTSE100", 0.88, 3.86, 10, 4166, 74950),
            ("SMI", 0.983, 8.14, 10, 6422, 109550), ("SP500", 1.058, 4.16, 50, 8624, 223850),
            ("HangSeng", 8.26, 7.26, 50, 72620, 819100), ("ASX200", 1.59, 6.29, 25, 11997, 177250)]
    for asset, per_eur, total_eur, mult, margin, order_value in grid:
        spec = brokers._contract(ibkr, asset, brokers.SIZING_MINI)
        assert spec["multiplier"] == mult, asset
        assert abs(2 * brokers.fee_per_side(spec) / per_eur - total_eur) < 0.011, asset
        if "margin_snapshot" not in spec:     # contracts in IBKR's later margin table carry that margin instead
            assert abs(spec["margin_pct"] - margin / order_value * 100) < 0.01, asset
    assert abs(brokers.fee_per_side(brokers._contract(ibkr, "Ger40", brokers.SIZING_MINI)) - 2.0) < 1e-12
    # the older fee table the walk-forward uses must agree with the profile where the grid speaks
    for asset in ("SP500", "Ger40"):
        legacy = ibkr_fees.FEE_TABLE[asset]["standard"]
        assert abs(ibkr_fees.fee_per_contract(legacy) - brokers.fee_per_side(brokers._contract(ibkr, asset, brokers.SIZING_MINI))) < 1e-12

    crypto_fx = brokers.load(paths["Interactive Brokers (CFDs + spot crypto)"])
    out = brokers.compile_profile(crypto_fx, ["Bitcoin", "EURUSD", "Cocoa"], brokers.SIZING_CFD, margin_calls=True)
    eng = out["engine"]
    assert not eng["USE_PER_CONTRACT_FEES"]
    assert eng["ASSET_COSTS"]["Bitcoin"]["commission_pct"] == 0.18 and eng["ASSET_COSTS"]["Bitcoin"]["holding_pct"] == 0.0
    assert eng["MARGIN_SPEC"]["Bitcoin"][0] == 1.0                                  # fully paid, no leverage
    assert abs(eng["MARGIN_SPEC"]["EURUSD"][0] - 0.0333) < 1e-12                    # retail: CBI minimum 3.33% initial
    assert abs(eng["MARGIN_SPEC"]["EURUSD"][1] - 0.03) < 1e-12                      # IBKR's 3% maintenance, whatever the stop-out
    stressed = brokers.compile_profile(crypto_fx, ["EURUSD"], brokers.SIZING_CFD, margin_calls=True,
                                       margin_stress=2.0, stop_out_pct=80.0)["engine"]["MARGIN_SPEC"]["EURUSD"]
    assert abs(stressed[0] - 0.0666) < 1e-12 and abs(stressed[1] - 0.06) < 1e-12    # stress raises both
    edited = brokers.apply_edits(crypto_fx, ["Bitcoin", "EURUSD"], brokers.SIZING_CFD, {"1": {"Leverage (x)": 10.0}})
    assert "maintenance_pct" not in edited["instruments"]["EURUSD"]                 # an edited leverage drops the stated rate
    assert any("Bitcoin: fully paid" in n for n in out["notes"]) and any("Cocoa" in n for n in out["notes"])
    assert any("lot_step" in n for n in out["notes"]) and not out["estimates"]
    with Settings():
        for name, value in eng.items():
            setattr(td, name, value)
        # grid: 10,000 EUR of Bitcoin costs 36.00 for entry and exit -> 18.00 on each fill
        assert abs(td.commission_cost(10000.0, "Bitcoin") - 18.0) < 1e-9
        # grid: 300,300 EUR of EUR.USD costs 12.01 commission for entry and exit, and 137.46 financing over 5 days
        assert abs(2 * td.commission_cost(300300.0, "EURUSD") - 12.01) < 0.01
        five_days = td.daily_holding_cost(300300.0, "EURUSD", calendar_days=5.0)
        assert abs(five_days - 137.46) < 0.2, five_days


def test_ibkr_cfd_margins():
    """Index CFDs: maintenance = the related future's overnight maintenance (minimum 5%), initial = 1.25 x that, never below the
    Central Bank of Ireland minimum; metals CFDs take IBKR's rates or the CBI minimum, whichever is higher."""
    paths = brokers.list_profiles()
    fut = brokers.load(paths["Interactive Brokers (futures)"])
    cfd = brokers.load(paths["Interactive Brokers (CFDs + spot crypto)"])
    assets = ["SP500", "Nasdaq", "Dow", "Ger40", "Nikkei225", "Gold", "Silver"]
    spec = brokers.compile_profile(cfd, assets, brokers.SIZING_CFD, margin_calls=True)["engine"]["MARGIN_SPEC"]
    for asset, name in (("SP500", "ES"), ("Nasdaq", "NQ"), ("Dow", "YM"), ("Ger40", "FDAX")):
        maint = max(fut["instruments"][asset]["standard"]["maintenance_pct"], 5.0)
        assert abs(spec[asset][1] * 100 - maint) < 0.006, asset
        assert abs(spec[asset][0] * 100 - 1.25 * maint) < 0.006, asset
    assert abs(spec["Nikkei225"][1] - 0.05) < 1e-12 and abs(spec["Nikkei225"][0] - 0.0625) < 1e-12   # the future's 4.78% is below the 5% minimum
    assert abs(spec["Gold"][0] - 0.0625) < 1e-12 and abs(spec["Gold"][1] - 0.05) < 1e-12
    assert abs(spec["Silver"][0] - 0.20) < 1e-12 and abs(spec["Silver"][1] - 0.10) < 1e-12           # CBI 20% beats IBKR's 14.85%
    assert all(spec[a][0] >= 0.05 and spec[a][1] >= 0.025 for a in assets)                           # the CBI floors for major indices
    out = brokers.compile_profile(cfd, assets, brokers.SIZING_CFD)
    assert set(out["estimates"]) == set(assets)                                                      # their costs are not in the documents


def test_ibkr_margin_table():
    """Margins in the IBKR profile are IBKR's overnight dollar margins over contract value (price x multiplier);
    the engine gets the broker's own initial and maintenance rates, which scale with the margin stress."""
    ibkr = brokers.load(brokers.list_profiles()["Interactive Brokers (futures)"])
    checked = 0
    for asset, options in ibkr["instruments"].items():
        for size, spec in options.items():
            snap = spec.get("margin_snapshot")
            if not snap:
                continue
            value = snap["reference_price"] * spec["multiplier"]
            assert abs(spec["margin_pct"] - snap["overnight_initial"] / value * 100) < 0.006, (asset, size)
            assert abs(spec["maintenance_pct"] - snap["overnight_maintenance"] / value * 100) < 0.006, (asset, size)
            assert 0.6 < spec["maintenance_pct"] / spec["margin_pct"] < 1.0, (asset, size)
            checked += 1
    assert checked >= 25
    es = ibkr["instruments"]["SP500"]["standard"]
    assert es["margin_snapshot"]["overnight_initial"] == 27556.25 and es["margin_snapshot"]["overnight_maintenance"] == 21974
    out = brokers.compile_profile(ibkr, ["SP500", "Nasdaq", "CAC40"], brokers.SIZING_MINI, margin_calls=True)["engine"]["MARGIN_SPEC"]
    assert abs(out["SP500"][0] - es["margin_pct"] / 100) < 1e-12 and abs(out["SP500"][1] - es["maintenance_pct"] / 100) < 1e-12
    assert abs(out["SP500"][1] / out["SP500"][0] - 0.797) < 0.003 and abs(out["Nasdaq"][1] / out["Nasdaq"][0] - 0.706) < 0.003
    assert abs(out["CAC40"][1] / out["CAC40"][0] - 0.9) < 1e-12                      # no stated rate: the profile's default ratio
    stressed = brokers.compile_profile(ibkr, ["SP500"], brokers.SIZING_MINI, margin_calls=True, margin_stress=2.0)["engine"]["MARGIN_SPEC"]["SP500"]
    assert abs(stressed[0] - 2 * out["SP500"][0]) < 1e-12 and abs(stressed[1] - 2 * out["SP500"][1]) < 1e-12
    rows = brokers.table_rows(ibkr, ["SP500"], brokers.SIZING_MINI)
    edited = brokers.apply_edits(ibkr, ["SP500"], brokers.SIZING_MINI, {"0": {"Initial margin (%)": 2 * es["margin_pct"]}})
    new = edited["instruments"]["SP500"]["standard"]
    assert abs(new["maintenance_pct"] / new["margin_pct"] - es["maintenance_pct"] / es["margin_pct"]) < 1e-12   # ratio kept
    assert rows[0]["Initial margin (%)"] == es["margin_pct"] and edited["instruments"]["SP500"]["micro"]["margin_pct"] == ibkr["instruments"]["SP500"]["micro"]["margin_pct"]


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
