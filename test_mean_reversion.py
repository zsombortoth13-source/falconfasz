"""Hand-checkable tests for mean_reversion.py. Run: .venv/Scripts/python.exe test_mean_reversion.py"""
import copy
import sys

import numpy as np
import pandas as pd

import mean_reversion as mr
import turtle_dual_system as td

COST_NAMES = mr.COST_SETTING_NAMES


class CleanCosts:
    """Zero costs, no per-asset overrides, restored afterwards."""
    def __enter__(self):
        self.saved = {n: copy.deepcopy(getattr(td, n)) for n in COST_NAMES}
        td.SLIPPAGE_PCT = td.SPREAD_PCT = td.COMMISSION_PCT = td.HOLDING_COST_PCT_PER_YEAR = 0.0
        td.USE_PER_CONTRACT_FEES = False
        td.FEE_SCHEDULE, td.ASSET_COSTS, td.SIZE_STEP = {}, {}, {}
        return self

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            setattr(td, n, v)


def bars(rows, start="2020-01-01"):
    """rows: list of (open, high, low, close)."""
    index = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)
    df.attrs["asset"] = "X"
    return df


FLAT = [(100.0, 101.0, 99.0, 100.0)] * 10          # ten identical days: ATR(5) settles at 2.0 exactly

# params that make the rules trivial to follow: a new 3-day closing low buys, close above the 2-day average sells
BASE = dict(mr.DEFAULT_PARAMS, direction="long", entry_type="nlow", nlow_days=3, trend_sma=0, exit_type="sma",
            exit_sma=2, max_hold=10, atr_period=5, stop_atr=2.0, risk_pct=1.0, max_pos_pct=1000.0,
            max_gross_pct=10000.0, max_positions=10)


def run(rows, p=None, capital=100000.0, record=True):
    data = mr.MRData({"X": bars(rows)})
    return mr.simulate(data, dict(BASE, **(p or {})), capital, record_trades=record), data


def test_long_trade_by_hand():
    # day 10 closes at 97, a new 3-day closing low (previous closes were 100) -> buy at day 11's OPEN (97.5).
    # day 11 closes 98: above its 2-day average (97.5) -> sell at day 12's OPEN (99.0).
    rows = FLAT + [(100.0, 100.0, 96.5, 97.0), (97.5, 98.5, 97.0, 98.0), (99.0, 99.5, 98.5, 99.0), (99.0, 99.5, 98.5, 99.0)]
    with CleanCosts():
        res, data = run(rows)
        atr_at_signal = float(data.atr(5)[0, 10])
        assert abs(atr_at_signal - (0.8 * 2.0 + 0.2 * 3.5)) < 1e-9          # TR on day 10 is 3.5
        (trade,) = res["trades"]
        assert trade["side"] == "long" and trade["reason"] == "signal"
        assert trade["entry_date"] == data.dates[11] and trade["exit_date"] == data.dates[12]
        assert trade["entry_price"] == 97.5 and trade["exit_price"] == 99.0
        shares = 100000.0 * 0.01 / (2.0 * atr_at_signal)                    # risk 1% of equity over a 2-ATR stop
        assert abs(trade["shares"] - shares) < 1e-6
        assert abs(trade["pnl"] - shares * 1.5) < 1e-6
        assert abs(res["equity_curve"].iloc[-1] - (100000.0 + shares * 1.5)) < 1e-6


def test_stop_and_gap_fills():
    atr = 0.8 * 2.0 + 0.2 * 3.5                       # 2.3 at the signal
    entry = 97.5
    stop = entry - 2.0 * atr                          # 92.9
    # intraday: day 11 trades down through the stop and closes lower -> filled at the stop level itself
    rows = FLAT + [(100.0, 100.0, 96.5, 97.0), (97.5, 97.6, 90.0, 91.0), (91.0, 92.0, 90.0, 91.0), (91.0, 92.0, 90.0, 91.0)]
    with CleanCosts():
        res, _ = run(rows)
        trade = res["trades"][0]
        assert trade["reason"] == "stop" and abs(trade["exit_price"] - stop) < 1e-9
        assert abs(trade["pnl"] - trade["shares"] * (stop - entry)) < 1e-6
        # overnight gap: day 12 OPENS below the stop -> filled at that open, worse than the stop
        rows = FLAT + [(100.0, 100.0, 96.5, 97.0), (97.5, 98.0, 97.0, 97.4), (90.0, 91.0, 89.0, 90.0), (90.0, 91.0, 89.0, 90.0)]
        res, _ = run(rows, p={"exit_type": "prev_extreme"})
        trade = res["trades"][0]
        assert trade["reason"] == "stop" and trade["exit_price"] == 90.0 and trade["exit_price"] < stop


def test_short_mirror():
    # a new 3-day closing HIGH at 103, direction both, no trend filter -> short at the next open
    rows = FLAT + [(100.0, 103.5, 100.0, 103.0), (102.5, 103.0, 101.5, 102.0), (101.0, 101.5, 100.5, 101.0), (101.0, 101.5, 100.5, 101.0)]
    with CleanCosts():
        res, data = run(rows, p={"direction": "both"})
        (trade,) = res["trades"]
        assert trade["side"] == "short" and trade["entry_price"] == 102.5
        # close 102 < its 2-day average (102.5) -> cover at day 12's open 101.0, a profit of 1.5 a share
        assert trade["exit_price"] == 101.0
        assert abs(trade["pnl"] - trade["shares"] * 1.5) < 1e-6
    # long-only never takes it
    with CleanCosts():
        res, _ = run(rows, p={"direction": "long"})
        assert res["trades"] == []


def test_costs_and_books():
    rows = FLAT + [(100.0, 100.0, 96.5, 97.0), (97.5, 98.5, 97.0, 98.0), (99.0, 99.5, 98.5, 99.0), (99.0, 99.5, 98.5, 99.0)]
    with CleanCosts():
        td.COMMISSION_PCT = 0.1                                           # 0.1% of trade value per side
        td.SLIPPAGE_PCT, td.SPREAD_PCT = 0.2, 0.0
        res, data = run(rows)
        (trade,) = res["trades"]
        buy, sell = 97.5 * 1.002, 99.0 * 0.998                              # slippage hurts both ways
        assert abs(trade["entry_price"] - buy) < 1e-9 and abs(trade["exit_price"] - sell) < 1e-9
        shares = trade["shares"]
        commission = 0.001 * shares * buy + 0.001 * shares * sell
        assert abs(res["total_commission_cost"] - commission) < 1e-6
        assert abs(trade["pnl"] - (shares * (sell - buy) - commission)) < 1e-6
        assert abs(res["equity_curve"].iloc[-1] - (100000.0 + trade["pnl"])) < 1e-6   # the books balance
        # per-asset costs replace the globals for that asset only
        td.ASSET_COSTS = {"X": {"slippage_pct": 0.0, "commission_pct": 0.0}}
        res2, _ = run(rows)
        assert abs(res2["trades"][0]["entry_price"] - 97.5) < 1e-9 and res2["total_commission_cost"] == 0.0


def test_sizing_caps_and_rounding():
    rows = FLAT + [(100.0, 100.0, 96.5, 97.0), (97.5, 98.5, 97.0, 98.0), (99.0, 99.5, 98.5, 99.0), (99.0, 99.5, 98.5, 99.0)]
    with CleanCosts():
        res, data = run(rows)
        base_shares = res["trades"][0]["shares"]
        # risk sizing alone buys about 21,000 of notional; a position cap of 10% of equity limits it to 10,000
        assert base_shares * 97.5 > 20000.0
        res, _ = run(rows, p={"max_pos_pct": 10.0})
        assert abs(res["trades"][0]["shares"] * 97.5 - 10000.0) < 1e-6
        # a whole-contract step of 100 shares: rounded DOWN, and skipped when it would be zero
        td.SIZE_STEP = {"X": 100.0}
        res, _ = run(rows)
        assert res["trades"][0]["shares"] % 100.0 == 0 and res["trades"][0]["shares"] <= base_shares
        td.SIZE_STEP = {"X": 100000.0}
        res, _ = run(rows)
        assert res["trades"] == [] and res["skipped_rounding"] == 1
        td.SIZE_STEP = {}
        res, _ = run(rows, p={"max_positions": 0})
        assert res["trades"] == [] and res["skipped_max_positions"] == 1


def test_time_limit_and_holding_cost():
    # a dip that never recovers above the 2-day average: sold at the open after max_hold bars
    rows = FLAT + [(100.0, 100.0, 96.5, 97.0)] + [(97.0, 97.2, 96.8, 97.0 - 0.01 * k) for k in range(1, 12)]
    with CleanCosts():
        res, data = run(rows, p={"max_hold": 4, "stop_atr": 50.0})
        trade = res["trades"][0]
        assert trade["reason"] == "time limit" and trade["bars_held"] == 4
        td.HOLDING_COST_PCT_PER_YEAR = 3.65                                  # 0.01% of value per calendar day
        res, _ = run(rows, p={"max_hold": 4, "stop_atr": 50.0})
        assert res["total_holding_cost"] > 0
        total_pnl = sum(t["pnl"] for t in res["trades"])
        assert abs(res["equity_curve"].iloc[-1] - (100000.0 + total_pnl - res["total_holding_cost"])) < 1e-6


def test_no_lookahead():
    """Truncating the future changes nothing that already happened."""
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 900)))
    opens = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, 900))
    frame = pd.DataFrame({"Open": opens, "High": np.maximum(opens, close) * 1.004, "Low": np.minimum(opens, close) * 0.996,
                          "Close": close}, index=pd.bdate_range("2015-01-01", periods=900))
    p = dict(mr.DEFAULT_PARAMS, trend_sma=100, stop_atr=3.0)
    with CleanCosts():
        td.COMMISSION_PCT, td.SLIPPAGE_PCT = 0.01, 0.02
        full = mr.simulate(mr.MRData({"X": frame}), p, 100000.0)["equity_curve"]
        for cut in (400, 650):
            part = mr.simulate(mr.MRData({"X": frame.iloc[:cut]}), p, 100000.0)["equity_curve"]
            assert np.allclose(part.iloc[:-1].values, full.iloc[:cut - 1].values, rtol=0, atol=1e-8), cut


def test_search_runs_and_scores():
    rng = np.random.default_rng(9)
    frames = {}
    for k in range(3):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 1500)))
        opens = np.concatenate([[100.0], close[:-1]])
        frames[f"M{k}"] = pd.DataFrame({"Open": opens, "High": np.maximum(opens, close) * 1.003,
                                        "Low": np.minimum(opens, close) * 0.997, "Close": close},
                                       index=pd.bdate_range("2014-01-01", periods=1500))
    data = mr.MRData(frames)
    results = mr.search(data, n_trials=12, seed=1, min_trades=10)
    assert results and all("curve" in r and r["n_trades"] >= 10 for r in results)
    assert results == sorted(results, key=lambda r: r["sharpe"], reverse=True)
    again = mr.search(data, n_trials=12, seed=1, min_trades=10)
    assert [r["sharpe"] for r in again] == [r["sharpe"] for r in results]       # same seed, same answer


def test_walk_forward_smoke():
    """Windows, search, ensemble and stitching run end to end on synthetic markets, deterministically."""
    import mr_walk_forward as mwf
    rng = np.random.default_rng(21)
    frames = {}
    for k in range(3):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, 3300)))
        opens = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.001, 3300))
        frames[f"M{k}"] = pd.DataFrame({"Open": opens, "High": np.maximum(opens, close) * 1.003,
                                        "Low": np.minimum(opens, close) * 0.997, "Close": close},
                                       index=pd.bdate_range("2004-01-01", periods=3300))
    space = dict(mr.PARAM_SPACE, direction=["long"], entry_type=["rsi"], rsi_period=[2, 3], rsi_entry=[10.0, 20.0],
                 trend_sma=[100], exit_type=["sma"], exit_sma=[5], max_hold=[10], atr_period=[14], stop_atr=[3.0])
    with CleanCosts():
        runs = [mwf.run_walk_forward(frames, n_trials=6, train_years=3, test_years=1, n_workers=1, pick_middle_n=2,
                                     min_trades=5, space=space, ensemble_k=k, verbose=False) for k in (1, 3, 3)]
    single, ens, ens_again = runs
    assert len(single["ok"]) >= 4 and len(ens["ok"]) == len(single["ok"])
    for out in (single, ens):
        curve = out["stitched"]
        assert curve.index.is_monotonic_increasing and curve.index.is_unique and (curve > 0).all()
        assert curve.index[0] >= out["ok"][0]["w"]["test_start"]
    assert np.allclose(ens["stitched"].values, ens_again["stitched"].values)        # same seeds, same answer
    # the first out-of-sample window starts after its own train window: nothing before it is scored
    first = ens["ok"][0]["w"]
    assert ens["stitched"].index[0] > first["train_end"]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
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
