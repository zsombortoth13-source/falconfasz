"""Walk-forward validation for the mean-reversion strategy (mean_reversion.py).

Same method as walk_forward_turtle.py: for every window the settings are searched on the TRAIN years
only, the chosen set (middle of the top N, not the single luckiest) then trades the following TEST
year it has never seen, and the test years are stitched into one out-of-sample curve.

Windows are built from a fixed overall start (default: the first date any market in the database has)
so different market universes get IDENTICAL windows and their out-of-sample curves line up in time. A
market is left out of a window until it has MIN_BARS bars of history before that window's cutoff.
"""
import multiprocessing as mp
import time
import traceback

import numpy as np
import pandas as pd

import mean_reversion as mr
import turtle_dual_system as td
import walk_forward_turtle as wf

MIN_BARS = 320          # a market joins a window only with this much history (the 300-bar trend average needs it)
WARMUP_YEARS = 3        # simulate from this long before the train start, so indicators and state are warm

_STATE = {}


def _init_worker(full_data: dict, cfg: dict):
    mr.apply_cost_config(cfg["costs"])
    _STATE.update(full=full_data, cfg=cfg)


def _slice(full: dict, upto: pd.Timestamp, start: pd.Timestamp) -> dict:
    out = {}
    for name, df in full.items():
        head = df.loc[:upto]
        if len(head) < MIN_BARS:
            continue
        out[name] = head.loc[start:]
        if len(out[name]) < 60:
            out.pop(name)
    return out


def _run_window(args: tuple) -> dict:
    wi, w = args
    try:
        return _run_window_inner(wi, w)
    except Exception:
        return {"wi": wi, "status": "error", "w": w, "error": traceback.format_exc()}


def _run_window_inner(wi: int, w: dict) -> dict:
    full, cfg = _STATE["full"], _STATE["cfg"]
    capital = cfg["capital"]
    sim_start = w["train_start"] - pd.DateOffset(years=WARMUP_YEARS)
    raw_is = _slice(full, w["train_end"], sim_start)
    if not raw_is:
        return {"wi": wi, "status": "no_data", "w": w}
    data = mr.MRData(raw_is)
    results = mr.search(data, cfg["n_trials"], seed=cfg["seed"] + wi, starting_capital=capital,
                        score_start=w["train_start"], score_end=w["train_end"], metric=cfg["metric"],
                        min_trades=cfg["min_trades"], space=cfg.get("space"), pinned=cfg.get("pinned"))
    if not results:
        return {"wi": wi, "status": "no_valid_trial", "w": w}
    n_pick = min(max(int(cfg["pick_middle_n"]), 1), len(results))
    chosen = results[(n_pick - 1) // 2]
    ensemble_k = int(cfg.get("ensemble_k") or 1)
    raw_oos = _slice(full, w["test_end"], sim_start)
    data_oos = mr.MRData(raw_oos)
    if ensemble_k > 1:
        # trade the K best settings side by side, capital split equally: the average of their daily returns
        members = results[:ensemble_k]
        daily, all_trades = [], []
        for m in members:
            res_k = mr.simulate(data_oos, m["params"], capital, record_trades=True)
            c_k = res_k["equity_curve"]
            window_k = c_k.loc[w["test_start"]:w["test_end"]].dropna()
            prior_k = wf._prior_value(c_k, w["test_start"])
            if len(window_k) < 20 or window_k.iloc[0] <= 0:
                continue
            first = (window_k.iloc[0] / prior_k - 1.0) if prior_k else 0.0
            r_k = window_k.pct_change()
            r_k.iloc[0] = first
            daily.append(r_k)
            all_trades += [t for t in res_k["trades"] if w["test_start"] <= t["exit_date"] <= w["test_end"]]
        if not daily:
            return {"wi": wi, "status": "no_oos", "w": w, "params": chosen["params"]}
        mean_r = pd.concat(daily, axis=1).mean(axis=1)
        oos = capital * (1.0 + mean_r).cumprod()
        prior, trades = capital, all_trades
    else:
        result = mr.simulate(data_oos, chosen["params"], capital, record_trades=True)
        curve = result["equity_curve"]
        oos = curve.loc[w["test_start"]:w["test_end"]].dropna()
        if len(oos) < 20 or oos.iloc[0] <= 0:
            return {"wi": wi, "status": "no_oos", "w": w, "params": chosen["params"]}
        prior = wf._prior_value(curve, w["test_start"])
        trades = [t for t in result["trades"] if w["test_start"] <= t["exit_date"] <= w["test_end"]]
    stats = td.performance_stats(oos, float(oos.iloc[0]))
    return {"wi": wi, "status": "ok", "w": w, "params": chosen["params"], "oos_curve": oos,
            "oos_prior": prior, "trades": trades,
            "is_sharpe": chosen["sharpe"], "is_cagr": chosen["cagr"], "is_trades": chosen["n_trades"],
            "n_markets": data.A, "top_is_sharpe": results[0]["sharpe"],
            "oos_sharpe": float(stats["Sharpe"]), "oos_cagr": float(stats["CAGR"]),
            "oos_max_dd": float(stats["Max Drawdown"]), "oos_trades": len(trades)}


def run_walk_forward(full_data: dict, n_trials: int = 80, train_years: int = 5, test_years: int = 1,
                     n_workers: int = 6, seed: int = 42, metric: str = "sharpe", pick_middle_n: int = 5,
                     min_trades: int = 40, capital: float = 100000.0, overall_start=None, overall_end=None,
                     space: dict = None, pinned: dict = None, verbose: bool = True, ensemble_k: int = 1) -> dict:
    wf.TRAIN_YEARS, wf.TEST_YEARS, wf.ANCHORED_TRAIN = train_years, test_years, False
    overall_start = overall_start or min(df.index.min() for df in full_data.values())
    overall_end = overall_end or max(df.index.max() for df in full_data.values())
    windows = wf.build_windows(overall_start, overall_end)
    cfg = {"n_trials": n_trials, "seed": seed, "metric": metric, "pick_middle_n": pick_middle_n,
           "min_trades": min_trades, "capital": capital, "costs": mr.current_cost_config(),
           "space": space, "pinned": pinned, "ensemble_k": ensemble_k}
    tasks = list(enumerate(windows, start=1))
    t0 = time.time()
    if n_workers > 1:
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(full_data, cfg)) as pool:
            done = pool.map(_run_window, tasks, chunksize=1)
    else:
        _init_worker(full_data, cfg)
        done = [_run_window(t) for t in tasks]
    done.sort(key=lambda r: r["wi"])
    ok = [r for r in done if r["status"] == "ok"]
    errors = [r for r in done if r["status"] == "error"]
    if errors and verbose:
        print("window errors:\n" + errors[0]["error"])
    out = {"windows": done, "ok": ok, "elapsed": time.time() - t0, "n_windows": len(windows)}
    if not ok:
        return out
    stitched = wf.stitch_oos_windows([{"MR": r["oos_curve"]} for r in ok], capital, [{"MR": r["oos_prior"]} for r in ok])
    curve = stitched["MR"]
    out["stitched"] = curve
    out["stats"] = td.performance_stats(curve, capital)
    out["trades"] = [t for r in ok for t in r["trades"]]
    out["mean_is_sharpe"] = float(np.mean([r["is_sharpe"] for r in ok]))
    out["mean_oos_sharpe"] = float(np.mean([r["oos_sharpe"] for r in ok]))
    out["positive_windows"] = int(sum(r["oos_sharpe"] > 0 for r in ok))
    return out
