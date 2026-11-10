"""
Isolated Optimization for the Dual-System Turtle Strategy
=============================================================
Unlike optimize_turtle_dual_system.py (which samples a FULL parameter set
for BOTH systems together in every trial, so a single trial's score reflects
the joint outcome of that S1 config paired with that S2 config), this script
tests each system COMPLETELY ALONE:

  1. Optimize System 1 by itself -- System 2 is entirely absent (td.SYSTEMS
     contains only S1). N_TRIALS_PER_SYSTEM random trials, scored the same
     per-asset-normalized way as the main optimizer.
  2. Optimize System 2 by itself -- System 1 entirely absent. Same process.
  3. Take the two winners and run ONE final combined backtest with both
     together, so you can see how they actually perform as a pair -- and,
     importantly, how much (if any) the shared exposure cap changes things
     now that the two systems have to compete for the same budget, which
     didn't happen during either system's isolated test.

WHY THIS IS MORE EFFICIENT: searching S1's space and S2's space TOGETHER
(as the main optimizer does) means the effective space is roughly the
PRODUCT of the two systems' individual spaces -- most trials are essentially
random pairings, so a great S1 config can get diluted by an unlucky S2
pairing in the same trial. Searching them separately needs roughly
N_TRIALS_PER_SYSTEM x 2 trials total instead of needing N^2 to get
comparable coverage of each system's own space.

THE HONEST LIMITATION: S1 and S2 aren't perfectly independent even in
reality -- they share the same equity pool and compete for the same
MAX_TOTAL_EXPOSURE_PCT cap. During each system's ISOLATED test, it gets the
entire exposure budget to itself. Once combined, some exposure-cap trimming
can occur that didn't happen in either isolated test. The final combined
backtest step exists specifically to surface and quantify this -- it prints
each system's Sharpe as it performed ALONE vs. how it performs WITHIN the
combined run, so you can see exactly how much (if any) interaction there was.

Requires turtle_dual_system.py and optimize_turtle_dual_system.py in the
same folder (this script reuses their parameter-space definitions and cost
model rather than redefining them).
"""

import os
import random
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

import turtle_dual_system as td
import optimize_turtle_dual_system as opt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
N_TRIALS_PER_SYSTEM = 400
RANDOM_SEED = 0
N_WORKERS = None

SCORE_AGGREGATION = "median"   # see optimize_turtle_dual_system.py for the full rationale
RANK_METRIC = "cagr"

# Breakout/exit lookback candidates per system "role" -- reused from the main optimizer's
# search space, just without the S1_/S2_ prefix (each is tested completely on its own now,
# so there's no cross-system ordering constraint to enforce -- only exit < breakout, per system)
LOOKBACK_SPACE = {
    "S1": {"breakout_lookback": [10, 15, 20, 25, 30], "exit_lookback": [5, 10, 15]},
    "S2": {"breakout_lookback": [40, 55, 70, 100], "exit_lookback": [15, 20, 25, 30]},
}


# ----------------------------------------------------------------------------
# SAMPLING (one system's full parameter set, unprefixed)
# ----------------------------------------------------------------------------
def is_valid_single_system_params(params: dict) -> bool:
    return params["exit_lookback"] < params["breakout_lookback"]


def sample_single_system_params_raw(rng: random.Random, system_role: str) -> dict:
    params = {name: rng.choice(values) for name, values in LOOKBACK_SPACE[system_role].items()}
    for name, values in opt.PER_SYSTEM_ALWAYS_SAMPLED.items():
        params[name] = rng.choice(values)
    for toggle_name, sub_params in opt.PER_SYSTEM_CONDITIONAL_TOGGLES.items():
        is_on = rng.choice([True, False])
        params[toggle_name] = is_on
        if is_on:
            for sub_name, sub_values in sub_params.items():
                params[sub_name] = rng.choice(sub_values)
    # global/account-level toggles -- relevant even for a single isolated system, since it's
    # still trading in a real account with its own equity curve and drawdowns
    for toggle_name, sub_params in opt.GLOBAL_CONDITIONAL_TOGGLES.items():
        is_on = rng.choice([True, False])
        params[toggle_name] = is_on
        if is_on:
            for sub_name, sub_values in sub_params.items():
                params[sub_name] = rng.choice(sub_values)
    return params


def sample_unique_single_system_params(rng: random.Random, system_role: str, n: int,
                                         max_attempts_mult: int = 30) -> list:
    seen = set()
    unique_params = []
    attempts = 0
    max_attempts = n * max_attempts_mult
    while len(unique_params) < n and attempts < max_attempts:
        attempts += 1
        params = sample_single_system_params_raw(rng, system_role)
        if not is_valid_single_system_params(params):
            continue
        signature = frozenset(params.items())
        if signature in seen:
            continue
        seen.add(signature)
        unique_params.append(params)
    if len(unique_params) < n:
        print(f"  (note: only found {len(unique_params)} valid unique combinations -- filling the rest with repeats)")
        while len(unique_params) < n:
            unique_params.append(sample_single_system_params_raw(rng, system_role))
    return unique_params


def params_to_single_system(params: dict, system_name: str) -> dict:
    full = dict(td.DEFAULT_SYSTEM_PARAMS)
    p = dict(params)
    breakout_lookback = p.pop("breakout_lookback")
    exit_lookback = p.pop("exit_lookback")
    # only fold in keys that are genuinely per-system params -- global toggles (e.g.
    # USE_DRAWDOWN_SCALING) get applied separately via setattr, not folded into the system dict
    system_param_keys = set(td.DEFAULT_SYSTEM_PARAMS.keys())
    p = {k: v for k, v in p.items() if k in system_param_keys}
    full.update(p)
    return {"name": system_name, "breakout_lookback": breakout_lookback, "exit_lookback": exit_lookback, **full}


# ----------------------------------------------------------------------------
# TRIAL EXECUTION -- runs the backtest with ONLY this one system in td.SYSTEMS
# ----------------------------------------------------------------------------
def score_single_system_trial(raw_data: dict, params: dict, system_name: str) -> dict | None:
    system_dict = params_to_single_system(params, system_name)
    td.SYSTEMS = [system_dict]   # the other system genuinely does not exist for this trial
    for toggle_name, sub_params in opt.GLOBAL_CONDITIONAL_TOGGLES.items():
        if toggle_name in params:
            setattr(td, toggle_name, params[toggle_name])
            for sub_name in sub_params:
                if sub_name in params:
                    setattr(td, sub_name, params[sub_name])
    try:
        features = {name: td.build_features(df) for name, df in raw_data.items()}
        results = {name: td.simulate_asset(df) for name, df in features.items()}
    except Exception:
        return None

    total_trades = sum(len(r["trades"]) for r in results.values())
    if total_trades == 0:
        return None

    per_asset_metric = {}
    for name, r in results.items():
        try:
            stats = td.performance_stats(r["equity_curve"], td.STARTING_CAPITAL_PER_ASSET)
        except Exception:
            continue
        if not np.isfinite(stats.get("Sharpe", np.nan)):
            continue
        per_asset_metric[name] = {
            "sharpe": stats["Sharpe"], "calmar": stats["Calmar"] if np.isfinite(stats["Calmar"]) else 0.0,
            "cagr": stats["CAGR"], "max_dd": stats["Max Drawdown"],
        }
    if not per_asset_metric:
        return None

    def aggregate(key):
        values = np.array([m[key] for m in per_asset_metric.values()])
        if SCORE_AGGREGATION == "mean":
            return values.mean()
        elif SCORE_AGGREGATION == "median":
            return np.median(values)
        elif SCORE_AGGREGATION == "min":
            return values.min()
        raise ValueError(f"Unknown SCORE_AGGREGATION: {SCORE_AGGREGATION}")

    score = aggregate("sharpe")
    if not np.isfinite(score):
        return None

    return {
        "params": dict(params), "system_dict": system_dict,
        "sharpe": score, "calmar": aggregate("calmar"), "cagr": aggregate("cagr"),
        "worst_dd": min(m["max_dd"] for m in per_asset_metric.values()),
        "per_asset_metric": per_asset_metric, "total_trades": total_trades,
        "per_asset_trades": {name: len(r["trades"]) for name, r in results.items()},
    }


# ----------------------------------------------------------------------------
# MULTIPROCESSING WORKER PLUMBING
# ----------------------------------------------------------------------------
_worker_raw_data = None


def _init_worker(raw_data: dict):
    global _worker_raw_data
    _worker_raw_data = raw_data


def _run_worker(args):
    params, system_name = args
    return score_single_system_trial(_worker_raw_data, params, system_name)


def optimize_system_isolated(raw_data: dict, system_role: str, system_name: str, n_trials: int,
                               n_workers: int, rng: random.Random) -> tuple:
    param_list = sample_unique_single_system_params(rng, system_role, n_trials)
    results = []
    start_time = time.time()

    if n_workers > 1:
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(raw_data,)) as pool:
            for trial_num, result in enumerate(
                    pool.imap_unordered(_run_worker, [(p, system_name) for p in param_list]), start=1):
                if result is not None:
                    results.append(result)
                if trial_num % 50 == 0 or trial_num == n_trials:
                    elapsed = time.time() - start_time
                    print(f"  ...{trial_num}/{n_trials} trials done ({elapsed:.0f}s, {len(results)} valid)")
    else:
        _init_worker(raw_data)
        for trial_num, params in enumerate(param_list, start=1):
            result = _run_worker((params, system_name))
            if result is not None:
                results.append(result)
            if trial_num % 50 == 0 or trial_num == n_trials:
                elapsed = time.time() - start_time
                print(f"  ...{trial_num}/{n_trials} trials done ({elapsed:.0f}s, {len(results)} valid)")

    if not results:
        return None, []
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results[0], results


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(f"ISOLATED OPTIMIZATION -- System 1 and System 2 tested completely separately")
    print("=" * 78)
    print(f"{N_TRIALS_PER_SYSTEM} trials for System 1 alone, {N_TRIALS_PER_SYSTEM} for System 2 alone")
    print(f"Scoring: {SCORE_AGGREGATION}-of-per-asset {RANK_METRIC}")
    print("NOTE: during each isolated test, that system has the ENTIRE exposure budget to itself --")
    print("      once combined at the end, both systems share MAX_TOTAL_EXPOSURE_PCT. The final")
    print("      combined backtest below quantifies how much (if any) difference that makes.")

    n_workers = N_WORKERS or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, N_TRIALS_PER_SYSTEM))
    print(f"\nWorkers: {n_workers}")

    print("\nDownloading data...")
    raw_data = td.download_data(td.TICKERS, td.START_DATE, td.END_DATE)
    original_systems = td.SYSTEMS
    original_exposure = td.MAX_TOTAL_EXPOSURE_PCT
    original_global_toggles = {}
    for toggle_name, sub_params in opt.GLOBAL_CONDITIONAL_TOGGLES.items():
        original_global_toggles[toggle_name] = getattr(td, toggle_name)
        for sub_name in sub_params:
            original_global_toggles[sub_name] = getattr(td, sub_name)

    rng = random.Random(RANDOM_SEED)

    print("\n" + "=" * 78)
    print("OPTIMIZING SYSTEM 1 -- ALONE (System 2 does not exist for these trials)")
    print("=" * 78)
    best_s1, all_s1 = optimize_system_isolated(raw_data, "S1", "S1", N_TRIALS_PER_SYSTEM, n_workers, rng)
    if best_s1 is None:
        print("No valid System 1 trial found -- aborting.")
        return
    print(f"\nBest System 1 (alone): Sharpe={best_s1['sharpe']:.2f}, Calmar={best_s1['calmar']:.2f}, "
          f"CAGR={best_s1['cagr']*100:.2f}%, worst-asset DD={best_s1['worst_dd']*100:.2f}%")
    print(f"Trades: {best_s1['total_trades']}, per asset: {best_s1['per_asset_trades']}")

    print("\n" + "=" * 78)
    print("OPTIMIZING SYSTEM 2 -- ALONE (System 1 does not exist for these trials)")
    print("=" * 78)
    best_s2, all_s2 = optimize_system_isolated(raw_data, "S2", "S2", N_TRIALS_PER_SYSTEM, n_workers, rng)
    if best_s2 is None:
        print("No valid System 2 trial found -- aborting.")
        return
    print(f"\nBest System 2 (alone): Sharpe={best_s2['sharpe']:.2f}, Calmar={best_s2['calmar']:.2f}, "
          f"CAGR={best_s2['cagr']*100:.2f}%, worst-asset DD={best_s2['worst_dd']*100:.2f}%")
    print(f"Trades: {best_s2['total_trades']}, per asset: {best_s2['per_asset_trades']}")

    # ------------------------------------------------------------------------
    # FINAL COMBINED BACKTEST -- put the two isolated winners together and see
    # how they actually perform as a pair, sharing the real exposure cap.
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL COMBINED BACKTEST -- both isolated winners together")
    print("=" * 78)
    td.MAX_TOTAL_EXPOSURE_PCT = original_exposure
    # global/account-level toggles (e.g. drawdown-based sizing) were sampled independently
    # within EACH system's own isolated search, so S1's winner and S2's winner may not agree --
    # this combined run uses S1's winning trial's global settings, since there's no principled
    # way to pick between them otherwise. Worth re-testing manually if this matters to you.
    for toggle_name, sub_params in opt.GLOBAL_CONDITIONAL_TOGGLES.items():
        if toggle_name in best_s1["params"]:
            setattr(td, toggle_name, best_s1["params"][toggle_name])
            for sub_name in sub_params:
                if sub_name in best_s1["params"]:
                    setattr(td, sub_name, best_s1["params"][sub_name])
    print("(using System 1's winning trial's global/account-level settings, e.g. drawdown scaling, "
          "for this combined run -- S1 and S2 were searched independently so their global settings "
          "could differ; there's no principled way to pick between them otherwise)")
    td.SYSTEMS = [best_s1["system_dict"], best_s2["system_dict"]]
    combined_features = {name: td.build_features(df) for name, df in raw_data.items()}
    combined_results = {name: td.simulate_asset(df) for name, df in combined_features.items()}

    combined_per_asset = {}
    for name, r in combined_results.items():
        stats = td.performance_stats(r["equity_curve"], td.STARTING_CAPITAL_PER_ASSET)
        combined_per_asset[name] = stats

    def combined_aggregate(stats_dict, key):
        values = np.array([s[key] for s in stats_dict.values() if np.isfinite(s.get(key, np.nan))])
        return np.median(values) if len(values) else float("nan")

    combined_sharpe = combined_aggregate(combined_per_asset, "Sharpe")
    print(f"Combined (both systems together): median-of-per-asset Sharpe = {combined_sharpe:.2f}")

    # per-system Sharpe WITHIN the combined run, to compare against each system's ISOLATED score
    for sys_name, isolated_result in [("S1", best_s1), ("S2", best_s2)]:
        sys_pnl_total = 0.0
        sys_trade_count = 0
        for asset_name, r in combined_results.items():
            sys_trades = r["trades_by_system"].get(sys_name, [])
            sys_pnl_total += sum(t["pnl"] for t in sys_trades)
            sys_trade_count += len(sys_trades)
        print(f"\n  System {sys_name}: isolated Sharpe was {isolated_result['sharpe']:.2f} "
              f"(alone, full exposure budget)")
        print(f"    Within the combined run: total PnL across assets = ${sys_pnl_total:,.0f}, "
              f"trades: {sys_trade_count}")

    print("\nPer-asset combined performance:")
    combined_rows = []
    for name, stats in combined_per_asset.items():
        combined_rows.append({
            "Asset": name, "Sharpe": round(stats["Sharpe"], 2), "Calmar": round(stats["Calmar"], 2),
            "CAGR": f"{stats['CAGR']*100:.2f}%", "MaxDD": f"{stats['Max Drawdown']*100:.2f}%",
            "Final Equity": f"${stats['Final Equity']:,.0f}",
        })
    print(pd.DataFrame(combined_rows).to_string(index=False))

    print("\nTo use this configuration, set these in turtle_dual_system.py:")
    print(f"  MAX_TOTAL_EXPOSURE_PCT = {original_exposure!r}")
    for toggle_name, sub_params in opt.GLOBAL_CONDITIONAL_TOGGLES.items():
        print(f"  {toggle_name} = {getattr(td, toggle_name)!r}")
        for sub_name in sub_params:
            print(f"  {sub_name} = {getattr(td, sub_name)!r}")
    print(f"  SYSTEMS = [")
    for s in td.SYSTEMS:
        print(f"      {{")
        for k, v in s.items():
            print(f"          {k!r}: {v!r},")
        print(f"      }},")
    print(f"  ]")

    td.SYSTEMS = original_systems
    td.MAX_TOTAL_EXPOSURE_PCT = original_exposure
    for name, value in original_global_toggles.items():
        setattr(td, name, value)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for label, all_results in [("S1", all_s1), ("S2", all_s2)]:
        rows = []
        for rank, r in enumerate(all_results, start=1):
            row = {"rank": rank, "sharpe": r["sharpe"], "calmar": r["calmar"], "cagr": r["cagr"],
                   "worst_dd": r["worst_dd"], "total_trades": r["total_trades"]}
            row.update(r["params"])
            rows.append(row)
        csv_path = os.path.join(OUTPUT_DIR, f"turtle_isolated_{label}_results.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n{label} isolated search results ({len(all_results)} trials) saved to: {csv_path}")


if __name__ == "__main__":
    main()