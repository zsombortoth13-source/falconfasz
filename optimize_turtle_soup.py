"""
Parameter optimizer for turtle_soup.py -- same architecture as optimize_turtle_dual_system.py
(worker pool, per-asset-normalized scoring so no single asset can dominate the ranking,
structural-validity pre-filtering before any compute is spent).
"""

import multiprocessing as mp
import os
import random
import time

import numpy as np
import pandas as pd

import turtle_soup as ts

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
N_TRIALS = 300
RANDOM_SEED = 42
N_WORKERS = None   # None = use all available CPU cores

RANK_METRIC = "sharpe"   # "sharpe", "calmar", "cagr", "total_return"
SCORE_AGGREGATION = "median"   # "mean", "median", "min" -- see optimize_turtle_dual_system.py
                                  # for the full rationale; same logic applies here

OUTPUT_DIR = ts.OUTPUT_DIR

# ----------------------------------------------------------------------------
# SEARCH SPACE
# ----------------------------------------------------------------------------
INDEPENDENT_PARAM_SPACE = {
    "channel_lookback": [10, 15, 20, 30, 40, 55],
    # Was a single fixed value of 5000% (50x notional leverage, for EVERY trial, not even
    # actually searched) -- see optimize_turtle_dual_system.py's own INDEPENDENT_PARAM_SPACE for
    # the full rationale. Now a real, grounded range: 100% (no leverage) up to 500% (5x).
    "MAX_TOTAL_EXPOSURE_PCT": [100, 150, 200, 300, 500],
}

ALWAYS_SAMPLED = {
    "quiet_period_days": [2, 3, 4, 5, 7, 10],
    "reversal_window_days": [1, 2, 3, 5],
    "n_atr_period": [10, 14, 20, 30],
    "risk_per_trade_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
    "stop_buffer_n_mult": [0.25, 0.5, 0.75, 1.0],
    "target_r_mult": [1.0, 1.5, 2.0, 3.0, 4.0],
    "time_stop_days": [3, 5, 7, 10, 15],
}

CONDITIONAL_TOGGLES = {
    "use_plus_one": {},   # no sub-parameters, just on/off
    "use_choppiness_filter": {"choppiness_period": [10, 14, 20, 30],
                                "choppiness_min": [30, 38.2, 45, 55, 61.8]},
}

_approx_independent = 1
for _vals in INDEPENDENT_PARAM_SPACE.values():
    _approx_independent *= len(_vals)
_approx_always = 1
for _vals in ALWAYS_SAMPLED.values():
    _approx_always *= len(_vals)
_approx_toggle = 1
for _toggle, _subs in CONDITIONAL_TOGGLES.items():
    _sub_combos = 1
    for _vals in _subs.values():
        _sub_combos *= len(_vals)
    _approx_toggle *= (1 + _sub_combos)
APPROX_SEARCH_SPACE_SIZE = _approx_independent * _approx_always * _approx_toggle


def sample_params_raw(rng: random.Random, only_optimize: set = None, fixed_values: dict = None) -> dict:
    """only_optimize (optional): if given, ONLY these parameter names are actually randomized --
    every other name is pinned to fixed_values[name] for every trial instead. None (the default)
    is the old, unrestricted behavior. See optimize_turtle_dual_system.py's own version of this
    for the full rationale -- same pattern, smaller search space here."""
    fixed_values = fixed_values or {}

    def pick(name, values):
        if only_optimize is not None and name not in only_optimize:
            return fixed_values[name]
        return rng.choice(values)

    params = {name: pick(name, values) for name, values in INDEPENDENT_PARAM_SPACE.items()}
    for name, values in ALWAYS_SAMPLED.items():
        params[name] = pick(name, values)
    for toggle_name, sub_params in CONDITIONAL_TOGGLES.items():
        if only_optimize is not None and toggle_name not in only_optimize:
            is_on = fixed_values[toggle_name]
        else:
            is_on = rng.choice([True, False])
        params[toggle_name] = is_on
        if is_on:
            for sub_name, sub_values in sub_params.items():
                params[sub_name] = pick(sub_name, sub_values)
    return params


def is_valid_params(params: dict) -> bool:
    """Rejects structurally nonsensical combos before any compute is spent."""
    # the reversal window shouldn't be silly relative to the quiet period -- not a hard
    # requirement, just a sanity bound to avoid wasting trials on degenerate configs
    if params["reversal_window_days"] > params["channel_lookback"]:
        return False
    return True


def sample_unique_param_sets(rng: random.Random, n: int, max_attempts_mult: int = 30,
                               only_optimize: set = None, fixed_values: dict = None) -> list:
    seen = set()
    unique_params = []
    attempts = 0
    max_attempts = n * max_attempts_mult
    while len(unique_params) < n and attempts < max_attempts:
        attempts += 1
        params = sample_params_raw(rng, only_optimize, fixed_values)
        if not is_valid_params(params):
            continue
        signature = frozenset(params.items())
        if signature in seen:
            continue
        seen.add(signature)
        unique_params.append(params)
    if len(unique_params) < n:
        print(f"  (note: only found {len(unique_params)} valid unique combinations after "
              f"{attempts} attempts -- filling the rest with repeats)")
        while len(unique_params) < n:
            unique_params.append(sample_params_raw(rng, only_optimize, fixed_values))
    return unique_params


def params_to_systems(params: dict) -> list:
    p = dict(params)
    system_param_keys = list(ts.DEFAULT_SOUP_PARAMS.keys())
    soup_params = dict(ts.DEFAULT_SOUP_PARAMS)
    for key in system_param_keys:
        if key in p:
            soup_params[key] = p[key]
    return [{"name": "Soup", **soup_params}]


def apply_params(params: dict):
    """Monkey-patch turtle_soup's globals with this trial's parameter set."""
    p = dict(params)
    ts.SYSTEMS = params_to_systems(params)
    system_param_keys = list(ts.DEFAULT_SOUP_PARAMS.keys())
    for key in system_param_keys:
        p.pop(key, None)
    for name, value in p.items():
        setattr(ts, name, value)


def run_trial(raw_data: dict, params: dict) -> dict | None:
    apply_params(params)
    try:
        features = {name: ts.build_features(df) for name, df in raw_data.items()}
        results = {name: ts.simulate_asset(df) for name, df in features.items()}
    except Exception:
        return None

    total_trades = sum(len(r["trades"]) for r in results.values())
    if total_trades == 0:
        return None

    per_asset_metric = {}
    per_asset_stats = {}
    valid_assets = []
    for name, r in results.items():
        stats = ts.performance_stats(r["equity_curve"], ts.STARTING_CAPITAL_PER_ASSET)
        per_asset_stats[name] = stats
        if not np.isfinite(stats["Sharpe"]):
            continue
        valid_assets.append(name)
        calmar = stats["CAGR"] / abs(stats["Max Drawdown"]) if stats["Max Drawdown"] < 0 else np.nan
        total_return = stats["Final Equity"] / ts.STARTING_CAPITAL_PER_ASSET - 1
        per_asset_metric[name] = {"sharpe": stats["Sharpe"], "calmar": calmar,
                                    "cagr": stats["CAGR"], "total_return": total_return,
                                    "max_dd": stats["Max Drawdown"]}

    if not valid_assets:
        return None

    def aggregate(key):
        values = np.array([per_asset_metric[name][key] for name in valid_assets
                            if np.isfinite(per_asset_metric[name][key])])
        if len(values) == 0:
            return np.nan
        if SCORE_AGGREGATION == "mean":
            return values.mean()
        elif SCORE_AGGREGATION == "min":
            return values.min()
        return np.median(values)

    score_sharpe = aggregate("sharpe")
    if not np.isfinite(score_sharpe):
        return None

    combined_equity = sum(r["equity_curve"].reindex(
        sorted(set().union(*[rr["equity_curve"].index for rr in results.values()]))
    ).ffill().bfill() for r in results.values())
    combined_final_equity = combined_equity.iloc[-1]

    per_asset_trades = {name: len(r["trades"]) for name, r in results.items()}

    return {
        "params": dict(params),
        "sharpe": score_sharpe, "calmar": aggregate("calmar"), "cagr": aggregate("cagr"),
        "total_return": aggregate("total_return"),
        "per_asset_metric": per_asset_metric,
        "max_dd": min(per_asset_stats[name]["Max Drawdown"] for name in valid_assets),
        "final_equity": combined_final_equity, "total_trades": total_trades,
        "per_asset_trades": per_asset_trades,
    }


_worker_raw_data = None


def _init_worker(raw_data: dict, _unused_ath_references: dict = None):
    # accepts the same (raw_data, ath_references) calling convention as
    # optimize_turtle_dual_system.py's _init_worker, purely so callers (e.g. turtle_ui.py) can
    # invoke either optimizer's worker the same way without conditional logic -- Turtle Soup has
    # no all-time-high filter, so the second argument is simply ignored here.
    global _worker_raw_data
    _worker_raw_data = raw_data


def _run_trial_worker(params: dict):
    return run_trial(_worker_raw_data, params)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(f"TURTLE SOUP -- PARAMETER OPTIMIZATION ({N_TRIALS} trials)")
    print("=" * 78)
    print(f"Ranking metric: {RANK_METRIC}")
    print(f"Approximate search space size: ~{APPROX_SEARCH_SPACE_SIZE:,} combinations")
    print("Independent:")
    for name, values in INDEPENDENT_PARAM_SPACE.items():
        print(f"  {name}: {values}")
    print("Always sampled:")
    for name, values in ALWAYS_SAMPLED.items():
        print(f"  {name}: {values}")
    print("Conditional toggles:")
    for toggle, subs in CONDITIONAL_TOGGLES.items():
        sub_desc = ", ".join(f"{k}={v}" for k, v in subs.items()) if subs else "(no sub-parameters)"
        print(f"  {toggle}: {sub_desc}")

    print("\nDownloading data...")
    raw_data = ts.download_data(ts.TICKERS, ts.START_DATE, ts.END_DATE)

    original_systems = ts.SYSTEMS
    original_exposure = ts.MAX_TOTAL_EXPOSURE_PCT

    n_workers = N_WORKERS if N_WORKERS is not None else (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, N_TRIALS))

    rng = random.Random(RANDOM_SEED)
    param_list = sample_unique_param_sets(rng, N_TRIALS)

    results = []
    n_failed = 0
    start_time = time.time()

    if n_workers > 1:
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(raw_data,)) as pool:
            for trial_num, result in enumerate(
                    pool.imap_unordered(_run_trial_worker, param_list, chunksize=1), start=1):
                if result is None:
                    n_failed += 1
                else:
                    results.append(result)
                if trial_num % 20 == 0 or trial_num == N_TRIALS:
                    elapsed = time.time() - start_time
                    rate = trial_num / elapsed if elapsed > 0 else 0
                    print(f"  ...{trial_num}/{N_TRIALS} trials done ({elapsed:.0f}s elapsed, "
                          f"{rate:.1f} trials/s, {len(results)} valid, {n_failed} skipped)")
    else:
        _init_worker(raw_data)
        for trial_num, params in enumerate(param_list, start=1):
            result = _run_trial_worker(params)
            if result is None:
                n_failed += 1
            else:
                results.append(result)
            if trial_num % 20 == 0 or trial_num == N_TRIALS:
                elapsed = time.time() - start_time
                rate = trial_num / elapsed if elapsed > 0 else 0
                print(f"  ...{trial_num}/{N_TRIALS} trials done ({elapsed:.0f}s elapsed, "
                      f"{rate:.1f} trials/s, {len(results)} valid, {n_failed} skipped)")

    ts.SYSTEMS = original_systems
    ts.MAX_TOTAL_EXPOSURE_PCT = original_exposure

    total_time = time.time() - start_time
    print(f"\nCompleted {N_TRIALS} trials in {total_time:.0f}s "
          f"({N_TRIALS/total_time:.1f} trials/s with {n_workers} worker(s)) "
          f"-- {len(results)} valid, {n_failed} skipped (no trades or errored)")

    if not results:
        print("No valid trials completed -- nothing to rank.")
        return

    key_fn = {"sharpe": lambda r: r["sharpe"], "calmar": lambda r: r["calmar"],
              "cagr": lambda r: r["cagr"], "total_return": lambda r: r["total_return"]}[RANK_METRIC]
    results.sort(key=key_fn, reverse=True)

    top_n = 15
    print("\n" + "=" * 78)
    print(f"TOP {min(top_n, len(results))} TRIALS (ranked by {SCORE_AGGREGATION}-of-per-asset {RANK_METRIC})")
    print("=" * 78)
    all_asset_names = sorted(set(name for r in results for name in r["per_asset_metric"]))
    rows = []
    for rank, r in enumerate(results[:top_n], start=1):
        row = {"rank": rank, f"{SCORE_AGGREGATION}_sharpe": round(r["sharpe"], 2),
               "CAGR": f"{r['cagr']*100:.2f}%", "worst_asset_MaxDD": f"{r['max_dd']*100:.2f}%",
               "final_equity": f"${r['final_equity']:,.0f}", "trades": r["total_trades"]}
        for name in all_asset_names:
            row[f"sharpe_{name}"] = round(r["per_asset_metric"][name]["sharpe"], 2) \
                if name in r["per_asset_metric"] else np.nan
        row.update(r["params"])
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    best = results[0]
    print("\n" + "=" * 78)
    print("BEST TRIAL -- full parameter set")
    print("=" * 78)
    for k, v in best["params"].items():
        print(f"  {k} = {v}")
    print(f"\n{SCORE_AGGREGATION.capitalize()}-of-per-asset score: Sharpe={best['sharpe']:.2f}, "
          f"Calmar={best['calmar']:.2f}, CAGR={best['cagr']*100:.2f}%")
    print(f"Worst single-asset drawdown: {best['max_dd']*100:.2f}%")
    print(f"Combined portfolio final equity (reporting only, NOT used for ranking): ${best['final_equity']:,.0f}")

    recommended_systems = params_to_systems(best["params"])
    print("\nTo use this configuration, set these in turtle_soup.py:")
    print(f"  MAX_TOTAL_EXPOSURE_PCT = {best['params'].get('MAX_TOTAL_EXPOSURE_PCT', original_exposure)!r}")
    print(f"  SYSTEMS = [")
    for s in recommended_systems:
        print(f"      {{")
        for k, v in s.items():
            print(f"          {k!r}: {v!r},")
        print(f"      }},")
    print(f"  ]")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    for rank, r in enumerate(results, start=1):
        row = {"rank": rank, "sharpe": r["sharpe"], "calmar": r["calmar"], "cagr": r["cagr"],
               "max_dd": r["max_dd"], "final_equity": r["final_equity"], "total_trades": r["total_trades"]}
        for name in all_asset_names:
            row[f"sharpe_{name}"] = r["per_asset_metric"][name]["sharpe"] if name in r["per_asset_metric"] else np.nan
        row.update(r["params"])
        all_rows.append(row)
    csv_path = os.path.join(OUTPUT_DIR, "turtle_soup_optimization_results.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nFull ranked results ({len(results)} trials) saved to: {csv_path}")


if __name__ == "__main__":
    main()