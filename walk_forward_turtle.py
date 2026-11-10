"""
Walk-Forward Validation for the Dual-System Turtle Breakout Strategy
======================================================================
Genuine walk-forward analysis: for each window, the parameter space is searched using ONLY
in-sample (train) data, then the single winning parameter set is evaluated on the out-of-sample
(test) period immediately after it -- data that search never saw. This repeats across a sequence
of rolling windows spanning all available history, and the out-of-sample slices are stitched
together into one continuous multi-year curve: the walk-forward equivalent of "how would
re-optimizing on a schedule and trading the winner forward have actually performed."

HOW IT RUNS FAST (the results for any given parameter set are unchanged by any of this)
  1. WINDOWS RUN IN PARALLEL. Each window (search + out-of-sample check) is one independent task
     on a process pool (N_WORKERS, default cpu_count - 1). Price data is sent to each worker
     once, not once per window. Window reports are printed in window order as they complete.
  2. OPTUNA SEARCH (SEARCH_METHOD = "optuna", the default). A TPE sampler concentrates trials
     where scores are good instead of sampling blindly, and a median pruner stops a trial early
     when its running score (after each asset) is already below what other trials had at the same
     point. "random" keeps the original blind random search for A/B comparison.
     Optuna runs single-threaded inside a window ON PURPOSE: the engine applies each trial's
     parameters to module-level globals (opt.apply_params), so concurrent threads would corrupt
     each other -- the parallelism is across windows (processes) instead, which scales better.
  3. INDICATOR CACHE (wf_fast.FeatureCache). Every trial in a window runs on the same prices, and
     each indicator parameter comes from a small discrete set, so each (asset, indicator,
     period) is computed once per window and reused. Bit-identical to td.build_features().
  4. OPTIONAL WARM START (SIM_WARMUP_YEARS). Trials normally simulate each asset from the very
     start of its history up to the window's end -- 30 years of bars to score 4. With a warm start
     the simulation begins SIM_WARMUP_YEARS before the window's train start instead. None keeps
     the original full-history behavior. See KNOWN SIMPLIFICATION for what this changes.
  5. Duplicate parameter sets proposed by the optimizer are answered from a cache, not re-run.

WINDOWING (see build_windows): rolling, fixed-length TRAIN_YEARS immediately before each
TEST_YEARS test window by default. The next window's test period starts the day after the
previous one ended, so out-of-sample periods never overlap and stitch into one continuous
record with no duplicated dates (the step between windows is therefore TEST_YEARS). Set
ANCHORED_TRAIN = True for an expanding train window (always starts at the very first
available date) instead of a fixed-length rolling one.

LEAKAGE: a trial's score for a window is computed from one simulate_asset() run per asset, but the
run only ever "knows" about data up to and including the current bar (every feature in
turtle_dual_system.py is built with .shift(1)-style lookahead avoidance) -- so slicing the
RESULTING equity curve down to just the in-sample dates [train_start, train_end] for search-phase
scoring never includes any information from later dates. In-sample search trials are additionally
run on price data sliced to end at train_end (never given test-period rows to begin with), and the
winning params are then re-run ONCE more on data through test_end for out-of-sample scoring -- so
there are two independent reasons test data can never influence which params get chosen for a
window. The indicator cache is built per window on that same train-sliced data, and a warm start
only moves the simulation's START earlier (more past data), never its end.

KNOWN SIMPLIFICATION: every trial in a window's SEARCH phase, and that window's final
winning-params OOS evaluation, simulate each asset from a start point (the asset's own data
start, or SIM_WARMUP_YEARS before train_start) through that call's own end date -- not starting
fresh at train_start/test_start. PATH-DEPENDENT state (cooldown-after-loss, drawdown scaling, any
still-open position carried in) therefore reflects however THAT TRIAL's own params would have
played out over the preceding history, not what an actual live walk-forward process would have
carried in (which would reflect each PRIOR window's own, different, winning params). A warm start
shortens that preceding history, so carried-in state differs somewhat from the full-history
default; it never leaks test-period information. The stitching step rebases each window onto the
previous window's ending equity by RETURNS, not dollar level.

Requires: pip install yfinance pandas numpy optuna --break-system-packages

Example (standalone):
    python walk_forward_turtle.py            # uses the CONFIG block below
or from code:
    import walk_forward_turtle as wf
    wf.N_TRIALS_PER_WINDOW = 100
    wf.SEARCH_METHOD = "optuna"
    result = wf.main()                       # {"window_reports", "summary", ...}
    print(result["summary"]["wfe_cagr"])
"""
import os
import random
import time
import threading
import traceback
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd

import turtle_dual_system as td
import optimize_turtle_dual_system as opt
import wf_fast
import ibkr_fees

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TRAIN_YEARS = 4
TEST_YEARS = 1          # also the step between windows (test periods must not overlap to stitch)
ANCHORED_TRAIN = False  # False = rolling TRAIN_YEARS-long train window; True = expanding from
                        # the very first available date instead

N_TRIALS_PER_WINDOW = 150
RANDOM_SEED = 42
N_WORKERS = None        # processes running windows in parallel; None = cpu_count() - 1

SEARCH_METHOD = "optuna"   # "optuna" (TPE + pruning) or "random" (original blind random search)
USE_PRUNING = True         # optuna only: stop clearly-bad trials after some of the assets
OPTUNA_MULTIVARIATE = False  # optuna only: True = TPE models parameters jointly (slower sampler); False =
                             # independent per parameter, which scored better in the optimizer tests
SIM_WARMUP_YEARS = 3       # simulate from (train_start - this many years); None = from data start.
                           # Needs to cover the longest indicator lookback (up to 400 bars).

# Set by turtle_ui.py's walk-forward tab before calling main() -- kept as plain module
# attributes (not main() parameters) since main() reads config the same way every other script
# in this project does. None/{} preserve unrestricted search for anyone running this standalone.
ONLY_OPTIMIZE = None
FIXED_VALUES = None

RANK_METRIC = "sharpe"        # "sharpe", "calmar", "cagr", "total_return" -- the objective Optuna
                                # maximizes and the metric that selects each window's winner
PICK_MIDDLE_OF_TOP_N = 1      # 1 = take the single best trial. N > 1 = take the top N by
                                # RANK_METRIC and use the one in the MIDDLE of that group (rank
                                # (N-1)//2 + 1, e.g. N=10 -> 5th best), which is less likely to be
                                # a lucky outlier than the #1 trial. Same rule as the Optimizer tab.
USE_IBKR_FEES = False         # True: charge Interactive Brokers per-contract futures fees (ibkr_fees.py)
                                # instead of the flat % commission, and drop the financing-style holding
                                # cost. Several fee figures are estimates -- see ibkr_fees.py.
IBKR_CONTRACT_SIZE = "micro"  # "micro" (where IBKR has a micro contract) or "standard"
COST_OVERRIDES = None         # e.g. {"SPREAD_PCT": 0.01, "SLIPPAGE_PCT": 0.005, "COMMISSION_PCT": 0.0,
                                # "HOLDING_COST_PCT_PER_YEAR": 0.0}: replaces the engine's cost settings for
                                # the run (td attributes of the same name; restored afterwards). Applied
                                # AFTER USE_IBKR_FEES, so it wins if both touch the same setting.
SCORE_AGGREGATION = "median"  # how per-asset in-sample scores combine into one number per trial

# "per_asset" (default): each window's IS/OOS score is SCORE_AGGREGATION-of-each-asset's-own-
#   normalized Sharpe/Calmar/etc, exactly as always -- every asset simulated independently.
# "portfolio": each window's IS/OOS score comes from ONE shared-capital simulation across every
#   asset at once (td.simulate_portfolio()).
SCORE_MODE = "per_asset"

MIN_ASSET_ROWS_BEFORE_CUTOFF = 50   # an asset with fewer rows than this before a window's own
                                      # cutoff date is treated as "not listed yet" and excluded
                                      # from that window entirely


# ----------------------------------------------------------------------------
# WINDOWING
# ----------------------------------------------------------------------------
def build_windows(overall_start: pd.Timestamp, overall_end: pd.Timestamp) -> list:
    """Builds the sequence of {train_start, train_end, test_start, test_end} windows spanning
    [overall_start, overall_end]. Test periods are back-to-back and non-overlapping, so the
    out-of-sample slices stitch into one continuous record with no gaps or double-counted dates.

    test_start is train_end + 1 DAY, not train_end itself: every score slice elsewhere in this
    file is a plain `.loc[score_start:score_end]`, which is INCLUSIVE on both ends. Setting
    test_start == train_end put that one boundary bar in BOTH the in-sample score and the
    out-of-sample score for the same window, and in both windows' OOS scores at the next
    boundary (confirmed: the stitched output had a duplicate-indexed row at every boundary).
    The +1 day doesn't need to land on an actual trading day -- .loc on a DatetimeIndex finds the
    nearest in-range date either way, it just needs to not equal train_end."""
    windows = []
    train_end = overall_start + pd.DateOffset(years=TRAIN_YEARS)
    while train_end < overall_end:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(years=TEST_YEARS), overall_end)
        train_start = overall_start if ANCHORED_TRAIN else train_end - pd.DateOffset(years=TRAIN_YEARS)
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
        if test_end >= overall_end:
            break
        train_end = test_end
    return windows


# ----------------------------------------------------------------------------
# TRIAL EXECUTION (in-sample search AND out-of-sample evaluation both go through this)
# ----------------------------------------------------------------------------
def _prior_value(curve: pd.Series, score_start: pd.Timestamp):
    """The equity on the bar just before score_start, or None. Stitching needs it: a window's own
    curve starts at its first bar, so the return from the bar before into that first bar belongs
    to no window unless the chain is anchored on this value."""
    position = curve.index.searchsorted(score_start, side="left")
    if position <= 0:
        return None
    value = float(curve.iloc[position - 1])
    return value if np.isfinite(value) and value > 0 else None


def _default_features_fn(ath_references: dict):
    ath_references = ath_references or {}
    def build(name, df):
        out = td.build_features(df, full_history_ath=ath_references.get(name))
        out.attrs["asset"] = name   # fee schedule lookup (see td.commission_for)
        return out
    return build


def _run_window_trial_portfolio(raw_data: dict, params: dict, ath_references: dict,
                                   score_start: pd.Timestamp, score_end: pd.Timestamp,
                                   return_curves: bool, capture_trades: bool = False,
                                   features_fn=None):
    """SCORE_MODE == "portfolio": ONE shared-capital simulation across every asset in raw_data
    (td.simulate_portfolio()), scored on that single equity curve's [score_start, score_end]
    slice -- same causal-slicing safety argument as the per-asset path (see LEAKAGE above)."""
    opt.apply_params(params)
    features_fn = features_fn or _default_features_fn(ath_references)
    try:
        features = {name: features_fn(name, df) for name, df in raw_data.items()}
        port_result = td.simulate_portfolio(features)
    except Exception:
        return None

    window_equity = port_result["equity_curve"].loc[score_start:score_end].dropna()
    if len(window_equity) < 2:
        return None
    baseline = window_equity.iloc[0]
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    try:
        stats = td.performance_stats(window_equity, baseline)
    except Exception:
        return None
    rank_stats_key = {"sharpe": "Sharpe", "calmar": "Calmar", "cagr": "CAGR",
                        "total_return": "Total Return"}[RANK_METRIC]
    if not np.isfinite(stats.get(rank_stats_key, np.nan)):
        return None

    per_asset_trades = {
        name: sum(1 for t in trades if score_start <= t["exit_date"] <= score_end)
        for name, trades in port_result["trades_by_asset"].items()
    }
    result = {
        "params": dict(params),
        "sharpe": stats["Sharpe"], "calmar": stats["Calmar"] if np.isfinite(stats["Calmar"]) else 0.0,
        "cagr": stats["CAGR"], "total_return": stats["Total Return"], "max_dd": stats["Max Drawdown"],
        "per_asset_metric": {name: {"sharpe": np.nan, "calmar": np.nan, "cagr": np.nan,
                                       "total_return": np.nan, "max_dd": np.nan}
                                for name in raw_data},
        "per_asset_trades": per_asset_trades,
        "total_trades": sum(per_asset_trades.values()), "n_assets_scored": len(raw_data),
    }
    if return_curves:
        result["equity_curves"] = {"PORTFOLIO": window_equity}
        result["equity_priors"] = {"PORTFOLIO": _prior_value(port_result["equity_curve"], score_start)}
    if capture_trades:
        result["oos_chart_data"] = {
            name: _extract_window_chart_data(
                {"trades_by_system": port_result["trades_by_system"].get(name, {}),
                 "exit_markers_by_system": port_result["exit_markers_by_asset"].get(name, {})},
                features[name].index, score_start, score_end)
            for name in features
        }
    return result


def _extract_window_chart_data(r: dict, feat_index: pd.Index,
                                 score_start: pd.Timestamp, score_end: pd.Timestamp) -> dict:
    """Filters one asset's full simulate_asset() result down to just the trades/exit-events that
    belong to THIS window's own out-of-sample test period [score_start, score_end]. Trades are
    kept by final exit_date. Exit-marker ilocs are resolved to actual dates here (they're only
    valid against THIS window's own `feat_index`); the UI re-resolves them on the full history."""
    trades_by_system = {
        sysname: [t for t in trades
                   if t.get("exit_date") is not None and score_start <= t["exit_date"] <= score_end]
        for sysname, trades in r.get("trades_by_system", {}).items()
    }
    exit_markers_by_system = {}
    for sysname, markers in r.get("exit_markers_by_system", {}).items():
        kept = []
        for iloc, price, reason in markers:
            date = feat_index[iloc]
            if score_start <= date <= score_end:
                kept.append((date, price, reason))
        exit_markers_by_system[sysname] = kept
    return {"trades_by_system": trades_by_system, "exit_markers_by_system": exit_markers_by_system}


def _score_asset(r: dict, score_start: pd.Timestamp, score_end: pd.Timestamp, rank_stats_key: str):
    """One asset's performance stats on its [score_start, score_end] equity slice, or None."""
    window_equity = r["equity_curve"].loc[score_start:score_end].dropna()
    if len(window_equity) < 2:
        return None
    baseline = window_equity.iloc[0]
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    try:
        stats = td.performance_stats(window_equity, baseline)
    except Exception:
        return None
    if not np.isfinite(stats.get(rank_stats_key, np.nan)):
        return None
    return stats, window_equity


def _aggregate(values: np.ndarray) -> float:
    if SCORE_AGGREGATION == "mean":
        return values.mean()
    elif SCORE_AGGREGATION == "median":
        return np.median(values)
    elif SCORE_AGGREGATION == "min":
        return values.min()
    raise ValueError(f"Unknown SCORE_AGGREGATION: {SCORE_AGGREGATION}")


def run_window_trial(raw_data: dict, params: dict, ath_references: dict,
                       score_start: pd.Timestamp, score_end: pd.Timestamp,
                       return_curves: bool = False, capture_trades: bool = False,
                       features_fn=None, progress_cb=None):
    """Simulates every asset in raw_data with `params` applied, then scores ONLY the equity
    curve slice [score_start, score_end] -- see this file's module docstring ("LEAKAGE") for why
    that slice is safe to use for in-sample scoring. Each asset is simulated and scored
    independently (a single asset's error or too-short history doesn't abort the whole trial).

    features_fn(name, df) -> features DataFrame: defaults to td.build_features; the window worker
    passes a wf_fast.FeatureCache's build for cached indicators (same output).
    progress_cb(per_asset_stats_so_far, n_assets_done): called after each scored asset, OUTSIDE
    any try/except, so it may raise (e.g. optuna.TrialPruned) to abandon the trial early.
    capture_trades: also extracts this window's own OOS trades/exit-markers for charting (only
    ever passed for the single winning-params OOS evaluation per window)."""
    if SCORE_MODE == "portfolio":
        return _run_window_trial_portfolio(raw_data, params, ath_references, score_start, score_end,
                                             return_curves, capture_trades, features_fn)

    opt.apply_params(params)
    features_fn = features_fn or _default_features_fn(ath_references)
    rank_stats_key = {"sharpe": "Sharpe", "calmar": "Calmar", "cagr": "CAGR",
                        "total_return": "Total Return"}[RANK_METRIC]

    results, features_by_asset = {}, {}
    per_asset_stats, per_asset_trades, window_curves, window_priors = {}, {}, {}, {}
    for name, df in raw_data.items():
        try:
            features = features_fn(name, df)
            r = td.simulate_asset(features)
        except Exception:
            continue
        scored = _score_asset(r, score_start, score_end, rank_stats_key)
        if scored is None:
            continue
        stats, window_equity = scored
        results[name], features_by_asset[name] = r, features
        per_asset_stats[name] = stats
        per_asset_trades[name] = sum(
            1 for t in r["trades"] if score_start <= t["exit_date"] <= score_end)
        if return_curves:
            window_curves[name] = window_equity
            window_priors[name] = _prior_value(r["equity_curve"], score_start)
        if progress_cb is not None:
            progress_cb(per_asset_stats, len(per_asset_stats))

    valid_assets = list(per_asset_stats.keys())
    if not valid_assets:
        return None

    per_asset_metric = {
        name: {
            "sharpe": per_asset_stats[name]["Sharpe"],
            "calmar": per_asset_stats[name]["Calmar"] if np.isfinite(per_asset_stats[name]["Calmar"]) else 0.0,
            "cagr": per_asset_stats[name]["CAGR"],
            "total_return": per_asset_stats[name]["Total Return"],
            "max_dd": per_asset_stats[name]["Max Drawdown"],
        }
        for name in valid_assets
    }

    def aggregate(metric_key):
        return _aggregate(np.array([per_asset_metric[name][metric_key] for name in valid_assets]))

    score_sharpe = aggregate("sharpe")
    score_calmar = aggregate("calmar")
    score_cagr = aggregate("cagr")
    score_total_return = aggregate("total_return")
    # aggregate("sharpe") can still be NaN here even after the per-asset gate (plain mean/median/
    # min propagate NaN from any INCLUDED asset whose own Sharpe was non-finite), so check
    # whichever score RANK_METRIC actually ranks by.
    score_by_rank_metric = {"sharpe": score_sharpe, "calmar": score_calmar, "cagr": score_cagr,
                              "total_return": score_total_return}[RANK_METRIC]
    if not np.isfinite(score_by_rank_metric):
        return None

    result = {
        "params": dict(params),
        "sharpe": score_sharpe, "calmar": score_calmar,
        "cagr": score_cagr, "total_return": score_total_return,
        "max_dd": min(per_asset_metric[name]["max_dd"] for name in valid_assets),
        "per_asset_metric": per_asset_metric, "per_asset_trades": per_asset_trades,
        "total_trades": sum(per_asset_trades.values()), "n_assets_scored": len(valid_assets),
    }
    if return_curves:
        result["equity_curves"] = window_curves
        result["equity_priors"] = window_priors
    if capture_trades:
        result["oos_chart_data"] = {
            name: _extract_window_chart_data(results[name], features_by_asset[name].index,
                                               score_start, score_end)
            for name in valid_assets
        }
    return result


# ----------------------------------------------------------------------------
# IN-SAMPLE SEARCH (one window): Optuna or random
# ----------------------------------------------------------------------------
def _metric_key() -> str:
    return {"sharpe": "sharpe", "calmar": "calmar", "cagr": "cagr", "total_return": "total_return"}[RANK_METRIC]


def _search_random(raw_is, ath_is, w, features_fn, st, seed, cancel_check):
    """Original blind random search over the (optionally restricted) space."""
    rng = random.Random(seed)
    param_list = opt.sample_unique_param_sets(rng, st["n_trials"], only_optimize=st["only_optimize"],
                                                 fixed_values=st["fixed_values"])
    results, n_failed = [], 0
    for params in param_list:
        if cancel_check():
            return results, n_failed, True
        r = run_window_trial(raw_is, params, ath_is, w["train_start"], w["train_end"], features_fn=features_fn)
        if r is None:
            n_failed += 1
        else:
            results.append(r)
    return results, n_failed, False


def _search_optuna(raw_is, ath_is, w, features_fn, st, seed, cancel_check):
    """Optuna TPE search maximizing the RANK_METRIC score. The budget (n_trials) counts actual
    simulations -- pruned ones included, cache-answered duplicates and structurally invalid
    combos not -- so the cost matches what random search would spend."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    n_trials = st["n_trials"]
    n_assets = len(raw_is)
    n_startup = max(10, n_trials // 4)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=st["optuna_multivariate"],
                                           group=st["optuna_multivariate"], n_startup_trials=n_startup)
    if st["use_pruning"] and n_assets >= 4:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=max(1, n_assets // 2))
    else:
        pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    metric_key = _metric_key()
    rank_stats_key = {"sharpe": "Sharpe", "calmar": "Calmar", "cagr": "CAGR",
                        "total_return": "Total Return"}[RANK_METRIC]
    seen, results = {}, []
    counters = {"sims": 0, "failed": 0}

    def objective(trial):
        params = wf_fast.suggest_params(trial, st["only_optimize"], st["fixed_values"])
        if not opt.is_valid_params(params):
            raise optuna.TrialPruned()
        sig = wf_fast.params_signature(params)
        if sig in seen:                       # already simulated: answer from cache, no re-run
            if seen[sig] is None:
                raise optuna.TrialPruned()
            return seen[sig]

        def progress(per_asset_stats, n_done):
            running = _aggregate(np.array([s[rank_stats_key] for s in per_asset_stats.values()]))
            trial.report(float(running), n_done)
            if trial.should_prune():
                raise optuna.TrialPruned()

        counters["sims"] += 1
        r = run_window_trial(raw_is, params, ath_is, w["train_start"], w["train_end"],
                               features_fn=features_fn, progress_cb=progress)
        if r is None:
            counters["failed"] += 1
            seen[sig] = None
            raise optuna.TrialPruned()
        seen[sig] = r[metric_key]
        results.append(r)
        return r[metric_key]

    def stop_when_done(study_, trial_):
        if counters["sims"] >= n_trials or cancel_check():
            study_.stop()

    study.optimize(objective, n_trials=n_trials * 5, callbacks=[stop_when_done],
                   catch=(), show_progress_bar=False)
    return results, counters["failed"], bool(cancel_check())


# ----------------------------------------------------------------------------
# ONE WINDOW (runs inside a worker process, or in-process when N_WORKERS == 1)
# ----------------------------------------------------------------------------
_STATE = {}   # per-process: full price data, settings -- set by _init_window_worker


def _init_window_worker(full_data: dict, full_ath: dict, cfg: dict):
    """Pool initializer. On Windows, worker processes are spawned and re-import this module fresh
    from disk, so every runtime setting (UI-set globals included) must be re-applied here --
    same reasoning as optimize_turtle_dual_system.py's own _init_worker."""
    global SCORE_MODE, RANK_METRIC, SCORE_AGGREGATION
    SCORE_MODE = cfg["score_mode"]
    RANK_METRIC = cfg["rank_metric"]
    SCORE_AGGREGATION = cfg["score_aggregation"]
    for name, value in (cfg.get("portfolio_config") or {}).items():
        setattr(td, name, value)
    opt.ENABLE_S1, opt.ENABLE_S2 = cfg["enable_s1"], cfg["enable_s2"]
    opt.set_search_space(cfg["search_space"])   # the ranges the user set, not this file's defaults
    _STATE.update(full_data=full_data, full_ath=full_ath, cfg=cfg, cancel_check=lambda: False)


def _slice_assets(full_data: dict, upto: pd.Timestamp, sim_start) -> dict:
    """Each asset's data through `upto` (never later), starting at sim_start if given."""
    out = {}
    for name, df in full_data.items():
        head = df.loc[:upto]
        if len(head) < MIN_ASSET_ROWS_BEFORE_CUTOFF:
            continue
        out[name] = head if sim_start is None else head.loc[sim_start:]
        out[name].attrs["asset"] = name
    return out


def _process_window(args: tuple) -> dict:
    """Search one window in-sample, evaluate the winner out-of-sample, return everything the
    parent needs (report, OOS curves/chart data, and the log lines to print). Never raises --
    an error becomes status 'error' so one bad window can't kill the whole run."""
    wi, n_windows, w = args
    try:
        return _process_window_inner(wi, n_windows, w)
    except Exception:
        return {"wi": wi, "w": w, "status": "error", "report": {**w, "status": "error"},
                "lines": [f"  ERROR in this window, skipped:\n{traceback.format_exc()}"]}


def _process_window_inner(wi: int, n_windows: int, w: dict) -> dict:
    st, cfg = _STATE, _STATE["cfg"]
    full_data, full_ath = st["full_data"], st["full_ath"]
    cancel_check = st["cancel_check"]
    t0 = time.time()
    lines = [f"WINDOW {wi}/{n_windows}  train [{w['train_start'].date()} -> {w['train_end'].date()}]"
             f"  test [{w['test_start'].date()} -> {w['test_end'].date()}]"]
    warm = cfg["warmup_years"]
    sim_start = (w["train_start"] - pd.DateOffset(years=warm)) if warm else None

    raw_is = _slice_assets(full_data, w["train_end"], sim_start)
    if not raw_is:
        lines.append("  no asset has enough history before this window's train_end -- skipping window")
        return {"wi": wi, "w": w, "status": "skipped_no_data", "report": {**w, "status": "skipped_no_data"},
                "lines": lines}
    ath_is = {name: s.loc[:w["train_end"]] for name, s in full_ath.items() if name in raw_is}

    cache = wf_fast.FeatureCache(raw_is, ath_is)
    seed = cfg["seed"] + wi
    search = _search_optuna if cfg["search_method"] == "optuna" else _search_random
    results, n_failed, cancelled = search(raw_is, ath_is, w, cache.build, cfg, seed, cancel_check)
    if cancelled:
        return {"wi": wi, "w": w, "status": "cancelled", "report": None, "lines": lines}
    lines.append(f"  in-sample search ({cfg['search_method']}): {len(results)} valid trials "
                 f"({n_failed} skipped, {time.time() - t0:.0f}s)")

    if not results:
        lines.append("  no valid in-sample trial (no trades or errored for every sampled config) -- skipping window")
        return {"wi": wi, "w": w, "status": "skipped_no_valid_trial",
                "report": {**w, "status": "skipped_no_valid_trial"}, "lines": lines}

    metric_key = _metric_key()
    results.sort(key=lambda r: r[metric_key], reverse=True)
    n_pick = min(max(int(cfg["pick_middle_n"]), 1), len(results))
    pick_idx = (n_pick - 1) // 2
    best_is = results[pick_idx]
    if pick_idx > 0:
        lines.append(f"  top in-sample {RANK_METRIC}: {results[0][metric_key]:.3f} -- picking rank "
                     f"{pick_idx + 1} of the top {n_pick} (middle) instead")
    lines.append(f"  {'picked' if pick_idx > 0 else 'best'} in-sample {RANK_METRIC}: {best_is[metric_key]:.3f} "
                 f"(CAGR={best_is['cagr']*100:.1f}%, {best_is['total_trades']} trades, "
                 f"{best_is['n_assets_scored']} assets scored)")

    raw_oos = _slice_assets(full_data, w["test_end"], sim_start)
    ath_oos = {name: s.loc[:w["test_end"]] for name, s in full_ath.items() if name in raw_oos}
    oos = run_window_trial(raw_oos, best_is["params"], ath_oos, w["test_start"], w["test_end"],
                             return_curves=True, capture_trades=True)

    if oos is None:
        lines.append("  out-of-sample: no trades / no usable equity in the test period for any "
                     "asset with this window's winning params")
        return {"wi": wi, "w": w, "status": "no_oos_activity", "lines": lines,
                "report": {**w, "status": "no_oos_activity", "params": best_is["params"],
                           "is_sharpe": best_is["sharpe"], "is_cagr": best_is["cagr"]}}

    lines.append(f"  out-of-sample {RANK_METRIC}: {oos[metric_key]:.3f} "
                 f"(CAGR={oos['cagr']*100:.1f}%, MaxDD={oos['max_dd']*100:.1f}%, "
                 f"{oos['total_trades']} trades, {oos['n_assets_scored']} assets)")
    lines.append(f"  IS -> OOS {RANK_METRIC} gap: {best_is[metric_key] - oos[metric_key]:+.3f} "
                 f"(a large positive gap here is the classic overfitting signature)")
    chart = None
    if oos.get("oos_chart_data"):
        chart = {"window": wi, "test_start": w["test_start"], "test_end": w["test_end"],
                 "params": best_is["params"], "assets": oos["oos_chart_data"]}
    return {"wi": wi, "w": w, "status": "ok", "lines": lines,
            "oos_curves": oos["equity_curves"], "oos_priors": oos.get("equity_priors") or {}, "chart": chart,
            "report": {
                **w, "status": "ok", "params": best_is["params"],
                "is_sharpe": best_is["sharpe"], "is_calmar": best_is["calmar"], "is_cagr": best_is["cagr"],
                "is_trades": best_is["total_trades"],
                "oos_sharpe": oos["sharpe"], "oos_calmar": oos["calmar"], "oos_cagr": oos["cagr"],
                "oos_max_dd": oos["max_dd"], "oos_trades": oos["total_trades"]}}


# ----------------------------------------------------------------------------
# OOS STITCHING
# ----------------------------------------------------------------------------
def stitch_oos_windows(window_oos_curves: list, starting_capital: float, window_oos_priors: list = None) -> dict:
    """Chains each window's own out-of-sample equity slice onto the end of the previous window's
    by RETURNS (not dollar level) -- rebases each new slice so it starts exactly where the
    running chain left off, then concatenates. Windows are non-overlapping in time (build_windows),
    so there's no double-counted date. An asset missing from an early window (e.g. Bitcoin before
    it has enough history) simply starts contributing whenever it first appears with a usable
    curve, picking up the chain from starting_capital at that point."""
    chain_value = {}
    pieces = {}
    for k, window_curves in enumerate(window_oos_curves):
        priors = (window_oos_priors[k] if window_oos_priors and k < len(window_oos_priors) else None) or {}
        for name, series in window_curves.items():
            series = series.dropna()
            if len(series) == 0 or series.iloc[0] <= 0:
                continue
            base = chain_value.get(name, starting_capital)
            # anchor on the equity of the bar BEFORE the window when known, so the first day's return
            # is kept; without it that day would vanish from the stitched curve at every boundary
            anchor = priors.get(name) or series.iloc[0]
            rebased = series / anchor * base
            pieces.setdefault(name, []).append(rebased)
            chain_value[name] = rebased.iloc[-1]
    return {name: pd.concat(parts).sort_index() for name, parts in pieces.items()}


def _empty_result(window_reports=None) -> dict:
    return {"window_reports": window_reports or [], "window_oos_curves": [], "window_oos_chart_data": [],
            "summary": {}}


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main(cancel_event: threading.Event = None):
    """Runs the whole walk-forward and returns {"window_reports", "window_oos_curves",
    "window_oos_chart_data", "summary"}; summary includes Walk-Forward Efficiency.

    cancel_event: checked while waiting for windows to finish -- when set, the worker pool is
    terminated immediately and whatever windows completed so far are returned (partial results,
    not discarded). None (the default) means uncancellable."""
    print("=" * 78)
    print("DUAL-SYSTEM TURTLE STRATEGY -- WALK-FORWARD VALIDATION")
    print("=" * 78)
    print(f"Train window: {'expanding from the start' if ANCHORED_TRAIN else f'{TRAIN_YEARS}y rolling'}"
          f" | Test window: {TEST_YEARS}y | {N_TRIALS_PER_WINDOW} search trials per window "
          f"({SEARCH_METHOD})")
    print(f"Ranking metric: {RANK_METRIC} " + (
        "(portfolio mode -- ONE shared-capital curve across every asset per window)"
        if SCORE_MODE == "portfolio" else f"({SCORE_AGGREGATION}-of-per-asset)"))

    print("\nDownloading full available history (shared across every window)...")
    full_data = td.download_data(td.TICKERS, td.START_DATE, td.END_DATE)
    if not full_data:
        print("No data downloaded -- aborting.")
        return _empty_result()

    overall_start = min(df.index.min() for df in full_data.values())
    overall_end = max(df.index.max() for df in full_data.values())
    windows = build_windows(overall_start, overall_end)
    if not windows:
        print(f"Not enough history for even one {TRAIN_YEARS}y train + {TEST_YEARS}y test window -- aborting.")
        return _empty_result()
    n_windows = len(windows)
    print(f"Full combined history: {overall_start.date()} -> {overall_end.date()} "
          f"({n_windows} walk-forward window(s))")

    # Full-history running max HIGH per asset -- reference for require_all_time_high, truncated
    # per window (never given data past that window's own cutoff; see LEAKAGE above).
    full_ath = {name: df["High"].cummax() for name, df in full_data.items()}

    original_systems = td.SYSTEMS
    all_scalar_param_names = [p for p in opt.INDEPENDENT_PARAM_SPACE
                                if p not in ("S1_BREAKOUT_LOOKBACK", "S1_EXIT_LOOKBACK",
                                             "S2_BREAKOUT_LOOKBACK", "S2_EXIT_LOOKBACK")]
    all_scalar_param_names += list(opt.GLOBAL_CONDITIONAL_TOGGLES)
    all_scalar_param_names += [sub for subs in opt.GLOBAL_CONDITIONAL_TOGGLES.values() for sub in subs]
    original_config = {name: getattr(td, name) for name in all_scalar_param_names}

    fee_snapshot = None
    if USE_IBKR_FEES:
        fee_snapshot, fee_lines = ibkr_fees.apply(td, IBKR_CONTRACT_SIZE, assets=list(full_data))
        print()
        print("\n".join(fee_lines))

    cost_snapshot = None
    if COST_OVERRIDES:
        cost_snapshot = {name: getattr(td, name) for name in COST_OVERRIDES}
        for name, value in COST_OVERRIDES.items():
            setattr(td, name, value)
        print()
        print("Custom trading costs: " + ", ".join(f"{k}={v}" for k, v in COST_OVERRIDES.items()))

    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, min(n_workers, n_windows))
    print(f"Running {n_windows} window(s) on {n_workers} worker process(es)"
          f"{' (parallel across windows)' if n_workers > 1 else ' (sequential)'}"
          + (f"; simulating from {SIM_WARMUP_YEARS}y before each window's train start" if SIM_WARMUP_YEARS
             else "; simulating each asset from its first bar"))

    cfg = {
        "score_mode": SCORE_MODE, "rank_metric": RANK_METRIC, "score_aggregation": SCORE_AGGREGATION,
        "portfolio_config": opt.current_portfolio_config(), "enable_s1": opt.ENABLE_S1,
        "enable_s2": opt.ENABLE_S2, "n_trials": N_TRIALS_PER_WINDOW, "seed": RANDOM_SEED,
        "only_optimize": ONLY_OPTIMIZE, "fixed_values": FIXED_VALUES, "search_method": SEARCH_METHOD,
        "use_pruning": USE_PRUNING, "warmup_years": SIM_WARMUP_YEARS, "pick_middle_n": PICK_MIDDLE_OF_TOP_N,
        "optuna_multivariate": OPTUNA_MULTIVARIATE, "search_space": opt.get_search_space(),
    }
    if SEARCH_METHOD not in ("optuna", "random"):
        raise ValueError(f"Unknown SEARCH_METHOD: {SEARCH_METHOD!r}")

    start_time = time.time()
    tasks = [(wi, n_windows, w) for wi, w in enumerate(windows, start=1)]
    done = {}          # wi -> window result
    next_to_print = 1
    cancelled = False

    def flush_in_order():
        """Prints finished windows strictly in window order, so the log (and the UI's "WINDOW x/y"
        progress bar) reads the same as a sequential run no matter which window finishes first."""
        nonlocal next_to_print
        while next_to_print in done:
            res = done[next_to_print]
            print("\n" + "-" * 78)
            for line in res["lines"]:
                print(line)
            next_to_print += 1
        if next_to_print <= n_windows:
            print(f"  ({len(done)}/{n_windows} windows finished, {time.time() - start_time:.0f}s elapsed)")

    if n_workers > 1:
        with mp.Pool(processes=n_workers, initializer=_init_window_worker,
                      initargs=(full_data, full_ath, cfg)) as pool:
            iterator = pool.imap_unordered(_process_window, tasks, chunksize=1)
            while len(done) < n_windows:
                if cancel_event is not None and cancel_event.is_set():
                    pool.terminate()
                    cancelled = True
                    break
                try:
                    res = iterator.next(timeout=0.5)
                except mp.TimeoutError:
                    continue
                except StopIteration:
                    break
                done[res["wi"]] = res
                flush_in_order()
    else:
        _init_window_worker(full_data, full_ath, cfg)
        _STATE["cancel_check"] = (lambda: cancel_event is not None and cancel_event.is_set())
        for task in tasks:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            res = _process_window(task)
            if res["status"] == "cancelled":
                cancelled = True
                break
            done[res["wi"]] = res
            flush_in_order()

    if cancelled:
        print(f"\nCancelled by user -- stopping with {len(done)}/{n_windows} window(s) completed.")

    td.SYSTEMS = original_systems
    for name, value in original_config.items():
        setattr(td, name, value)
    if cost_snapshot is not None:
        for name, value in cost_snapshot.items():
            setattr(td, name, value)
    if fee_snapshot is not None:
        ibkr_fees.restore(td, fee_snapshot)

    # Windows that finished out of order (parallel) but behind an unfinished earlier one are
    # still valid results -- keep them, in window order.
    finished = [done[wi] for wi in sorted(done) if done[wi]["status"] != "cancelled"]
    window_reports = [r["report"] for r in finished if r.get("report")]
    window_oos_curves = [r["oos_curves"] for r in finished if r["status"] == "ok"]
    window_oos_priors = [r.get("oos_priors") or {} for r in finished if r["status"] == "ok"]
    window_oos_chart_data = [r["chart"] for r in finished if r["status"] == "ok" and r.get("chart")]

    ok_windows = [r for r in window_reports if r["status"] == "ok"]
    total_time = time.time() - start_time
    print("\n" + "=" * 78)
    print(f"WALK-FORWARD SUMMARY -- {len(ok_windows)}/{n_windows} windows produced a scoreable "
          f"out-of-sample result ({total_time:.0f}s total)")
    print("=" * 78)

    result = {"window_reports": window_reports, "window_oos_curves": window_oos_curves,
              "window_oos_chart_data": window_oos_chart_data, "summary": {}}
    if not ok_windows:
        print("No window produced an out-of-sample result -- nothing to summarize.")
        return result

    is_scores = np.array([r["is_sharpe"] for r in ok_windows])
    oos_scores = np.array([r["oos_sharpe"] for r in ok_windows])
    is_cagr = np.array([r["is_cagr"] for r in ok_windows])
    oos_cagr = np.array([r["oos_cagr"] for r in ok_windows])
    # Walk-Forward Efficiency: how much of the in-sample performance survives out-of-sample.
    # Rule of thumb: >= ~0.5 is reasonable, near 0 or negative means the optimization found noise.
    # The out-of-sample windows are short (TEST_YEARS), each window's CAGR is just that period's
    # return, and a plain average of such returns is the ARITHMETIC mean -- always above what the
    # windows actually compound to (by about half the yearly variance: several points for a
    # volatile strategy), while the in-sample figures are multi-year compounded rates. Comparing the
    # two directly flattered the ratio, so the efficiency uses the GEOMETRIC mean of the windows.
    oos_cagr_geo = float(np.expm1(np.mean(np.log1p(np.maximum(oos_cagr, -0.999999)))))
    wfe_cagr = oos_cagr_geo / is_cagr.mean() if is_cagr.mean() > 0 else float("nan")
    wfe_cagr_arith = oos_cagr.mean() / is_cagr.mean() if is_cagr.mean() > 0 else float("nan")
    wfe_sharpe = oos_scores.mean() / is_scores.mean() if is_scores.mean() > 0 else float("nan")
    print(f"Mean in-sample Sharpe:     {is_scores.mean():.3f}")
    print(f"Mean out-of-sample Sharpe: {oos_scores.mean():.3f}")
    print(f"Mean IS -> OOS Sharpe gap: {(is_scores - oos_scores).mean():+.3f} "
          f"(bigger = more overfitting from the in-sample search)")
    print(f"Mean out-of-sample CAGR:   {oos_cagr.mean()*100:.1f}% arithmetic average of the windows, "
          f"{oos_cagr_geo*100:.1f}% compounded (geometric) -- the compounded one is what the windows "
          f"really earned")
    print(f"Walk-Forward Efficiency:   {wfe_cagr:.2f} on CAGR (compounded OOS CAGR / mean IS CAGR; "
          f"{wfe_cagr_arith:.2f} with the arithmetic average), {wfe_sharpe:.2f} on Sharpe")
    print(f"Windows with positive OOS Sharpe: {(oos_scores > 0).sum()}/{len(ok_windows)}")
    result["summary"] = {
        "windows_ok": len(ok_windows), "windows_total": n_windows,
        "mean_is_sharpe": float(is_scores.mean()), "mean_oos_sharpe": float(oos_scores.mean()),
        "mean_is_cagr": float(is_cagr.mean()), "mean_oos_cagr": float(oos_cagr.mean()),
        "geo_mean_oos_cagr": oos_cagr_geo, "wfe_cagr_arithmetic": float(wfe_cagr_arith),
        "wfe_cagr": float(wfe_cagr), "wfe_sharpe": float(wfe_sharpe),
        "positive_oos_windows": int((oos_scores > 0).sum()), "elapsed_seconds": total_time,
    }

    print("\nPer-window results:")
    table_rows = [{
        "window": i + 1, "test_start": r["test_start"].date(), "test_end": r["test_end"].date(),
        "IS_sharpe": round(r["is_sharpe"], 2), "OOS_sharpe": round(r["oos_sharpe"], 2),
        "OOS_CAGR": f"{r['oos_cagr']*100:.1f}%", "OOS_MaxDD": f"{r['oos_max_dd']*100:.1f}%",
        "OOS_trades": r["oos_trades"],
    } for i, r in enumerate(ok_windows)]
    print(pd.DataFrame(table_rows).to_string(index=False))

    stitched = stitch_oos_windows(window_oos_curves, td.STARTING_CAPITAL_PER_ASSET, window_oos_priors)
    if stitched:
        print("\nStitched out-of-sample equity curve (all windows chained together by returns) --")
        print("this is the closest thing in this file to 'what would actually re-optimizing on a")
        print(f"schedule and trading the winner forward have returned', per asset:")
        stitched_rows = []
        stitched_stats = {}
        for name, curve in stitched.items():
            try:
                stats = td.performance_stats(curve, td.STARTING_CAPITAL_PER_ASSET)
            except Exception:
                continue
            stitched_stats[name] = stats
            stitched_rows.append({
                "Asset": name, "Sharpe": round(stats["Sharpe"], 2), "CAGR": f"{stats['CAGR']*100:.1f}%",
                "MaxDD": f"{stats['Max Drawdown']*100:.1f}%", "Total Return": f"{stats['Total Return']*100:.1f}%",
            })
        print(pd.DataFrame(stitched_rows).to_string(index=False))
        finite_sharpes = [s["Sharpe"] for s in stitched_stats.values() if np.isfinite(s["Sharpe"])]
        if finite_sharpes:
            stitched_score = (np.median(finite_sharpes) if SCORE_AGGREGATION == 'median'
                              else np.mean(finite_sharpes))
            result["summary"]["stitched_oos_sharpe"] = float(stitched_score)
            print(f"\n{SCORE_AGGREGATION.capitalize()}-of-per-asset stitched OOS Sharpe: {stitched_score:.3f}")
        result["stitched_oos_curves"] = stitched

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    windows_csv_rows = []
    for i, r in enumerate(window_reports, start=1):
        row = {"window": i, "status": r["status"], "train_start": r["train_start"].date(),
               "train_end": r["train_end"].date(), "test_start": r["test_start"].date(),
               "test_end": r["test_end"].date()}
        for key in ("is_sharpe", "is_calmar", "is_cagr", "is_trades",
                     "oos_sharpe", "oos_calmar", "oos_cagr", "oos_max_dd", "oos_trades"):
            row[key] = r.get(key)
        if "params" in r:
            row.update(r["params"])
        windows_csv_rows.append(row)
    windows_csv_path = os.path.join(OUTPUT_DIR, "walk_forward_windows.csv")
    pd.DataFrame(windows_csv_rows).to_csv(windows_csv_path, index=False)
    print(f"\nPer-window results saved to: {windows_csv_path}")

    if stitched:
        equity_csv_path = os.path.join(OUTPUT_DIR, "walk_forward_oos_equity.csv")
        pd.DataFrame(stitched).to_csv(equity_csv_path)
        print(f"Stitched out-of-sample equity curves saved to: {equity_csv_path}")

    return result


if __name__ == "__main__":
    main()
