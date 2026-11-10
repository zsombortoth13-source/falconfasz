"""
Parameter Optimization for the Dual-System Turtle Breakout Strategy
=======================================================================
Same random-search approach as optimize_breakout_pyramid.py, applied to
turtle_dual_system.py. Covers EVERY changeable parameter in that strategy:

  - Both systems' breakout and exit lookbacks (System 1's shorter-term pair
    and System 2's longer-term pair, sampled independently but constrained
    to keep the intended structure: each system's exit lookback shorter
    than its own breakout lookback, and System 1 shorter-term than System 2
    overall).
  - Entry buffer, N (ATR) period, risk per unit, total exposure cap,
    pyramid spacing, max units per trend, stop multiple.
  - Every optional entry filter (RSI, EMA, ADX, volatility squeeze, weekly
    alignment, re-entry cooldown) and exit mechanic (time stop, scale-out),
    each as an on/off switch whose sub-parameters are only sampled when
    that switch is on for a given trial (no wasted "diversity" on numbers
    that can't affect an off filter).

Transaction costs (slippage, spread, commission, holding cost) are NOT
included in the search space -- those represent real-world trading costs,
not strategy choices, and "optimizing" them would just mean cheating by
searching for lower costs rather than a better strategy. They stay fixed
at whatever turtle_dual_system.py has configured.

SAMPLING is deduplicated (exact repeats are re-sampled) and constrained
(invalid combinations, like an exit lookback longer than its own breakout
lookback, are rejected and re-sampled) before any compute is spent on them.

PARALLELIZATION: trials run across multiple CPU processes via
multiprocessing.Pool (see optimize_breakout_pyramid.py for the full
rationale -- same approach here).

Requires: pip install yfinance pandas numpy matplotlib --break-system-packages
"""

import os
import random
import time
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd

import turtle_dual_system as td

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
N_TRIALS = 400
RANDOM_SEED = 42
N_WORKERS = None   # None = use all available CPU cores

# Coarse-to-fine search: given how astronomically large the full search space actually is
# (see APPROX_SEARCH_SPACE_SIZE below -- on the order of 10^23 combinations), spending the
# entire trial budget on blind random sampling explores a genuinely negligible fraction of it.
# Instead: run a broad blind search first (STAGE1_FRACTION of the budget), find the top
# TOP_K_FOR_REFINEMENT trials, narrow the search space to the neighborhood of what those trials
# actually used (see narrow_param_space), then spend the REMAINING budget sampling densely from
# that narrowed space instead of the full one. Set USE_TWO_STAGE_SEARCH = False to fall back to
# a single blind search over the whole budget, matching the old behavior exactly.
SEARCH_METHOD = "optuna"   # "optuna" (TPE, see search_optuna) or "random" (blind random search).
                             # The experimental two-stage search below only applies to "random".
USE_TWO_STAGE_SEARCH = False   # tested rigorously across 6 seeds (see optimize_turtle_dual_system
                                 # commit history/conversation) -- narrowing to the top trials'
                                 # neighborhood did not beat plain blind search on average (0.713
                                 # vs 0.760 mean Sharpe), and was meaningfully worse in 2/6 runs,
                                 # even with the stage-2 blind hedge below. Left available as an
                                 # option -- and the machinery is fully tested and correct -- but
                                 # OFF by default since the evidence doesn't support it as this
                                 # search space's structure (many weakly-correlated, largely
                                 # independent toggles) makes "what the top trials share" an
                                 # unreliable signal; narrowing risks excluding a genuinely good
                                 # region purely by chance, for no measured benefit.
STAGE1_FRACTION = 0.6
TOP_K_FOR_REFINEMENT = 20
NARROW_WINDOW = 1   # how many neighboring candidate values to include around what stage 1 used
# HEDGE: narrowing risks excluding a genuinely good region that stage 1's own (necessarily
# smaller) sample just didn't happen to find -- tested directly and confirmed this can actually
# happen (a 60-trial run where the fully-blind version beat the narrowed version). So stage 2
# itself splits ITS budget: STAGE2_BLIND_FRACTION stays fully blind (same as stage 1 would have
# been), the rest uses the narrowed/refined space. This means stage 2 can never make things
# WORSE than a single blind search would have -- at worst it ties it.
STAGE2_BLIND_FRACTION = 0.4

RANK_METRIC = "sharpe"   # "sharpe", "calmar", "cagr", "total_return"

# HOW to combine each asset's own normalized performance into one score for ranking.
# "mean": average across assets -- a good overall trial can still be pulled up by one
#   standout asset, but nowhere near as badly as dollar-summing would.
# "median": robust to one asset being an outlier in either direction.
# "min": the WORST-performing asset's score -- the strictest option, actively selects
#   for trials that work reasonably well EVERYWHERE, not just on average. Recommended
#   if you specifically want to stop one asset (e.g. Bitcoin) from carrying the result.
SCORE_AGGREGATION = "median"

# "per_asset" (default): score = SCORE_AGGREGATION-of-each-asset's-own-normalized-Sharpe/Calmar/etc,
#   exactly as above -- each asset simulated independently via td.simulate_asset(), same as always.
# "portfolio": score = a SINGLE Sharpe/Calmar/etc computed from td.simulate_portfolio()'s one
#   shared-capital equity curve across every asset at once -- rewards parameter sets that work well
#   for the REAL account (shared risk budget, portfolio-wide exposure/drawdown controls), not
#   whichever set merely does well asset-by-asset in isolation. SCORE_AGGREGATION is irrelevant in
#   this mode (there's only one curve to score, nothing to aggregate across).
SCORE_MODE = "per_asset"

# Set either of these to False to search with only ONE system active for EVERY trial -- the
# other system's lookback params are simply never sampled or included in td.SYSTEMS. Different
# from optimize_turtle_isolated.py (which runs two SEPARATE searches, one per system, then
# combines the winners): this just restricts the WHOLE joint search to one system throughout.
ENABLE_S1 = True
ENABLE_S2 = True
assert ENABLE_S1 or ENABLE_S2, "At least one system must be enabled"

# ----------------------------------------------------------------------------
# SEARCH SPACE
# ----------------------------------------------------------------------------
# System lookbacks are sampled as 4 separate independent parameters (S1's pair,
# S2's pair), then reassembled into the SYSTEMS list structure in apply_params.
# MAX_TOTAL_EXPOSURE_PCT is the ONLY thing left truly shared -- it's a constraint
# on total notional across BOTH systems combined, so it can't sensibly be per-system.
INDEPENDENT_PARAM_SPACE = {
    # S1 (fast system): kept a genuinely "fast" option at the low end, but reaches much higher
    # now too, so the search can find a slower, more selective S1 if that's what actually works
    "S1_BREAKOUT_LOOKBACK": [5, 7, 10, 15, 20, 25, 30, 40, 55, 70, 90],
    "S1_EXIT_LOOKBACK": [3, 5, 7, 10, 15, 20, 25, 35],
    # S2 (slow system): the big expansion -- this is the one meant to catch genuinely large,
    # multi-month trends, so it now reaches out to 150/200-day breakouts, not just 100
    "S2_BREAKOUT_LOOKBACK": [75, 100, 125, 150, 200, 250, 300, 400, 500],
    "S2_EXIT_LOOKBACK": [10, 15, 20, 30, 40, 60, 80, 100, 130],
    # Was a single fixed value of 5000% (50x notional leverage on ONE asset's S1+S2 combined,
    # for EVERY trial, not even actually searched) -- confirmed directly to produce results like
    # a -98% max drawdown and billions in final equity on a 12-asset portfolio run, an artifact
    # of the cap being essentially unconstrained rather than a genuine finding about the
    # strategy. Now a real range, grounded in what a leveraged-but-not-reckless futures/margin
    # account could plausibly run: 100% (no leverage at all if fully committed) up to 500% (5x).
    "MAX_TOTAL_EXPOSURE_PCT": [50, 75, 100, 150, 200, 300, 500],
}

# EVERY other changeable strategy parameter -- entry buffer, N (ATR) period, risk per unit,
# pyramid spacing, max units, stop multiple, time-stop, scale-out ("TP"), and every entry
# filter -- is sampled ONCE and SHARED by both systems (only the lookbacks stay per-system).
PER_SYSTEM_ALWAYS_SAMPLED = {
    "entry_buffer_n_mult": [0, 0.25, 0.5, 1.0, 2, 3, 4],
    # N period: coarser and reaching higher -- a longer N gives a smoother, slower-reacting
    # volatility measure, which fits a "hold through bigger moves" system better than a twitchy one
    "n_atr_period": [7, 10, 14, 20, 30, 40, 60, 80, 100, 150, 200, 300],
    "risk_per_trade_pct": [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "risk_per_trade_pct_schedule": [None, (0.5, 1.0), (0.5, 1.5), (1.0, 2.0),
                                      (0.25, 0.5, 1.0), (0.25, 0.5, 1.0, 1.5)],
    # pyramid spacing: the big one for "adds happen over weeks, not days" -- reaches out to 4N
    # now (vs. 1N max before), so a trial can require a MUCH bigger favorable move before adding
    "pyramid_spacing_n_mult": [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5, 6, 7, 8, 10],
    # NEW: an alternative way to decide WHEN to pyramid -- "spacing" (the original, above) adds
    # once price has moved a fixed multiple of N past the last add; "breakout" instead adds
    # whenever price makes a fresh N-day high (its own separate lookback, right below), mirroring
    # the entry signal's own logic but applied continuously to adds too; "swing_high" requires a
    # genuinely CONFIRMED swing pivot (real structure, not a rolling window) -- the sparsest,
    # most structurally-gated of the three, built specifically for months-long trend riding
    "pyramid_method": ["spacing", "breakout", "swing_high", "monthly_high"],
    "pyramid_breakout_lookback": [3, 5, 10, 15, 20],
    "swing_lookback": [5, 8, 10, 15, 20, 30],
    # 1 unit = no pyramiding at all -- a genuine "single-shot, let it run" option, directly in
    # the spirit of "fewer positions, bigger trends" rather than always stacking multiple units
    "max_units_per_trend": [1, 2, 3, 4, 5, 6, 8, 10],
    # wider stops give a real trend more room to breathe before getting shaken out
    "stop_n_mult": [1.5, 2, 3.0, 4.0, 5.0, 6, 7, 8, 10, 12, 15, 20],
    # TP now has its OWN ATR period and its OWN multiplier, fully decoupled from the stop's --
    # a trial can require a much bigger (or smaller) move to take profit than to get stopped out
    "tp_atr_period": [10, 14, 20, 30, 40, 60, 80, 100, 150, 200],
    "tp_n_mult": [0.5, 1.0, 1.5, 2.0, 3.0, 5, 10, 15, 20, 30],
    # kept ONLY for the optional regime_scale_out take-profit mechanic -- no entry-side ADX
    # filter exists anymore, but regime_scale_out still needs its own ADX reading
    "adx_period": [10, 14, 20, 30],
}

PER_SYSTEM_CONDITIONAL_TOGGLES = {
    "use_time_stop": {"time_stop_days": [5, 10, 20, 30, 45], "time_stop_min_r": [0.1, 0.25, 0.5]},
    "use_scale_out": {},   # "TP": SCALE_OUT_LEVELS itself isn't varied, just on/off per system
    # trailing stop lookback: reaches further out toward a "weeks" scale, not just 30 days max
    "use_n_day_low_trailing_stop": {"trailing_stop_lookback_days": [5, 10, 20, 30, 50]},
    "use_breakeven_stop": {"breakeven_trigger_r": [0.5, 1.0, 1.5, 2.0, 3.0]},
    # a wider Chandelier multiple gives more room before the trailing stop kicks in, consistent
    # with "let a real trend run" rather than trailing tightly
    "use_chandelier_exit": {"chandelier_atr_mult": [2.0, 3.0, 4.0, 5.0]},
    # regime_scale_out is a take-profit mechanic that reacts to trend strength MID-TRADE, not an
    # entry filter -- it's the only thing that still needs an ADX reading (adx_period lives in
    # PER_SYSTEM_ALWAYS_SAMPLED below, shared with this).
    "use_regime_scale_out": {"regime_scale_out_adx_threshold": [25, 30, 35, 40],
                               "regime_scale_out_fraction": [0.25, 0.5, 0.75]},
    # EMA trend entry filter: a meaningful long-term trend filter is usually a LONG EMA
    "use_ema_filter": {"ema_period": [50, 100, 150, 200, 300, 400]},
    # a moving-average regime in place of the breakout/exit channels (see td.DEFAULT_SYSTEM_PARAMS)
    "use_ma_regime": {"ma_regime_period": [100, 150, 200, 250, 300], "ma_regime_band_pct": [0.0, 1.0, 2.0, 3.0]},
    # enter on "the close breaks above an EMA" instead of the N-day-high breakout (see td.DEFAULT_SYSTEM_PARAMS); shared by S1 and S2 like everything
    # here except the lookbacks. "cross" = only the day it crosses, "above" = whenever flat and above it.
    "use_ema_entry": {"ema_entry_period": [20, 30, 50, 75, 100, 150, 200, 300], "ema_entry_mode": ["cross", "above"]},
    # leave on a close below an EMA instead of the exit-lookback channel, independent of the entry (a breakout entry can use it too)
    "use_ema_exit": {"ema_exit_period": [10, 20, 30, 50, 75, 100, 150, 200]},
    # monthly-high breakout entry (see td.DEFAULT_SYSTEM_PARAMS): enters, and adds on every further signal, when a monthly high (each completed month's high) that is mh_min_age..mh_max_age
    # bars old breaks out for the first time, with the close above its mh_sma_period-day EMA or SMA (mh_ma_type). Days are daily bars. Shared by S1 and S2.
    "use_monthly_high_entry": {"mh_min_age": [10, 15, 20, 30, 40, 60], "mh_max_age": [120, 180, 250, 365, 500], "mh_sma_period": [100, 150, 200, 250, 300],
                               "mh_ma_type": ["ema", "sma"]},
    # a resting trailing stop that many ATRs under the highest high since entry (see td.DEFAULT_SYSTEM_PARAMS)
    "use_peak_trail": {"peak_trail_atr_mult": [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]},
    # Choppiness Index entry filter: requires the market to genuinely be trending (CI below
    # threshold) before allowing an entry -- built directly from the regime diagnostic finding
    # that every asset performed best in its own "Trending" bucket by this exact measure
    "use_choppiness_filter": {"choppiness_period": [10, 14, 20, 30],
                                "choppiness_max": [30, 38.2, 45, 55, 61.8]},
    # entry must ALSO be a genuine all-time high (not just an N-day one) -- no sub-params
    "require_all_time_high": {},
    # anchors the pyramid stop ratchet to the AVERAGE entry price across open units instead of
    # always the latest one -- gentler tightening, doesn't need sub-params, just on/off
    "use_average_entry_stop": {},
    # NEW: whether the fixed stop tightens at all as units are pyramided -- if False, the stop
    # stays exactly where the first entry set it for the life of the position
    "ratchet_stop_on_pyramid": {},
    # NEW: the core ratcheting stop itself (exit_reason="SL") -- previously had no toggle at
    # all. Left freely searchable like everything else here; if disabling it tanks performance
    # (e.g. because trailing_stop/chandelier are also off), the backtest itself will show that
    # and the optimizer will naturally rank it low -- no special-casing needed
    "use_fixed_stop": {},
    # cooldown after a loss -- prevents immediately re-entering near the same level that just
    # stopped the position out (a real issue observed with wide, slow-moving breakout levels)
    "use_cooldown": {"cooldown_days": [3, 5, 10, 15, 20, 30]},
    # blocks a pyramid ADD (not the initial entry) if ADX is already above this -- built
    # directly from the regime diagnostic finding that high-ADX-confirmed moves underperform
    "use_pyramid_adx_filter": {"pyramid_max_adx": [25, 30, 35, 40, 45]},
    # per-unit trailing profit give-back (see td.DEFAULT_SYSTEM_PARAMS' own comment on
    # use_giveback_stop for the full mechanic) -- was missing from the search space entirely
    # despite existing in the engine, so every trial silently used the file default (off) no
    # matter what; added here so it's actually explorable like every other optional exit mechanic.
    "use_giveback_stop": {"giveback_fraction": [0.25, 0.4, 0.5, 0.6, 0.75],
                            "giveback_min_profit_r": [0.5, 1.0, 1.5, 2.0]},
}

# GLOBAL toggles -- sampled ONCE (not per system), since these are genuinely account-level, not
# specific to either system's own trading logic. Applied via the same generic setattr fallback
# in apply_params as MAX_TOTAL_EXPOSURE_PCT already is -- no special-casing needed there.
GLOBAL_CONDITIONAL_TOGGLES = {
    # the Turtles' account adjustment: a step every `threshold` of loss, each step cutting the notional equity by `factor`
    "USE_DRAWDOWN_SCALING": {"DRAWDOWN_SCALING_THRESHOLD": [0.05, 0.10, 0.15, 0.20],
                               "DRAWDOWN_SCALING_FACTOR": [0.10, 0.20, 0.30]},
}


# Per-market risk multipliers: {market name: candidate multipliers}. EMPTY by default = the multipliers
# are not searched and the manual td.ASSET_RISK_MULT applies. A caller (the UI, or a script) fills it
# with the markets of the run; a market's value is then the parameter td.ASSET_RISK_PARAM_PREFIX + name
# ("RISKMULT::Gold"), sampled like any other when named in only_optimize and read from fixed_values
# when not. apply_params() folds all of them into td.ASSET_RISK_MULT. Each searched market adds one
# dimension, so 15 markets is 15 more parameters to fit on the same history: see td.ASSET_RISK_NORMALIZE
# for keeping the search from simply raising everything.
ASSET_RISK_SPACE = {}


_SEARCH_SPACE_DICTS = {
    "independent": INDEPENDENT_PARAM_SPACE, "always": PER_SYSTEM_ALWAYS_SAMPLED,
    "conditional": PER_SYSTEM_CONDITIONAL_TOGGLES, "global": GLOBAL_CONDITIONAL_TOGGLES,
    "asset_risk": ASSET_RISK_SPACE,
}


def get_search_space() -> dict:
    """Deep copy of the four search-space dicts (the candidate values each parameter is sampled
    from), so a caller can edit them and hand the result to set_search_space() -- or ship it to a
    spawned worker process, which would otherwise re-import this file's own defaults."""
    import copy
    return {key: copy.deepcopy(d) for key, d in _SEARCH_SPACE_DICTS.items()}


def set_search_space(space: dict):
    """Replaces the contents of the module's search-space dicts IN PLACE (so every existing
    reference -- sample_params_raw, wf_fast.suggest_params, narrow_param_space -- sees the change)."""
    import copy
    for key, d in _SEARCH_SPACE_DICTS.items():
        d.clear()
        d.update(copy.deepcopy(space.get(key, {})))   # a space saved before a section existed lacks it


DEFAULT_SEARCH_SPACE = get_search_space()   # this file's own ranges, for "reset" and restoring after a run

_approx_independent_combos = 1
for _vals in INDEPENDENT_PARAM_SPACE.values():
    _approx_independent_combos *= len(_vals)
_approx_always_combos = 1
for _vals in PER_SYSTEM_ALWAYS_SAMPLED.values():
    _approx_always_combos *= len(_vals)
_approx_toggle_combos = 1
for _toggle, _subs in PER_SYSTEM_CONDITIONAL_TOGGLES.items():
    _sub_combos = 1
    for _vals in _subs.values():
        _sub_combos *= len(_vals)
    _approx_toggle_combos *= (1 + _sub_combos)
_approx_per_system_combos = _approx_always_combos * _approx_toggle_combos
_approx_global_toggle_combos = 1
for _toggle, _subs in GLOBAL_CONDITIONAL_TOGGLES.items():
    _sub_combos = 1
    for _vals in _subs.values():
        _sub_combos *= len(_vals)
    _approx_global_toggle_combos *= (1 + _sub_combos)
# lookbacks are still sampled per-system independently, but everything else (N, risk, stops,
# pyramiding, filters, time-stop, scale-out, etc.) is now sampled ONCE and shared by both
# systems -- so this is no longer squared
APPROX_SEARCH_SPACE_SIZE = _approx_independent_combos * _approx_per_system_combos * _approx_global_toggle_combos


# ----------------------------------------------------------------------------
# TRIAL EXECUTION
# ----------------------------------------------------------------------------
def is_valid_params(params: dict) -> bool:
    """Rejects structurally nonsensical combos before any compute is spent."""
    # the entry types replace one another (the monthly-high breakout beats the EMA break, which beats the moving-average regime): sampling two at once would
    # only hide one of them, so a trial may have at most one on
    if sum(bool(params.get(k)) for k in ("use_ma_regime", "use_ema_entry", "use_monthly_high_entry")) > 1:
        return False
    if params.get("use_monthly_high_entry") and params["mh_min_age"] >= params["mh_max_age"]:
        return False       # the high must be older than min_age but no older than max_age: an empty age band
    if td.MH_CLEAN_MODE:
        return True        # the clean monthly-high mode has no breakout / exit lookbacks (and one system only)
    if not td.ALLOW_EXIT_ABOVE_ENTRY:
        if ENABLE_S1 and params["S1_EXIT_LOOKBACK"] >= params["S1_BREAKOUT_LOOKBACK"]:
            return False
        if ENABLE_S2 and params["S2_EXIT_LOOKBACK"] >= params["S2_BREAKOUT_LOOKBACK"]:
            return False
    # the "S1 is faster than S2" constraint only makes sense when BOTH are actually in play
    if ENABLE_S1 and ENABLE_S2 and params["S1_BREAKOUT_LOOKBACK"] >= params["S2_BREAKOUT_LOOKBACK"]:
        return False
    return True


def params_to_systems(params: dict) -> list:
    """Reconstructs the td.SYSTEMS list from a flat params dict -- only including
    whichever system(s) are enabled via ENABLE_S1/ENABLE_S2. Breakout/exit
    lookbacks stay genuinely per-system (that's what makes S1 "fast" and S2
    "slow"). EVERY other parameter -- N period, risk %, pyramiding, stops,
    time-stop, scale-out, and every filter -- is sampled ONCE and applied
    IDENTICALLY to whichever system(s) are active, since that's the point:
    same risk management approach, only the time horizon differs."""
    p = dict(params)
    system_param_keys = list(td.DEFAULT_SYSTEM_PARAMS.keys())

    shared_params = dict(td.DEFAULT_SYSTEM_PARAMS)
    for key in system_param_keys:
        if key in p:
            shared_params[key] = p[key]

    if td.MH_CLEAN_MODE:
        # the clean monthly-high mode: ONE system with only the entry / pyramiding / stop / take-profit mechanics. The constants win over anything else in the params dict, so a stray
        # key can never switch another mechanism back on; the searched values (mh_*, N / TP periods, multipliers, risk %) come through from p as usual. The lookbacks are unused
        # (the monthly-high level replaces the breakout level, and there is no channel exit) - they are only placeholders.
        shared_params.update(td.MH_CLEAN_SYSTEM_PARAMS)
        shared_params["entry_buffer_n_mult"] = p.get("entry_buffer_n_mult", 0.0)       # a setting of the entry (x ATR above the monthly high), not the 0.25 of the standard defaults
        return [{"name": "S1", "breakout_lookback": p.get("S1_BREAKOUT_LOOKBACK", 20),
                 "exit_lookback": p.get("S1_EXIT_LOOKBACK", 10), **shared_params}]

    systems = []
    if ENABLE_S1:
        systems.append({"name": "S1", "breakout_lookback": p["S1_BREAKOUT_LOOKBACK"],
                         "exit_lookback": p["S1_EXIT_LOOKBACK"], **shared_params})
    if ENABLE_S2:
        systems.append({"name": "S2", "breakout_lookback": p["S2_BREAKOUT_LOOKBACK"],
                         "exit_lookback": p["S2_EXIT_LOOKBACK"], **shared_params})
    return systems


def apply_params(params: dict):
    """Monkey-patch turtle_dual_system's globals with this trial's parameter set."""
    p = dict(params)
    td.SYSTEMS = params_to_systems(params)
    system_param_keys = list(td.DEFAULT_SYSTEM_PARAMS.keys())
    for prefix in ("S1", "S2"):
        p.pop(f"{prefix}_BREAKOUT_LOOKBACK", None)
        p.pop(f"{prefix}_EXIT_LOOKBACK", None)
    for key in system_param_keys:
        p.pop(key, None)
    prefix = td.ASSET_RISK_PARAM_PREFIX
    market_mults = {name[len(prefix):]: p.pop(name) for name in [k for k in p if k.startswith(prefix)]}
    if market_mults:
        td.ASSET_RISK_MULT = market_mults   # a complete set: every market is either sampled or pinned
    for name, value in p.items():
        setattr(td, name, value)
    if td.MH_CLEAN_MODE:
        td.USE_DRAWDOWN_SCALING = False      # not one of the four mechanics


def _run_trial_portfolio(features: dict, params: dict) -> dict | None:
    """SCORE_MODE == "portfolio" path: one shared-capital simulation across every asset via
    td.simulate_portfolio(), scored off that SINGLE equity curve -- no per-asset aggregation,
    since there's only one curve. Returns the same dict SHAPE as the per-asset path below (so
    main()'s reporting/CSV code needs no separate branch), but per_asset_metric is NaN
    placeholders: there genuinely isn't a per-asset Sharpe/Calmar/CAGR when capital is shared."""
    try:
        port_result = td.simulate_portfolio(features)
    except Exception:
        return None

    total_trades = sum(len(trades) for trades in port_result["trades_by_asset"].values())
    if total_trades == 0:
        return None
    try:
        stats = td.performance_stats(port_result["equity_curve"], port_result["starting_capital"])
    except Exception:
        return None
    # Gated on whichever metric RANK_METRIC actually ranks by, not always Sharpe -- same reasoning
    # as the per-asset path's own fix (see run_trial above).
    rank_stats_key = {"sharpe": "Sharpe", "calmar": "Calmar", "cagr": "CAGR",
                        "total_return": "Total Return"}[RANK_METRIC]
    if not np.isfinite(stats.get(rank_stats_key, np.nan)):
        return None

    per_asset_trades = {name: len(trades) for name, trades in port_result["trades_by_asset"].items()}
    per_system_trades = {
        sysdef["name"]: sum(len(port_result["trades_by_system"][name][sysdef["name"]]) for name in features)
        for sysdef in td.SYSTEMS
    }
    return {
        "params": dict(params),
        "sharpe": stats["Sharpe"], "calmar": stats["Calmar"] if np.isfinite(stats["Calmar"]) else 0.0,
        "cagr": stats["CAGR"], "total_return": stats["Total Return"],
        "per_asset_metric": {name: {"sharpe": np.nan, "calmar": np.nan, "cagr": np.nan, "total_return": np.nan}
                                for name in features},
        "max_dd": stats["Max Drawdown"], "final_equity": stats["Final Equity"], "total_trades": total_trades,
        "per_asset_trades": per_asset_trades, "per_system_trades": per_system_trades,
    }


def run_trial(raw_data: dict, params: dict, ath_references: dict = None, features_fn=None) -> dict | None:
    """features_fn(name, df) -> features DataFrame: defaults to td.build_features; workers pass a
    wf_fast.FeatureCache's build, which returns identical frames from cached indicators."""
    apply_params(params)
    try:
        if features_fn is None:
            features = {}
            for name, df in raw_data.items():
                features[name] = td.build_features(df, full_history_ath=(ath_references or {}).get(name))
                features[name].attrs["asset"] = name   # fee schedule lookup (td.commission_for)
        else:
            features = {name: features_fn(name, df) for name, df in raw_data.items()}
    except Exception:
        return None

    if SCORE_MODE == "portfolio":
        return _run_trial_portfolio(features, params)

    try:
        results = {name: td.simulate_asset(df) for name, df in features.items()}
    except Exception:
        return None

    total_trades = sum(len(r["trades"]) for r in results.values())
    if total_trades == 0:
        return None

    # Score on PER-ASSET normalized performance (Sharpe/Calmar/etc, each computed against
    # that asset's OWN starting capital), then aggregate across assets -- NOT on the dollar-
    # summed combined equity curve. Dollar-summing lets whichever asset has the biggest DOLLAR
    # swings (Bitcoin, given its volatility, even with equal starting capital per asset)
    # mechanically dominate the combined curve's Sharpe/Calmar -- the optimizer would then just
    # be tuning itself to whatever makes that one asset's trades look good, ignoring the rest.
    # Per-asset Sharpe is a unitless ratio, so no asset's dollar scale can dominate the average.
    per_asset_stats = {}
    for name, r in results.items():
        try:
            per_asset_stats[name] = td.performance_stats(r["equity_curve"], td.STARTING_CAPITAL_PER_ASSET)
        except Exception:
            per_asset_stats[name] = None

    # Gated on whichever metric RANK_METRIC actually ranks by, not always Sharpe -- an asset with
    # a perfectly valid CAGR/Total Return but a non-finite Sharpe (e.g. zero-variance returns)
    # used to get silently dropped from the aggregate even when ranking by CAGR/Total Return,
    # where that asset's result was entirely usable.
    rank_stats_key = {"sharpe": "Sharpe", "calmar": "Calmar", "cagr": "CAGR",
                        "total_return": "Total Return"}[RANK_METRIC]
    valid_assets = [name for name, s in per_asset_stats.items()
                    if s is not None and np.isfinite(s.get(rank_stats_key, np.nan))]
    if not valid_assets:
        return None

    per_asset_metric = {}
    for name in valid_assets:
        s = per_asset_stats[name]
        per_asset_metric[name] = {
            "sharpe": s["Sharpe"], "calmar": s["Calmar"] if np.isfinite(s["Calmar"]) else 0.0,
            "cagr": s["CAGR"], "total_return": s["Total Return"],
        }

    def aggregate(metric_key):
        values = np.array([per_asset_metric[name][metric_key] for name in valid_assets])
        if SCORE_AGGREGATION == "mean":
            return values.mean()
        elif SCORE_AGGREGATION == "median":
            return np.median(values)
        elif SCORE_AGGREGATION == "min":
            return values.min()
        else:
            raise ValueError(f"Unknown SCORE_AGGREGATION: {SCORE_AGGREGATION}")

    score_sharpe = aggregate("sharpe")
    score_calmar = aggregate("calmar")
    score_cagr = aggregate("cagr")
    score_total_return = aggregate("total_return")

    # Same reasoning as the valid_assets gate above -- checks whichever score RANK_METRIC
    # actually ranks by, not always Sharpe (which, when ranking by a different metric, can still
    # be NaN here even after that gate: aggregate("sharpe") propagates NaN from any INCLUDED
    # asset whose own Sharpe happened to be non-finite despite having a perfectly valid
    # CAGR/Total Return, since plain mean/median/min don't skip NaN the way nanmean would).
    score_by_rank_metric = {"sharpe": score_sharpe, "calmar": score_calmar, "cagr": score_cagr,
                              "total_return": score_total_return}[RANK_METRIC]
    if not np.isfinite(score_by_rank_metric):
        return None

    # combined dollar curve is still computed for REPORTING (final equity, headline numbers) --
    # just no longer used to RANK trials
    date_range = pd.date_range(
        min(r["equity_curve"].index.min() for r in results.values()),
        max(r["equity_curve"].index.max() for r in results.values()), freq="D")
    total_capital = td.STARTING_CAPITAL_PER_ASSET * len(raw_data)
    combined_equity = sum(
        results[name]["equity_curve"].reindex(date_range).ffill().fillna(td.STARTING_CAPITAL_PER_ASSET)
        for name in results)
    combined_final_equity = combined_equity.iloc[-1]

    per_asset_trades = {name: len(r["trades"]) for name, r in results.items()}
    per_system_trades = {
        sysdef["name"]: sum(len(r["trades_by_system"][sysdef["name"]]) for r in results.values())
        for sysdef in td.SYSTEMS
    }

    return {
        "params": dict(params),
        "sharpe": score_sharpe, "calmar": score_calmar, "cagr": score_cagr, "total_return": score_total_return,
        "per_asset_metric": per_asset_metric,
        "max_dd": min(per_asset_stats[name]["Max Drawdown"] for name in valid_assets),   # worst single-asset drawdown
        "final_equity": combined_final_equity, "total_trades": total_trades,
        "per_asset_trades": per_asset_trades, "per_system_trades": per_system_trades,
    }


def sample_params_raw(rng: random.Random, independent_space: dict = None, always_sampled: dict = None,
                        conditional_toggles: dict = None, toggle_bias: dict = None,
                        only_optimize: set = None, fixed_values: dict = None) -> dict:
    """only_optimize (optional): if given, ONLY these parameter names are actually randomized --
    every other name (independent_space/always_sampled key, or a toggle/sub-param name) is pinned
    to fixed_values[name] instead of sampled at all, for EVERY trial. None (the default) means
    the old, unrestricted behavior -- everything above gets sampled, matching every direct script
    run and every call site that doesn't opt into this (e.g. turtle_ui.py's per-parameter
    optimizer panel is what actually populates these two args; nothing else needs to).
    A name that's excluded from only_optimize MUST have a value in fixed_values, or this raises
    a KeyError -- deliberate, so a caller can't silently end up with an unset parameter.
    """
    independent_space = INDEPENDENT_PARAM_SPACE if independent_space is None else independent_space
    always_sampled = PER_SYSTEM_ALWAYS_SAMPLED if always_sampled is None else always_sampled
    conditional_toggles = PER_SYSTEM_CONDITIONAL_TOGGLES if conditional_toggles is None else conditional_toggles
    toggle_bias = toggle_bias or {}
    fixed_values = fixed_values or {}

    def pick(name, values):
        if only_optimize is not None and name not in only_optimize:
            return fixed_values[name]
        return rng.choice(values)

    params = {name: pick(name, values) for name, values in independent_space.items()}
    # lookbacks stay per-system (S1_/S2_ prefixed) -- that's the "fast vs slow" distinction.
    # EVERYTHING else is sampled ONCE and shared by both systems: same N period, risk %,
    # pyramiding, stops, time-stop/scale-out, and filters for S1 and S2 alike.
    for name, values in always_sampled.items():
        params[name] = pick(name, values)
    for toggle_name, sub_params in conditional_toggles.items():
        if only_optimize is not None and toggle_name not in only_optimize:
            is_on = fixed_values[toggle_name]
        else:
            # toggle_bias[toggle_name], if given, is the probability of sampling True -- defaults
            # to a blind 50/50 coin flip when not refining based on a prior stage's results
            p_true = toggle_bias.get(toggle_name, 0.5)
            is_on = rng.random() < p_true
        params[toggle_name] = is_on
        if is_on:
            for sub_name, sub_values in sub_params.items():
                params[sub_name] = pick(sub_name, sub_values)
    # global/account-level toggles -- sampled once, applied directly as td module attributes.
    # NOT narrowed in the refinement stage (see narrow_param_space) -- left at full range always.
    for toggle_name, sub_params in GLOBAL_CONDITIONAL_TOGGLES.items():
        if only_optimize is not None and toggle_name not in only_optimize:
            is_on = fixed_values[toggle_name]
        else:
            is_on = rng.choice([True, False])
        params[toggle_name] = is_on
        if is_on:
            for sub_name, sub_values in sub_params.items():
                params[sub_name] = pick(sub_name, sub_values)
    for market, values in ASSET_RISK_SPACE.items():
        name = td.ASSET_RISK_PARAM_PREFIX + market
        params[name] = pick(name, values)
    return params


def sample_unique_param_sets(rng: random.Random, n: int, max_attempts_mult: int = 30,
                               independent_space: dict = None, always_sampled: dict = None,
                               conditional_toggles: dict = None, toggle_bias: dict = None,
                               only_optimize: set = None, fixed_values: dict = None) -> list:
    """Draws n parameter sets, skipping duplicates AND structurally invalid combos.
    only_optimize/fixed_values: see sample_params_raw -- passed straight through."""
    seen = set()
    unique_params = []
    attempts = 0
    max_attempts = n * max_attempts_mult
    while len(unique_params) < n and attempts < max_attempts:
        attempts += 1
        params = sample_params_raw(rng, independent_space, always_sampled, conditional_toggles, toggle_bias,
                                     only_optimize, fixed_values)
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
            unique_params.append(sample_params_raw(rng, independent_space, always_sampled, conditional_toggles,
                                                      toggle_bias, only_optimize, fixed_values))
    return unique_params


def _narrow_value_list(original_values: list, used_values: set, window: int = 1) -> list:
    """Narrows a candidate list to the neighborhood of values actually used by the top trials
    from a prior stage. Short lists (<=3 -- e.g. booleans) use window=0, an exact match only,
    since there's no meaningful 'neighbor' of True/False. Longer numeric lists include immediate
    neighbors too, so the refined stage can still explore slightly beyond the exact values that
    happened to win, rather than locking in on them completely."""
    effective_window = 0 if len(original_values) <= 3 else window
    matched_indices = set()
    for v in used_values:
        if v in original_values:
            idx = original_values.index(v)
            for w in range(-effective_window, effective_window + 1):
                ni = idx + w
                if 0 <= ni < len(original_values):
                    matched_indices.add(ni)
    if not matched_indices:
        return original_values   # nothing matched -- keep the full range, don't narrow blindly
    return [original_values[i] for i in sorted(matched_indices)]


def narrow_param_space(top_params_list: list, window: int = 1):
    """Builds narrowed versions of INDEPENDENT_PARAM_SPACE, PER_SYSTEM_ALWAYS_SAMPLED, and
    PER_SYSTEM_CONDITIONAL_TOGGLES, concentrated around whatever values the best trials from an
    earlier stage actually used -- the core of the coarse-to-fine search: a broad blind search
    first, then a denser search concentrated in the neighborhood of what worked, rather than
    spending the entire trial budget scattered uniformly across an astronomically large space.

    Also returns toggle_bias: the fraction of top trials that had each toggle ON, used to weight
    stage 2's on/off coin flip toward what actually worked -- without this, narrowing a toggle's
    SUB-parameters would be pointless half the time, since the toggle itself would still get
    flipped off by a blind 50/50 draw regardless of what stage 1 found.

    GLOBAL_CONDITIONAL_TOGGLES is deliberately left untouched (always full range, always 50/50)
    -- it's a small, cheap space on its own and narrowing it isn't where the real benefit is.
    """
    narrowed_independent = {}
    for name, values in INDEPENDENT_PARAM_SPACE.items():
        used = {p[name] for p in top_params_list if name in p}
        narrowed_independent[name] = _narrow_value_list(values, used, window)

    narrowed_always = {}
    for name, values in PER_SYSTEM_ALWAYS_SAMPLED.items():
        used = {p[name] for p in top_params_list if name in p}
        narrowed_always[name] = _narrow_value_list(values, used, window)

    narrowed_toggles = {}
    toggle_bias = {}
    for toggle_name, sub_params in PER_SYSTEM_CONDITIONAL_TOGGLES.items():
        relevant = [p[toggle_name] for p in top_params_list if toggle_name in p]
        toggle_bias[toggle_name] = (sum(relevant) / len(relevant)) if relevant else 0.5
        narrowed_toggles[toggle_name] = {}
        for sub_name, sub_values in sub_params.items():
            # only look at values used AMONG trials where this toggle was actually True --
            # a trial where the toggle was off never sampled a value for its sub-params at all
            used = {p[sub_name] for p in top_params_list if p.get(toggle_name) and sub_name in p}
            narrowed_toggles[toggle_name][sub_name] = _narrow_value_list(sub_values, used, window)

    return narrowed_independent, narrowed_always, narrowed_toggles, toggle_bias


# ----------------------------------------------------------------------------
# MULTIPROCESSING WORKER PLUMBING
# ----------------------------------------------------------------------------
_worker_raw_data = None
_worker_ath_references = None
_worker_cache = None


_PORTFOLIO_CONFIG_ATTRS = ("PORTFOLIO_STARTING_CAPITAL", "USE_PORTFOLIO_RISK_CAP", "MAX_PORTFOLIO_RISK_PCT",
                            "USE_PER_CONTRACT_FEES", "FEE_SCHEDULE", "HOLDING_COST_PCT_PER_YEAR",
                            "SLIPPAGE_PCT", "SPREAD_PCT", "COMMISSION_PCT",
                            "MARK_TO_MARKET", "SIZE_STEP", "TAKE_MIN_ONE_UNIT", "USE_MARGIN_CALLS",
                            "MARGIN_SPEC", "DEFAULT_MARGIN_RATES", "ASSET_COSTS", "EXECUTION_TIMING",
                            "STOP_PRIORITY", "ASSET_PRIORITY", "MAX_GROSS_EXPOSURE_PCT", "VOL_TARGET_PCT", "VOL_TARGET_WINDOW",
                            "VOL_TARGET_FLOOR", "VOL_TARGET_CAP", "HITRATE_WINDOW", "HITRATE_MIN_AVG_R",
                            "HITRATE_SCALE", "ATH_RISK_MULT", "BELOW_ATH_RISK_MULT", "ATH_FIRST_UNIT_ONLY", "ASSET_RISK_MULT",
                            "ASSET_RISK_NORMALIZE", "CHOP_RISK_MULT", "CHOP_TREND_MAX", "ATH_MAX_DISTANCE_PCT",
                            "ALLOW_EXIT_ABOVE_ENTRY", "CARRY_MODE", "CARRY_RATE_SERIES", "CARRY_RATE_FLAT",
                            "CARRY_DIVIDEND_YIELD", "STOP_SLIPPAGE_PCT", "DRAWDOWN_SCALING_REFERENCE",
                            "TAIL_WINDOW", "TAIL_MIN_BARS", "TAIL_FALLBACK", "GAP_SIZING_K", "STRESS_TOTAL_CAP_PCT",
                            "STRESS_SINGLE_CAP_PCT", "STRESS_CLUSTER_CAP_PCT", "MARKET_CLUSTERS", "BOOK_VOL_CAP_PCT",
                            "BOOK_VOL_WINDOW", "EQUITY_MA_WINDOW", "EQUITY_MA_SCALE", "SHOCK_PAUSE_LOSS_PCT",
                            "SHOCK_PAUSE_DAYS", "EXTENSION_MA", "EXTENSION_N_MULT", "EXTENSION_RISK_MULT",
                            "DAILY_LOSS_STOP_N_MULT", "DAILY_LOSS_STOP_PCT", "ENTRY_TOP_K", "ENTRY_STRENGTH_LOOKBACK",
                            "ENTRY_MIN_STRENGTH", "ENTRY_RANK_FIRST_UNIT_ONLY", "VOL_RISK_GAMMA", "VOL_RISK_ATR", "VOL_RISK_REF",
                            "VOL_RISK_MIN", "VOL_RISK_MAX", "ENTRY_EMA_PERIOD", "ENTRY_EMA_MODE", "ENTRY_EMA_EXIT", "MH_CLEAN_MODE")


def current_portfolio_config() -> dict:
    """Snapshot of turtle_dual_system's own portfolio-mode config, for threading through
    multiprocessing.Pool's initargs (see _init_worker) -- these live on `td`, not this module, so
    apply_params()'s usual param-sampling path never touches them; a caller like turtle_ui.py's
    portfolio-mode checkbox sets them directly on td right before running, and worker processes
    need that exact snapshot too, not the file's own defaults."""
    return {name: getattr(td, name) for name in _PORTFOLIO_CONFIG_ATTRS}


def _init_worker(raw_data: dict, ath_references: dict = None, score_mode: str = None,
                   portfolio_config: dict = None, score_aggregation: str = None,
                   enable_s1: bool = None, enable_s2: bool = None):
    # score_mode / portfolio_config / score_aggregation / enable_s1 / enable_s2: explicitly
    # re-applied here, not just left to read the module-level globals -- on Windows,
    # multiprocessing.Pool uses spawn (not fork), so each worker process re-imports this module
    # (and turtle_dual_system) FRESH from disk and would otherwise see the file's own defaults
    # ("median", ENABLE_S1=ENABLE_S2=True, etc.), silently ignoring whatever the caller (e.g.
    # turtle_ui.py's score-aggregation selectbox or Enable S1/S2 checkboxes) set at runtime, the
    # moment N_WORKERS > 1. Passed through Pool's initargs (pickled, like raw_data already is) so
    # every worker actually gets the SAME config the parent process intended.
    # Confirmed previously missing for score_aggregation/enable_s1/enable_s2 specifically: a trial
    # run in a worker process would silently score with "median" aggregation regardless of what
    # the UI showed, and silently simulate BOTH S1 and S2 even when the user disabled one -- the
    # printed "ranking metric"/"systems active" summary and the worker's ACTUAL behavior would
    # quietly disagree, every single parallel run (the default; N_WORKERS defaults > 1).
    global _worker_raw_data, _worker_ath_references, _worker_cache
    global SCORE_MODE, SCORE_AGGREGATION, ENABLE_S1, ENABLE_S2
    _worker_raw_data = raw_data
    _worker_ath_references = ath_references
    import wf_fast   # lazy: wf_fast imports this module at its own top level
    _worker_cache = wf_fast.FeatureCache(raw_data, ath_references)   # indicators computed once per worker
    if score_mode is not None:
        SCORE_MODE = score_mode
    if portfolio_config is not None:
        for name, value in portfolio_config.items():
            setattr(td, name, value)
    if score_aggregation is not None:
        SCORE_AGGREGATION = score_aggregation
    if enable_s1 is not None:
        ENABLE_S1 = enable_s1
    if enable_s2 is not None:
        ENABLE_S2 = enable_s2


def _run_trial_worker(params: dict):
    return run_trial(_worker_raw_data, params, ath_references=_worker_ath_references,
                     features_fn=_worker_cache.build)


def search_optuna(raw_data: dict, ath_references: dict, n_trials: int, n_workers: int,
                    only_optimize: set = None, fixed_values: dict = None, seed: int = None,
                    progress_cb=None, label: str = "search", multivariate: bool = False) -> tuple:
    """Optuna (TPE) search over the same space as sample_unique_param_sets(), maximizing the
    RANK_METRIC score. Returns (results, n_failed) in the same shape as _run_trial_batch, so
    every caller's reporting works unchanged.

    Parallelism: ONE persistent worker pool for the whole search. Each round asks the sampler for
    n_workers parameter sets, runs them in parallel, then tells Optuna all the scores before the
    next round -- so TPE still learns round to round, just in batches instead of one at a time.
    The budget (n_trials) counts actual simulations: structurally invalid combinations and exact
    repeats of an already-scored parameter set are answered without re-running and don't count.

    progress_cb(sims_done, n_trials, n_valid, n_failed): called after every round."""
    import optuna
    import wf_fast   # lazy: wf_fast imports this module at its own top level
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    seed = RANDOM_SEED if seed is None else seed
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=multivariate, group=multivariate,
                                           n_startup_trials=max(10, n_trials // 4))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    pruned = optuna.trial.TrialState.PRUNED
    metric_key = {"sharpe": "sharpe", "calmar": "calmar", "cagr": "cagr", "total_return": "total_return"}[RANK_METRIC]

    results, seen = [], {}
    sims = failed = asks = 0
    pool = None
    if n_workers > 1:
        pool = mp.Pool(processes=n_workers, initializer=_init_worker,
                       initargs=(raw_data, ath_references, SCORE_MODE, current_portfolio_config(),
                                 SCORE_AGGREGATION, ENABLE_S1, ENABLE_S2))
    else:
        _init_worker(raw_data, ath_references)
    try:
        while sims < n_trials and asks < n_trials * 5:
            round_size = max(1, min(max(n_workers, 1), n_trials - sims))
            pending, to_run = [], {}
            for _ in range(round_size):
                trial = study.ask()
                asks += 1
                params = wf_fast.suggest_params(trial, only_optimize, fixed_values)
                if not is_valid_params(params):
                    study.tell(trial, state=pruned)
                    continue
                sig = wf_fast.params_signature(params)
                if sig in seen:
                    if seen[sig] is None:
                        study.tell(trial, state=pruned)
                    else:
                        study.tell(trial, seen[sig])
                    continue
                pending.append((trial, sig))
                to_run.setdefault(sig, params)
            if not to_run:
                continue
            sigs = list(to_run)
            param_list = [to_run[s] for s in sigs]
            outputs = (pool.map(_run_trial_worker, param_list, chunksize=1) if pool is not None
                       else [_run_trial_worker(p) for p in param_list])
            sims += len(sigs)
            for sig, result in zip(sigs, outputs):
                if result is None:
                    failed += 1
                    seen[sig] = None
                else:
                    seen[sig] = result[metric_key]
                    results.append(result)
            for trial, sig in pending:
                if seen[sig] is None:
                    study.tell(trial, state=pruned)
                else:
                    study.tell(trial, seen[sig])
            if progress_cb is not None:
                progress_cb(sims, n_trials, len(results), failed)
    finally:
        if pool is not None:
            pool.terminate()
    return results, failed


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def _run_trial_batch(param_list: list, raw_data: dict, ath_references: dict, n_workers: int,
                       stage_label: str, start_time: float) -> tuple:
    """Runs one batch of trials (parallel or sequential), printing progress. Shared by both
    stage 1 (broad) and stage 2 (refined) of the coarse-to-fine search, and by the old
    single-stage path when USE_TWO_STAGE_SEARCH is False.
    Returns (results, n_failed).
    """
    total = len(param_list)
    results = []
    n_failed = 0
    if n_workers > 1:
        with mp.Pool(processes=n_workers, initializer=_init_worker,
                      initargs=(raw_data, ath_references, SCORE_MODE, current_portfolio_config(),
                                 SCORE_AGGREGATION, ENABLE_S1, ENABLE_S2)) as pool:
            for trial_num, result in enumerate(
                    pool.imap_unordered(_run_trial_worker, param_list, chunksize=1), start=1):
                if result is None:
                    n_failed += 1
                else:
                    results.append(result)
                if trial_num % 20 == 0 or trial_num == total:
                    elapsed = time.time() - start_time
                    rate = trial_num / elapsed if elapsed > 0 else 0
                    print(f"  [{stage_label}] ...{trial_num}/{total} trials done ({elapsed:.0f}s elapsed, "
                          f"{rate:.1f} trials/s, {len(results)} valid, {n_failed} skipped)")
    else:
        _init_worker(raw_data, ath_references)
        for trial_num, params in enumerate(param_list, start=1):
            result = _run_trial_worker(params)
            if result is None:
                n_failed += 1
            else:
                results.append(result)
            if trial_num % 20 == 0 or trial_num == total:
                elapsed = time.time() - start_time
                rate = trial_num / elapsed if elapsed > 0 else 0
                print(f"  [{stage_label}] ...{trial_num}/{total} trials done ({elapsed:.0f}s elapsed, "
                      f"{rate:.1f} trials/s, {len(results)} valid, {n_failed} skipped)")
    return results, n_failed


def main():
    print("=" * 78)
    print(f"DUAL-SYSTEM TURTLE STRATEGY -- PARAMETER OPTIMIZATION ({N_TRIALS} trials)")
    print("=" * 78)
    print(f"Ranking metric: {RANK_METRIC}")
    print(f"Score mode: {SCORE_MODE}" + (
        f" -- ONE shared-capital equity curve across every asset (td.simulate_portfolio()); "
        f"SCORE_AGGREGATION does not apply, there's only one curve to score"
        if SCORE_MODE == "portfolio" else
        f" -- each asset simulated independently, combined via {SCORE_AGGREGATION}-of-per-asset"))
    print(f"Systems active for every trial: S1={ENABLE_S1}, S2={ENABLE_S2}")
    print(f"Approximate search space size: ~{APPROX_SEARCH_SPACE_SIZE:,} combinations "
          f"(before the S1/S2 structural constraint, which reduces this further)")
    print("Per-system (S1 and S2 sampled independently -- this is what makes S1 'fast' and S2 'slow'):")
    for name, values in INDEPENDENT_PARAM_SPACE.items():
        if (name.startswith("S1_") and not ENABLE_S1) or (name.startswith("S2_") and not ENABLE_S2):
            continue   # don't show a disabled system's lookback range as if it's part of the search
        if name.startswith("S1_") or name.startswith("S2_"):
            print(f"  {name}: {values}")
    print("Portfolio-level (applies once, shared across both systems combined):")
    for name, values in INDEPENDENT_PARAM_SPACE.items():
        if not (name.startswith("S1_") or name.startswith("S2_")):
            print(f"  {name}: {values}")
    print("\nEVERYTHING below is sampled ONCE and applied IDENTICALLY to both S1 and S2 -- same N")
    print("period, risk %, pyramiding, stops, time-stop/scale-out, and filters for both systems;")
    print("only the lookbacks above differ between them:")
    print("  Always varied (shared):")
    for name, values in PER_SYSTEM_ALWAYS_SAMPLED.items():
        print(f"    {name}: {values}")
    print("  Conditional (shared; on/off switch; sub-params only sampled when ON):")
    for toggle, subs in PER_SYSTEM_CONDITIONAL_TOGGLES.items():
        sub_desc = ", ".join(f"{k}={v}" for k, v in subs.items()) if subs else "(no sub-parameters)"
        print(f"    {toggle}: {sub_desc}")
    print("\nGlobal/account-level toggles (sampled once, not per-system):")
    for toggle, subs in GLOBAL_CONDITIONAL_TOGGLES.items():
        sub_desc = ", ".join(f"{k}={v}" for k, v in subs.items()) if subs else "(no sub-parameters)"
        print(f"  {toggle}: {sub_desc}")
    print("\nNOT optimized (fixed, real trading costs): SLIPPAGE_PCT, SPREAD_PCT, COMMISSION_PCT, "
          "HOLDING_COST_PCT_PER_YEAR")

    print("\nDownloading data (once, shared with all worker processes)...")
    raw_data = td.download_data(td.TICKERS, td.START_DATE, td.END_DATE)

    # require_all_time_high is part of the search space (any trial might sample it True), so
    # fetch the full-history reference once here too, same reasoning as td.main() -- a one-time
    # cost regardless of how many trials actually end up using it
    print("\nFetching full available history for a genuine all-time-high reference "
          "(separate from the backtest window above, used by any trial that samples "
          "require_all_time_high=True)...")
    ath_references = td.fetch_all_time_highs({name: td.TICKERS[name] for name in raw_data})

    original_systems = td.SYSTEMS
    # real td module attributes this optimizer touches directly now: MAX_TOTAL_EXPOSURE_PCT plus
    # the global drawdown-scaling toggle + its sub-params -- everything else lives inside
    # td.SYSTEMS, which original_systems already covers
    all_scalar_param_names = [p for p in INDEPENDENT_PARAM_SPACE
                                if p not in ("S1_BREAKOUT_LOOKBACK", "S1_EXIT_LOOKBACK",
                                             "S2_BREAKOUT_LOOKBACK", "S2_EXIT_LOOKBACK")]
    all_scalar_param_names += list(GLOBAL_CONDITIONAL_TOGGLES)
    all_scalar_param_names += [sub for subs in GLOBAL_CONDITIONAL_TOGGLES.values() for sub in subs]
    original_config = {name: getattr(td, name) for name in all_scalar_param_names}

    n_workers = N_WORKERS or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, N_TRIALS))
    print(f"\nUsing {n_workers} worker process(es) "
          f"({'parallel' if n_workers > 1 else 'sequential -- only 1 core available or N_WORKERS=1'})")

    metric_key = {"sharpe": "sharpe", "calmar": "calmar", "cagr": "cagr", "total_return": "total_return"}[RANK_METRIC]
    rng = random.Random(RANDOM_SEED)
    start_time = time.time()
    results = []
    n_failed = 0
    stage1_best_score = None

    if USE_TWO_STAGE_SEARCH and N_TRIALS >= 20:
        n_stage1 = max(10, int(N_TRIALS * STAGE1_FRACTION))
        n_stage2 = N_TRIALS - n_stage1
        print(f"\nCoarse-to-fine search: {n_stage1} broad trials, then {n_stage2} refined trials "
              f"concentrated around the top {TOP_K_FOR_REFINEMENT} of stage 1.")

        print(f"\n-- STAGE 1: broad search over the full space ({n_stage1} trials) --")
        stage1_params = sample_unique_param_sets(rng, n_stage1)
        stage1_results, stage1_failed = _run_trial_batch(stage1_params, raw_data, ath_references,
                                                            n_workers, "stage 1/broad", start_time)
        results.extend(stage1_results)
        n_failed += stage1_failed

        if stage1_results and n_stage2 > 0:
            stage1_sorted = sorted(stage1_results, key=lambda r: r[metric_key], reverse=True)
            stage1_best_score = stage1_sorted[0][metric_key]
            top_k = stage1_sorted[:min(TOP_K_FOR_REFINEMENT, len(stage1_sorted))]
            top_k_params = [r["params"] for r in top_k]
            narrowed_indep, narrowed_always, narrowed_toggles, toggle_bias = narrow_param_space(
                top_k_params, window=NARROW_WINDOW)

            print(f"\n-- STAGE 2: refined search concentrated around the top {len(top_k)} of stage 1 "
                  f"({n_stage2} trials) --")
            print("  Narrowed ranges (vs. original):")
            for name in narrowed_always:
                if len(narrowed_always[name]) < len(PER_SYSTEM_ALWAYS_SAMPLED[name]):
                    print(f"    {name}: {PER_SYSTEM_ALWAYS_SAMPLED[name]} -> {narrowed_always[name]}")
            for name in narrowed_indep:
                if len(narrowed_indep[name]) < len(INDEPENDENT_PARAM_SPACE[name]):
                    print(f"    {name}: {INDEPENDENT_PARAM_SPACE[name]} -> {narrowed_indep[name]}")
            for toggle, bias in toggle_bias.items():
                if bias not in (0.5,):
                    print(f"    {toggle}: on/off 50/50 -> {bias*100:.0f}% chance ON")

            n_stage2_blind = max(0, int(n_stage2 * STAGE2_BLIND_FRACTION))
            n_stage2_refined = n_stage2 - n_stage2_blind
            print(f"  Splitting stage 2: {n_stage2_refined} refined + {n_stage2_blind} still fully "
                  f"blind (hedge against narrowing excluding a good region stage 1 just didn't find)")

            stage2_params = sample_unique_param_sets(
                rng, n_stage2_refined, independent_space=narrowed_indep, always_sampled=narrowed_always,
                conditional_toggles=narrowed_toggles, toggle_bias=toggle_bias)
            if n_stage2_blind > 0:
                stage2_params += sample_unique_param_sets(rng, n_stage2_blind)
            stage2_results, stage2_failed = _run_trial_batch(stage2_params, raw_data, ath_references,
                                                                n_workers, "stage 2/refined+blind", start_time)
            results.extend(stage2_results)
            n_failed += stage2_failed
        elif n_stage2 > 0:
            print("\n  (stage 1 found no valid trials at all -- skipping stage 2's refinement, "
                  "nothing to narrow around)")
            leftover_params = sample_unique_param_sets(rng, n_stage2)
            leftover_results, leftover_failed = _run_trial_batch(leftover_params, raw_data, ath_references,
                                                                    n_workers, "stage 2/fallback", start_time)
            results.extend(leftover_results)
            n_failed += leftover_failed
    elif SEARCH_METHOD == "optuna":
        print("\nOptuna (TPE) search -- learns from each round of trials where to look next.")

        def _print_progress(done, total, n_valid, n_bad):
            elapsed = time.time() - start_time
            print(f"  [optuna] ...{done}/{total} trials done ({elapsed:.0f}s elapsed, "
                  f"{done / max(elapsed, 1e-9):.1f} trials/s, {n_valid} valid, {n_bad} skipped)")

        results, n_failed = search_optuna(raw_data, ath_references, N_TRIALS, n_workers,
                                            seed=RANDOM_SEED, progress_cb=_print_progress)
    else:
        param_list = sample_unique_param_sets(rng, N_TRIALS)
        results, n_failed = _run_trial_batch(param_list, raw_data, ath_references, n_workers, "search", start_time)

    td.SYSTEMS = original_systems
    for name, value in original_config.items():
        setattr(td, name, value)

    total_time = time.time() - start_time
    print(f"\nCompleted {N_TRIALS} trials in {total_time:.0f}s "
          f"({N_TRIALS/total_time:.1f} trials/s with {n_workers} worker(s)) "
          f"-- {len(results)} valid, {n_failed} skipped (no trades or errored)")
    if stage1_best_score is not None and results:
        final_best_score = max(r[metric_key] for r in results)
        improvement = final_best_score - stage1_best_score
        print(f"Stage 1 alone found best {RANK_METRIC}={stage1_best_score:.3f}; after stage 2's refinement, "
              f"best {RANK_METRIC}={final_best_score:.3f} ({'+' if improvement >= 0 else ''}{improvement:.3f})")

    if not results:
        print("No valid trials completed -- nothing to rank.")
        return

    results.sort(key=lambda r: r[metric_key], reverse=True)

    top_n = min(15, len(results))
    print("\n" + "=" * 78)
    if SCORE_MODE == "portfolio":
        print(f"TOP {top_n} TRIALS (ranked by portfolio {RANK_METRIC} -- ONE shared-capital "
              f"equity curve across every asset, td.simulate_portfolio())")
    else:
        print(f"TOP {top_n} TRIALS (ranked by {SCORE_AGGREGATION}-of-per-asset {RANK_METRIC} -- "
              f"NOT the dollar-summed combined curve, so no single asset can dominate the ranking)")
    print("=" * 78)
    all_asset_names = sorted(set(name for r in results for name in r["per_asset_metric"]))
    rows = []
    for rank, r in enumerate(results[:top_n], start=1):
        score_label = "sharpe" if SCORE_MODE == "portfolio" else f"{SCORE_AGGREGATION}_sharpe"
        row = {"rank": rank, score_label: round(r["sharpe"], 2),
               "CAGR": f"{r['cagr']*100:.2f}%", "MaxDD": f"{r['max_dd']*100:.2f}%",
               "final_equity": f"${r['final_equity']:,.0f}", "trades": r["total_trades"],
               "S1_trades": r["per_system_trades"].get("S1", 0), "S2_trades": r["per_system_trades"].get("S2", 0)}
        if SCORE_MODE == "portfolio":
            for name in all_asset_names:
                row[f"trades_{name}"] = r["per_asset_trades"].get(name, 0)
        else:
            row["worst_asset_sharpe"] = round(min(m["sharpe"] for m in r["per_asset_metric"].values()), 2)
            for name in all_asset_names:
                row[f"sharpe_{name}"] = round(r["per_asset_metric"][name]["sharpe"], 2) if name in r["per_asset_metric"] else np.nan
        row.update(r["params"])
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    best = results[0]
    print("\n" + "=" * 78)
    print("BEST TRIAL -- FULL DETAIL")
    print("=" * 78)
    print("Parameters:")
    for k, v in best["params"].items():
        print(f"  {k} = {v}")
    if SCORE_MODE == "portfolio":
        print(f"\nPortfolio score: Sharpe={best['sharpe']:.2f}, Calmar={best['calmar']:.2f}, "
              f"CAGR={best['cagr']*100:.2f}%")
        print(f"Portfolio max drawdown: {best['max_dd']*100:.2f}%")
        print(f"Portfolio final equity: ${best['final_equity']:,.0f}")
        print("\nTrades per asset (there's no per-asset Sharpe/Calmar in shared-capital mode -- "
              "only one equity curve exists):")
        per_asset_rows = [{"Asset": name, "Trades": best["per_asset_trades"].get(name, 0)}
                            for name in best["per_asset_metric"]]
        print(pd.DataFrame(per_asset_rows).to_string(index=False))
        print("\nTrades per system:", best["per_system_trades"])
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        all_rows = []
        for rank, r in enumerate(results, start=1):
            row = {"rank": rank, "sharpe": r["sharpe"], "calmar": r["calmar"], "cagr": r["cagr"],
                   "max_dd": r["max_dd"], "final_equity": r["final_equity"], "total_trades": r["total_trades"]}
            row.update(r["params"])
            all_rows.append(row)
        csv_path = os.path.join(OUTPUT_DIR, "turtle_optimization_results.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\nFull ranked results ({len(results)} trials) saved to: {csv_path}")
        return
    print(f"\n{SCORE_AGGREGATION.capitalize()}-of-per-asset score: Sharpe={best['sharpe']:.2f}, "
          f"Calmar={best['calmar']:.2f}, CAGR={best['cagr']*100:.2f}%")
    print(f"Worst single-asset drawdown: {best['max_dd']*100:.2f}%")
    print(f"Combined portfolio final equity (reporting only, NOT used for ranking): ${best['final_equity']:,.0f}")
    print("\nPer-asset breakdown (this is what actually got scored):")
    per_asset_rows = []
    for name, m in best["per_asset_metric"].items():
        per_asset_rows.append({
            "Asset": name, "Sharpe": round(m["sharpe"], 2), "Calmar": round(m["calmar"], 2),
            "CAGR": f"{m['cagr']*100:.2f}%", "Total Return": f"{m['total_return']*100:.2f}%",
            "Trades": best["per_asset_trades"].get(name, 0),
        })
    print(pd.DataFrame(per_asset_rows).to_string(index=False))
    print("\nTrades per system:", best["per_system_trades"])

    print("\nTo use this configuration, set these in turtle_dual_system.py:")
    bp = best["params"]
    recommended_systems = params_to_systems(bp)
    print(f"  SYSTEMS = [")
    for s in recommended_systems:
        print(f"      {{")
        for k, v in s.items():
            print(f"          {k!r}: {v!r},")
        print(f"      }},")
    print(f"  ]")
    shared_keys = set(INDEPENDENT_PARAM_SPACE)
    shared_keys -= {"S1_BREAKOUT_LOOKBACK", "S1_EXIT_LOOKBACK", "S2_BREAKOUT_LOOKBACK", "S2_EXIT_LOOKBACK"}
    shared_keys |= set(GLOBAL_CONDITIONAL_TOGGLES)
    shared_keys |= {sub for subs in GLOBAL_CONDITIONAL_TOGGLES.values() for sub in subs}
    print("  # shared/portfolio-level (both systems combined):")
    for k, v in bp.items():
        if k in shared_keys:
            print(f"  {k} = {v!r}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    for rank, r in enumerate(results, start=1):
        row = {"rank": rank, "sharpe": r["sharpe"], "calmar": r["calmar"], "cagr": r["cagr"],
               "max_dd": r["max_dd"], "final_equity": r["final_equity"], "total_trades": r["total_trades"]}
        for name in all_asset_names:
            row[f"sharpe_{name}"] = r["per_asset_metric"][name]["sharpe"] if name in r["per_asset_metric"] else np.nan
        row.update(r["params"])
        all_rows.append(row)
    csv_path = os.path.join(OUTPUT_DIR, "turtle_optimization_results.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nFull ranked results ({len(results)} trials) saved to: {csv_path}")


if __name__ == "__main__":
    main()