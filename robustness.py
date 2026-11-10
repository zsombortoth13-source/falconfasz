"""Parameter-nudge robustness test for one backtest configuration.

Takes the S1/S2 settings of a backtest that was actually run, makes variations where each tuned
numeric parameter either stays or moves one step along its candidate list (a 20-day lookback in
[10, 15, 20, 30] becomes 15 or 30), re-runs the shared-portfolio backtest for every variation in
parallel, and reports how the results compare with the unmodified baseline (scored the same way).
S1 and S2 are nudged independently, so a configuration where they differ is reproduced exactly.
"""
import copy
import multiprocessing as mp

import numpy as np

import turtle_dual_system as td
import monte_carlo

# td globals that make up a backtest's environment: re-applied in every worker process, because
# spawned processes re-import the engine with the file's defaults
TD_CONFIG_NAMES = (
    "MAX_TOTAL_EXPOSURE_PCT", "STARTING_CAPITAL_PER_ASSET", "PORTFOLIO_STARTING_CAPITAL",
    "USE_PORTFOLIO_RISK_CAP", "MAX_PORTFOLIO_RISK_PCT", "USE_DRAWDOWN_SCALING",
    "DRAWDOWN_SCALING_THRESHOLD", "DRAWDOWN_SCALING_FACTOR", "DRAWDOWN_SCALING_REFERENCE", "SLIPPAGE_PCT", "SPREAD_PCT",
    "COMMISSION_PCT", "HOLDING_COST_PCT_PER_YEAR", "USE_PER_CONTRACT_FEES", "FEE_SCHEDULE",
    "MARK_TO_MARKET", "SIZE_STEP", "TAKE_MIN_ONE_UNIT", "USE_MARGIN_CALLS", "MARGIN_SPEC",
    "DEFAULT_MARGIN_RATES", "ASSET_COSTS", "EXECUTION_TIMING", "STOP_PRIORITY", "ASSET_PRIORITY",
    "MAX_GROSS_EXPOSURE_PCT", "VOL_TARGET_PCT", "VOL_TARGET_WINDOW", "VOL_TARGET_FLOOR", "VOL_TARGET_CAP",
    "HITRATE_WINDOW", "HITRATE_MIN_AVG_R", "HITRATE_SCALE",
    "ATH_RISK_MULT", "BELOW_ATH_RISK_MULT", "ATH_FIRST_UNIT_ONLY", "ASSET_RISK_MULT", "ASSET_RISK_NORMALIZE", "CHOP_RISK_MULT", "CHOP_TREND_MAX", "ATH_MAX_DISTANCE_PCT",
    "ALLOW_EXIT_ABOVE_ENTRY", "CARRY_MODE", "CARRY_RATE_SERIES", "CARRY_RATE_FLAT", "CARRY_DIVIDEND_YIELD",
    "STOP_SLIPPAGE_PCT",
    "TAIL_WINDOW", "TAIL_MIN_BARS", "TAIL_FALLBACK", "GAP_SIZING_K", "STRESS_TOTAL_CAP_PCT", "STRESS_SINGLE_CAP_PCT",
    "STRESS_CLUSTER_CAP_PCT", "MARKET_CLUSTERS", "BOOK_VOL_CAP_PCT", "BOOK_VOL_WINDOW", "EQUITY_MA_WINDOW", "EQUITY_MA_SCALE",
    "SHOCK_PAUSE_LOSS_PCT", "SHOCK_PAUSE_DAYS", "EXTENSION_MA", "EXTENSION_N_MULT", "EXTENSION_RISK_MULT",
    "DAILY_LOSS_STOP_N_MULT", "DAILY_LOSS_STOP_PCT", "ENTRY_TOP_K", "ENTRY_STRENGTH_LOOKBACK", "ENTRY_MIN_STRENGTH",
    "ENTRY_RANK_FIRST_UNIT_ONLY", "VOL_RISK_GAMMA", "VOL_RISK_ATR", "VOL_RISK_REF", "VOL_RISK_MIN", "VOL_RISK_MAX",
    "ENTRY_EMA_PERIOD", "ENTRY_EMA_MODE", "ENTRY_EMA_EXIT", "MH_CLEAN_MODE",
)
METRICS = ("Sharpe", "CAGR", "Calmar", "Max Drawdown")


def capture_td_config() -> dict:
    return {name: copy.deepcopy(getattr(td, name)) for name in TD_CONFIG_NAMES}


def _valid(systems: list) -> bool:
    for s in systems:
        if not td.ALLOW_EXIT_ABOVE_ENTRY and s["exit_lookback"] >= s["breakout_lookback"]:
            return False
    if len(systems) == 2 and systems[0]["breakout_lookback"] >= systems[1]["breakout_lookback"]:
        return False
    return True


def _signature(variant: dict) -> str:
    return repr(([sorted(s.items(), key=lambda kv: kv[0]) for s in variant["systems"]],
                 sorted(variant["td"].items(), key=lambda kv: kv[0])))


def neighbor_variants(systems: list, td_config: dict, space: dict, searched: set, n: int,
                      seed: int = 42, move_probability: float = 0.5) -> tuple:
    """(variants, nudged): up to n distinct variations of (systems, MAX_TOTAL_EXPOSURE_PCT) that
    differ from the given configuration by one-step moves of the searched numeric parameters, and
    the names of the parameters that can be nudged. Each variant is
    {"systems": [...], "td": {"MAX_TOTAL_EXPOSURE_PCT": ...}}. A parameter is only nudged when its
    current value is one of the candidate values in its range."""
    rng = np.random.default_rng(seed)
    candidates = monte_carlo.candidate_lists(space)

    def ordered_list(name, value):
        values = candidates.get(name)
        if not values or len(values) < 2 or name not in searched:
            return None
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return None
        ordered = sorted(values)
        return ordered if value in ordered else None

    dims = []   # (system index or None, key, candidate name, ordered values)
    for si, s in enumerate(systems):
        for key, value in s.items():
            if key == "name":
                continue
            cand = {"breakout_lookback": f"{s['name']}_BREAKOUT_LOOKBACK",
                    "exit_lookback": f"{s['name']}_EXIT_LOOKBACK"}.get(key, key)
            ordered = ordered_list(cand, value)
            if ordered:
                dims.append((si, key, ordered))
    exposure = td_config["MAX_TOTAL_EXPOSURE_PCT"]
    ordered = ordered_list("MAX_TOTAL_EXPOSURE_PCT", exposure)
    if ordered:
        dims.append((None, "MAX_TOTAL_EXPOSURE_PCT", ordered))
    nudged = [f"{systems[si]['name']} {key}" if si is not None else key for si, key, _ in dims]
    if not dims:
        return [], nudged

    base = {"systems": systems, "td": {"MAX_TOTAL_EXPOSURE_PCT": exposure}}
    seen = {_signature(base)}
    out, attempts = [], 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        variant = {"systems": copy.deepcopy(systems), "td": {"MAX_TOTAL_EXPOSURE_PCT": exposure}}
        for si, key, ordered in dims:
            if rng.random() >= move_probability:
                continue
            current = systems[si][key] if si is not None else exposure
            idx = min(max(ordered.index(current) + int(rng.choice([-1, 1])), 0), len(ordered) - 1)
            if si is None:
                variant["td"][key] = ordered[idx]
            else:
                variant["systems"][si][key] = ordered[idx]
        sig = _signature(variant)
        if sig in seen or not _valid(variant["systems"]):
            continue
        seen.add(sig)
        out.append(variant)
    return out, nudged


_W = {}


def _init_worker(raw_data, ath_references, td_config):
    import wf_fast
    for name, value in td_config.items():
        setattr(td, name, value)
    _W.update(raw=raw_data, cache=wf_fast.FeatureCache(raw_data, ath_references))


def _run_variant(variant: dict):
    td.SYSTEMS = variant["systems"]
    for name, value in variant["td"].items():
        setattr(td, name, value)
    try:
        features = {name: _W["cache"].build(name) for name in _W["raw"]}
        result = td.simulate_portfolio(features)
        stats = td.performance_stats(result["equity_curve"], result["starting_capital"])
    except Exception:
        return None
    return {m: float(stats[m]) for m in METRICS}


def evaluate(variants: list, raw_data: dict, ath_references: dict, td_config: dict, n_workers: int) -> list:
    """Scores each variant (a result dict of METRICS, or None if it failed), in the given order."""
    if n_workers > 1 and len(variants) > 1:
        with mp.Pool(processes=n_workers, initializer=_init_worker,
                     initargs=(raw_data, ath_references, td_config)) as pool:
            return pool.map(_run_variant, variants, chunksize=1)
    saved_systems = td.SYSTEMS
    saved = {name: getattr(td, name) for name in td_config}
    try:
        _init_worker(raw_data, ath_references, td_config)
        return [_run_variant(v) for v in variants]
    finally:
        td.SYSTEMS = saved_systems
        for name, value in saved.items():
            setattr(td, name, value)
