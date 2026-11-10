"""Monte Carlo robustness tests for the Turtle backtests.

All of these answer one question from different angles: how much of a backtest's headline result
is luck (trade order, which trades happened, a particular parameter value) rather than edge?

simulate_trades      replays a run's own trade returns many times -- reshuffled order, bootstrapped
                     (resampled with replacement), or with a share of trades randomly skipped
bootstrap_mean       confidence interval for an average (e.g. mean out-of-sample Sharpe over windows)
neighbor_params      parameter sets one step away from a chosen one, for a "does a small nudge to the
                     parameters wreck the result?" test

Returns, not dollars: each trade is its P&L as a fraction of the equity available when IT opened,
and paths compound those fractions. Position size scales with equity at entry, so a reshuffled
dollar amount would be applied to a different equity level than it was ever sized for; compounding
fractions keeps every simulated path internally consistent and can never go below zero.

Approximation to keep in mind: trades that overlapped in time are replayed one after another, so
absolute drawdown figures are indicative, and the comparison between actual and simulated is the
useful part.
"""
import numpy as np

METHODS = {
    "shuffle": "Reshuffle the trade order",
    "bootstrap": "Resample trades with replacement",
    "skip": "Randomly skip some trades",
}


def entry_equity(trade: dict, equity_curve=None):
    """Equity available when a trade/episode first opened. Dual System trades carry it on their
    first pyramid unit; otherwise it is looked up from the equity curve (last value at or before
    the entry date)."""
    units = trade.get("unit_details")
    if units:
        return units[0].get("equity_at_entry")
    if equity_curve is None or len(equity_curve) == 0:
        return None
    entry_date = trade.get("entry_date")
    if entry_date is None and trade.get("entry_dates"):
        entry_date = min(trade["entry_dates"])
    if entry_date is None:
        return None
    try:
        return float(equity_curve.asof(entry_date))
    except Exception:
        return None


def trade_returns(trades: list, equity_curve=None) -> np.ndarray:
    """Each trade's P&L as a fraction of the equity at its entry (trades without a usable
    reference are left out)."""
    out = []
    for t in trades:
        eq = entry_equity(t, equity_curve)
        if eq and eq > 0:
            out.append(t["pnl"] / eq)
    return np.array(out, dtype=float)


def _path_stats(returns: np.ndarray) -> tuple:
    """(max drawdown, final multiple of starting capital) for one or many paths (rows).
    Floored at zero per step: a loss beyond 100% of a trade's entry equity is total ruin, not a
    negative balance."""
    r = np.atleast_2d(returns)
    equity = np.cumprod(np.maximum(1.0 + r, 0.0), axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    safe_max = np.where(running_max > 0, running_max, 1.0)
    max_dd = ((equity - running_max) / safe_max).min(axis=1)
    return max_dd, equity[:, -1]


def simulate_trades(returns: np.ndarray, method: str = "shuffle", n_sims: int = 1000,
                    seed: int = 42, skip_fraction: float = 0.1, years: float = None):
    """Replays `returns` n_sims times. Returns None with fewer than 10 trades, else a dict:
      max_dd, final_multiple (arrays, one per simulation), cagr (if years given),
      actual_max_dd, actual_final_multiple, actual_max_dd_percentile (where the real chronological
      result ranks among the simulations: 100 = worse than every one), method, n_trades."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 10:
        return None
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    rng = np.random.default_rng(seed)
    if method == "shuffle":
        paths = np.array([rng.permutation(returns) for _ in range(n_sims)])
    elif method == "bootstrap":
        paths = rng.choice(returns, size=(n_sims, n), replace=True)
    else:
        keep = rng.random((n_sims, n)) >= skip_fraction      # skipped trade = a 0% return
        paths = returns[None, :] * keep
    max_dd, final = _path_stats(paths)
    actual_dd, actual_final = (x[0] for x in _path_stats(returns))
    out = {
        "max_dd": max_dd, "final_multiple": final, "actual_max_dd": float(actual_dd),
        "actual_final_multiple": float(actual_final),
        "actual_max_dd_percentile": float(100 * (max_dd <= actual_dd).mean()),
        "method": method, "n_trades": n,
    }
    if years and years > 0:
        out["cagr"] = np.where(final > 0, final, np.nan) ** (1.0 / years) - 1.0
    return out


SERIES_METHODS = {
    "shuffle": "Reshuffle blocks of days",
    "bootstrap": "Resample blocks of days with replacement",
    "skip": "Miss random stretches of days",
}


def equity_returns(equity_curve) -> np.ndarray:
    """Daily returns of an equity curve. For a PORTFOLIO this is the right input to simulate from:
    the curve already contains overlapping positions and the swings while trades are still open,
    which replaying closed trades one after another cannot reproduce."""
    r = equity_curve.dropna().pct_change().dropna().to_numpy(dtype=float)
    return np.where(np.isfinite(r), np.maximum(r, -1.0), 0.0)


def _longest_underwater(returns: np.ndarray) -> np.ndarray:
    """Longest stretch (in steps) spent below a previous equity peak, for one or many paths."""
    r = np.atleast_2d(returns)
    equity = np.cumprod(np.maximum(1.0 + r, 0.0), axis=1)
    under = equity < np.maximum.accumulate(equity, axis=1)
    out = np.zeros(len(r), dtype=int)
    for i, row in enumerate(under):
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        if len(edges):
            out[i] = (edges[1::2] - edges[0::2]).max()
    return out


def drift_standard_error(returns: np.ndarray, window: int = 252) -> float:
    """Standard error of the average daily return, from the spread of ~yearly averages. Uses long
    windows on purpose: a strategy's good and bad YEARS are what make its true average uncertain,
    and day-to-day noise averages out of that."""
    nb = len(returns) // window
    if nb < 3:
        return 0.0
    means = np.asarray(returns[:nb * window]).reshape(nb, window).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(nb))


# How many standard errors of the average return are folded into bootstrap / skip simulations (see
# simulate_series). Calibrated on the first-half -> second-half test in the project's study.
DEFAULT_DRIFT_UNCERTAINTY = 2.0


def simulate_series(returns: np.ndarray, method: str = "shuffle", n_sims: int = 1000, seed: int = 42,
                    block: int = 126, skip_fraction: float = 0.1, years: float = None,
                    drift_uncertainty: float = 0.0):
    """Block Monte Carlo on a daily-return series (blocks keep volatility clustering and streaks
    intact). Same result shape as simulate_trades, plus the longest time under water:
      shuffle    random order of the blocks -- the final result is the same as the actual one (order
                 does not change the product), only the drawdown spreads
      bootstrap  blocks drawn with replacement
      skip       the original order with a share of blocks replaced by flat (0%) stretches
    drift_uncertainty (bootstrap and skip only): each simulated path also gets a random shift in
    its average daily return, drawn with a spread of `drift_uncertainty` standard errors of the
    observed average (see drift_standard_error). Without it every path assumes the sample's
    average return is the true one, which understates how different the future can be (the range
    of final results came out too narrow when tested against held-out history)."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 60:
        return None
    if method not in SERIES_METHODS:
        raise ValueError(f"unknown method {method!r}")
    block = max(1, int(block))
    rng = np.random.default_rng(seed)
    blocks = [returns[i:i + block] for i in range(0, n, block)]
    nb = len(blocks)
    shift = np.zeros(n_sims)
    if drift_uncertainty > 0 and method != "shuffle":
        shift = rng.normal(0.0, drift_uncertainty * drift_standard_error(returns), size=n_sims)
    paths = np.empty((n_sims, n))
    for s in range(n_sims):
        if method == "shuffle":
            seq = np.concatenate([blocks[i] for i in rng.permutation(nb)])
        elif method == "bootstrap":
            seq = np.concatenate([blocks[i] for i in rng.integers(0, nb, size=nb)])
            seq = np.resize(seq, n)
        else:
            keep = rng.random(nb) >= skip_fraction
            seq = np.concatenate([b if k else np.zeros_like(b) for b, k in zip(blocks, keep)])
        paths[s] = seq + shift[s]
    max_dd, final = _path_stats(paths)
    actual_dd, actual_final = (x[0] for x in _path_stats(returns))
    out = {
        "max_dd": max_dd, "final_multiple": final, "actual_max_dd": float(actual_dd),
        "actual_final_multiple": float(actual_final),
        "actual_max_dd_percentile": float(100 * (max_dd <= actual_dd).mean()),
        "method": method, "n_trades": n,
        "underwater_days": _longest_underwater(paths), "actual_underwater_days": int(_longest_underwater(returns)[0]),
    }
    if years and years > 0:
        out["cagr"] = np.where(final > 0, final, np.nan) ** (1.0 / years) - 1.0
    return out


def summarize(mc: dict, dd_threshold: float = -0.5) -> dict:
    """Percentiles and probabilities from a simulate_trades result."""
    dd5, dd50, dd95 = np.percentile(mc["max_dd"], [5, 50, 95])
    f5, f50, f95 = np.percentile(mc["final_multiple"], [5, 50, 95])
    return {
        "dd_p5": dd5, "dd_p50": dd50, "dd_p95": dd95,
        "final_p5": f5, "final_p50": f50, "final_p95": f95,
        "p_loss": float((mc["final_multiple"] < 1.0).mean()),
        "p_dd_worse": float((mc["max_dd"] <= dd_threshold).mean()),
        "p_ruin": float((mc["final_multiple"] <= 0.1).mean()),
    }


def bootstrap_mean(values, n_sims: int = 5000, seed: int = 42, block: int = 1):
    """Bootstrap distribution of the mean of `values` (kept in their given, time, order).
    block > 1 resamples runs of `block` consecutive values (circular), which respects the fact that
    neighbouring walk-forward windows are not independent -- good and bad stretches of the market
    run across several windows -- and gives honestly wider intervals than resampling single values.
    Returns None for fewer than 3 values, else {mean, p5, p95, p_le_zero, n}: the 5th/95th
    percentile of the resampled mean and the share of resamples whose mean is at or below zero."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(arr)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), n))
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = arr[idx.reshape(n_sims, -1)[:, :n]].mean(axis=1)
    return {"mean": float(arr.mean()), "p5": float(np.percentile(means, 5)),
            "p95": float(np.percentile(means, 95)), "p_le_zero": float((means <= 0).mean()),
            "n": n}


def deflated_sharpe(sharpe: float, trial_sharpes, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                    periods_per_year: int = 252):
    """Probability that a chosen strategy's Sharpe ratio is genuinely above what the BEST of many
    luck-only tries would have shown (Bailey & Lopez de Prado's deflated Sharpe ratio).
    sharpe: the chosen trial's annualized Sharpe; trial_sharpes: annualized Sharpes of every trial
    that was tried (their spread sets how lucky the best one is expected to be); n_obs: number of
    return observations behind the Sharpe; skew/kurt: of those returns (normal if unknown).
    Returns (probability, luck_benchmark_sharpe_annualized). Assumes the tries were independent;
    a guided search such as Optuna concentrates its tries, so treat the result as optimistic."""
    from scipy.stats import norm
    s = np.asarray([x for x in trial_sharpes if np.isfinite(x)], dtype=float)
    n_trials = len(s)
    if n_trials < 2 or n_obs < 30:
        return None
    root_ppy = np.sqrt(periods_per_year)
    sd = (s / root_ppy).std(ddof=1)
    euler = 0.5772156649
    benchmark = sd * ((1 - euler) * norm.ppf(1 - 1.0 / n_trials) + euler * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr = sharpe / root_ppy
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-9))
    z = (sr - benchmark) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z)), float(benchmark * root_ppy)


def candidate_lists(space: dict) -> dict:
    """{parameter name: list of candidate values} from an optimize_turtle_dual_system search space."""
    found = {}
    for key in ("independent", "always"):
        found.update(space[key])
    for key in ("conditional", "global"):
        for subs in space[key].values():
            found.update(subs)
    for market, values in space.get("asset_risk", {}).items():
        found["RISKMULT::" + market] = values   # = td.ASSET_RISK_PARAM_PREFIX (monte_carlo does not import the engine)
    return found


def neighbor_params(params: dict, space: dict, searched: set, n: int, seed: int = 42,
                    move_probability: float = 0.5, is_valid=None, max_attempts_factor: int = 30) -> list:
    """Up to n distinct parameter sets that differ from `params` by moving some of the SEARCHED
    numeric parameters one step along their candidate list (a lookback of 20 in [10, 15, 20, 30]
    becomes 15 or 30). Toggles, categorical choices and pinned parameters are left alone, so only
    the dimensions the search actually tuned are nudged."""
    rng = np.random.default_rng(seed)
    candidates = candidate_lists(space)
    movable = []
    for name in searched:
        values = candidates.get(name)
        if name not in params or not values or len(values) < 2:
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        ordered = sorted(values)
        if params[name] in ordered:
            movable.append((name, ordered))
    if not movable:
        return []
    def signature(p):
        return repr(sorted(p.items(), key=lambda kv: kv[0]))

    seen = {signature(params)}
    out = []
    attempts = 0
    while len(out) < n and attempts < n * max_attempts_factor:
        attempts += 1
        variant = dict(params)
        for name, ordered in movable:
            if rng.random() < move_probability:
                idx = ordered.index(params[name]) + int(rng.choice([-1, 1]))
                variant[name] = ordered[min(max(idx, 0), len(ordered) - 1)]
        sig = signature(variant)
        if sig in seen:
            continue
        if is_valid is not None and not is_valid(variant):
            continue
        seen.add(sig)
        out.append(variant)
    return out
