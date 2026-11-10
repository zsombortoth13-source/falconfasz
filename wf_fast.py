"""Speed-ups for walk_forward_turtle.py that leave the strategy engine's results unchanged.

FeatureCache
    td.build_features() recomputes ~13 indicators per system per trial even though every trial
    in a window runs on the SAME price data and each indicator parameter comes from a small
    discrete set (lookbacks, ATR periods, ...). The cache computes each (asset, indicator,
    period) once and reuses it, then assembles the same DataFrame td.build_features() would
    have built. Output is bit-identical (checked with assert_frame_equal in the test script).

suggest_params
    An Optuna "define-by-run" version of optimize_turtle_dual_system.sample_params_raw(): the
    same search space, same conditional sub-parameters (only suggested when their on/off
    toggle is on), same only_optimize/fixed_values pinning rules.
"""
import numpy as np
import pandas as pd

import turtle_dual_system as td
import optimize_turtle_dual_system as opt


class FeatureCache:
    """Per-window indicator cache. One instance per (price-data slice); build(name) reads
    td.SYSTEMS at call time, so it follows whatever opt.apply_params() just set."""

    def __init__(self, raw_data: dict, ath_references: dict = None):
        self._raw = raw_data
        self._ath = ath_references or {}
        self._memo = {}

    def _get(self, name: str, key: tuple, compute):
        k = (name,) + key
        hit = self._memo.get(k)
        if hit is None:
            hit = compute()
            self._memo[k] = hit
        return hit

    def build(self, name: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """df is accepted (and ignored) so this can be passed directly as features_fn(name, df);
        the cache always reads the slice it was constructed with."""
        df = self._raw[name]
        out = df.copy()
        out.attrs["asset"] = name
        close, low = df["Close"], df["Low"]
        for sysdef in td.SYSTEMS:
            s = sysdef["name"]
            bl, el = sysdef["breakout_lookback"], sysdef["exit_lookback"]
            pl, sw = sysdef["pyramid_breakout_lookback"], sysdef["swing_lookback"]
            tl, na, ta = sysdef["trailing_stop_lookback_days"], sysdef["n_atr_period"], sysdef["tp_atr_period"]
            out[f"breakout_level_{s}"] = self._get(name, ("brk", bl), lambda: close.rolling(bl).max().shift(1))
            out[f"exit_low_level_{s}"] = self._get(name, ("exl", el), lambda: low.rolling(el).min().shift(1))
            if not sysdef.get("use_channel_exit", True):
                out[f"exit_low_level_{s}"] = np.nan          # mirrors td.build_features: no channel exit
            out[f"pyramid_breakout_level_{s}"] = self._get(
                name, ("pyb", pl), lambda: close.rolling(pl).max().shift(1))
            out[f"swing_high_level_{s}"] = self._get(
                name, ("swg", sw), lambda: td.compute_confirmed_swing_high_level(df, sw))
            out[f"trailing_stop_low_{s}"] = self._get(name, ("trl", tl), lambda: low.rolling(tl).min().shift(1))
            out[f"n_{s}"] = self._get(name, ("atr", na), lambda: td.compute_atr(df, na))
            out[f"tp_n_{s}"] = self._get(name, ("atr", ta), lambda: td.compute_atr(df, ta))
            ap, ep, cp = sysdef["adx_period"], sysdef["ema_period"], sysdef["choppiness_period"]
            out[f"adx_{s}"] = self._get(name, ("adx", ap), lambda: td.compute_adx(df, ap))
            out[f"ema_trend_{s}"] = self._get(name, ("ema", ep), lambda: close.ewm(span=ep, adjust=False).mean())
            out[f"choppiness_{s}"] = self._get(name, ("chp", cp), lambda: td.compute_choppiness_index(df, cp))
            if sysdef.get("use_ma_regime"):
                mp, mb = sysdef["ma_regime_period"], sysdef["ma_regime_band_pct"]
                out[f"breakout_level_{s}"], out[f"exit_low_level_{s}"] = self._get(
                    name, ("mar", mp, mb), lambda: td.ma_regime_levels(close, mp, mb))
            ema_levels = td.entry_ema_levels(close, s, sysdef)   # the ENTRY_EMA_* switch (None when off); not memoised, it is cheap
            if ema_levels is not None:
                if ema_levels[0] is not None:
                    out[f"breakout_level_{s}"] = ema_levels[0]
                if ema_levels[1] is not None:
                    out[f"exit_low_level_{s}"] = ema_levels[1]
            mh = td.monthly_high_settings(s, sysdef)    # the monthly-high breakout entry; the month table is cached per market
            if mh is not None:
                months = self._get(name, ("mhm",), lambda: td.monthly_high_months(df.index, df["High"]))
                out[f"breakout_level_{s}"] = td.monthly_high_entry_level(close, months, out[f"n_{s}"], sysdef["entry_buffer_n_mult"], mh[0], mh[1], mh[2], mh[3])
            mha = td.monthly_high_add_settings(sysdef)    # pyramid_method "monthly_high": the same event, used to add while in a position
            if mha is not None:
                months_a = self._get(name, ("mhm",), lambda: td.monthly_high_months(df.index, df["High"]))
                out[f"mh_add_level_{s}"] = td.monthly_high_entry_level(close, months_a, out[f"n_{s}"], sysdef["entry_buffer_n_mult"], mha[0], mha[1], mha[2], mha[3])
            out[f"all_time_high_{s}"] = self._get(name, ("ath",), lambda: self._all_time_high(name, df))
        return out

    def _all_time_high(self, name: str, df: pd.DataFrame) -> pd.Series:
        full_history_ath = self._ath.get(name)
        if full_history_ath is not None:
            reindexed = full_history_ath.reindex(df.index.union(full_history_ath.index)).ffill()
            reindexed = reindexed.reindex(df.index)
            combined = pd.concat([reindexed, df["High"].cummax()], axis=1).max(axis=1)
            return combined.shift(1)
        return df["High"].cummax().shift(1)


def _is_ordered_numeric(values: list) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _pick(trial, name: str, values: list, only_optimize, fixed_values: dict):
    """Pinned parameters come from fixed_values (KeyError if missing, same as the random
    sampler). Ordered numeric lists are suggested as an integer INDEX so TPE can exploit the
    ordering; anything else (strings, None, tuples) as an unordered categorical index."""
    if only_optimize is not None and name not in only_optimize:
        return fixed_values[name]
    if len(values) == 1:
        return values[0]
    if _is_ordered_numeric(values):
        return values[trial.suggest_int(name, 0, len(values) - 1)]
    return values[trial.suggest_categorical(name, list(range(len(values))))]


def _toggle(trial, name: str, only_optimize, fixed_values: dict) -> bool:
    if only_optimize is not None and name not in only_optimize:
        return fixed_values[name]
    return trial.suggest_categorical(name, [False, True])


def suggest_params(trial, only_optimize: set = None, fixed_values: dict = None) -> dict:
    """Optuna counterpart of opt.sample_params_raw() over the module-level search spaces."""
    fixed_values = fixed_values or {}
    params = {n: _pick(trial, n, v, only_optimize, fixed_values) for n, v in opt.INDEPENDENT_PARAM_SPACE.items()}
    for n, v in opt.PER_SYSTEM_ALWAYS_SAMPLED.items():
        params[n] = _pick(trial, n, v, only_optimize, fixed_values)
    for spaces in (opt.PER_SYSTEM_CONDITIONAL_TOGGLES, opt.GLOBAL_CONDITIONAL_TOGGLES):
        for toggle_name, sub_params in spaces.items():
            is_on = _toggle(trial, toggle_name, only_optimize, fixed_values)
            params[toggle_name] = is_on
            if is_on:
                for sub_name, sub_values in sub_params.items():
                    params[sub_name] = _pick(trial, sub_name, sub_values, only_optimize, fixed_values)
    for market, values in opt.ASSET_RISK_SPACE.items():
        name = td.ASSET_RISK_PARAM_PREFIX + market
        params[name] = _pick(trial, name, values, only_optimize, fixed_values)
    return params


def params_signature(params: dict) -> frozenset:
    return frozenset(params.items())
