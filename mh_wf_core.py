"""Core of the lean Walk-forward app for the monthly-high strategy (mh_walkforward_ui.py).

Everything that is not drawing lives here, and this module is imported ONCE per server process (Streamlit re-runs the app script on every click but not imported modules), which is what makes
the app fast and lets a run keep going in a background thread while the page stays usable:
  * a price cache on disk, per market (a walk-forward run downloaded all 30 years of every market from Yahoo each time before),
  * the settings of the clean monthly-high strategy and how they become a walk-forward search (same engine mode as the Strategy Lab's "Monthly high" tab: turtle_dual_system.MH_CLEAN_MODE),
  * one background run at a time (RUN), with live progress parsed from walk_forward_turtle's own output,
  * the summary of a finished run (stitched out-of-sample curve and its statistics, per-window table, chosen parameters, equal-weight buy & hold for reference).
The strategy: INITIAL ENTRY = close above its EMA and the first close above a monthly high (plus entry buffer x ATR) that is min..max days old; PYRAMIDING = one more unit on every further
such signal; SL = ATR period x multiplier under the newest unit (every add re-sets it); TP = every unit closes at its entry + multiplier x ATR(TP period). Nothing else."""
import contextlib
import copy
import hashlib
import io
import os
import pickle
import re
import threading
import time
import traceback

import numpy as np
import pandas as pd

import brokers
import ibkr_fees
import optimize_turtle_dual_system as opt
import turtle_dual_system as td
import walk_forward_turtle as wf

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "cache_prices")
OUTPUT_DIR = os.path.join(ROOT, "output_mh_wf")          # its own folder: the Strategy Lab's walk-forward writes to output/
CACHE_HOURS = 12

ALL_TICKERS = dict(td.TICKERS)                           # the engine's market list as it was when this module was first imported
ENGINE_COSTS = ibkr_fees.remember_engine_defaults(td)    # the engine file's own cost settings (the "flat %" model)
_ORIGINAL_DOWNLOAD = td.download_data

# ---------------------------------------------------------------------------------------------------------------------------------- settings of the strategy
# (key, label, group, default, candidate values to search, lowest value accepted). Defaults = the numbers in the TradingView script.
PARAMS = [
    ("mh_sma_period", "Price must be above this EMA (days)", "Initial entry and pyramiding", 200, "100, 150, 200, 250, 300", 1),
    ("mh_min_age", "Monthly high at least this many days old", "Initial entry and pyramiding", 20, "10, 15, 20, 30, 40, 60", 1),
    ("mh_max_age", "Monthly high at most this many days old", "Initial entry and pyramiding", 365, "120, 180, 250, 365, 500", 2),
    ("entry_buffer_n_mult", "Entry buffer: the close must be this many ATR above the monthly high", "Initial entry and pyramiding", 0.0, "0, 0.25, 0.5, 1, 1.5, 2", 0.0),
    ("n_atr_period", "SL ATR period (days)", "Stop loss", 185, "14, 20, 40, 60, 100, 185", 2),
    ("stop_n_mult", "SL ATR multiplier", "Stop loss", 6.0, "1.5, 2, 3, 4, 6, 8", 0.5),
    ("tp_atr_period", "TP ATR period (days)", "Take profit", 185, "14, 20, 40, 100, 185", 2),
    ("tp_n_mult", "TP ATR multiplier", "Take profit", 30.0, "3, 6, 10, 20, 30", 0.25),
]
PARAM_NAMES = {"mh_sma_period": "EMA", "mh_min_age": "min age", "mh_max_age": "max age", "entry_buffer_n_mult": "buffer", "n_atr_period": "SL ATR",
               "stop_n_mult": "SL x", "tp_atr_period": "TP ATR", "tp_n_mult": "TP x"}
DEFAULTS = {key: default for key, _, _, default, _, _ in PARAMS}
RISK_DEFAULT = 1.0
UNIT_CAP = 999                                          # no cap of its own: the exposure and open-risk limits still apply

COST_NONE = "No costs"
COST_FLAT = "Flat % costs (the engine's own, fractional sizes)"
COST_IBKR = "Interactive Brokers futures (per-contract fees, micro contracts)"


def parse_candidates(text: str, default) -> list:
    """'10, 15, 20' -> [10, 15, 20]; whole numbers stay whole for the integer settings, sorted and de-duplicated. Raises ValueError with a readable reason."""
    tokens = [t for t in re.split(r"[,\s;]+", str(text).strip()) if t]
    if not tokens:
        raise ValueError("enter at least one value")
    whole = isinstance(default, int) and not isinstance(default, bool)
    out = []
    for token in tokens:
        x = float(token)
        if not np.isfinite(x) or x < 0:
            raise ValueError(f"'{token}' is not a valid number")
        if whole and x != int(x):
            raise ValueError(f"'{token}' must be a whole number")
        out.append(int(x) if whole else x)
    return sorted(set(out))


def build_search(values: dict, searched: dict, candidates: dict, exposure_pct: float):
    """(space, only_optimize, fixed_values, errors): the walk-forward search for the clean strategy. A ticked setting is drawn from its candidate list; an unticked one is held at its value."""
    space = {"independent": {"MAX_TOTAL_EXPOSURE_PCT": [int(exposure_pct)]}, "always": {}, "conditional": {"use_monthly_high_entry": {}}, "global": {}, "asset_risk": {}}
    errors, only = [], set()
    for key, label, _, default, default_text, _ in PARAMS:
        cands = [values[key]]
        if searched.get(key, True):
            only.add(key)
            try:
                cands = parse_candidates(candidates.get(key, default_text), default)
            except ValueError as e:
                errors.append(f"{label}: {e}")
                cands = parse_candidates(default_text, default)
        (space["conditional"]["use_monthly_high_entry"] if key.startswith("mh_") else space["always"])[key] = cands
    space["always"]["risk_per_trade_pct"] = [values["risk_per_trade_pct"]]
    space["always"]["max_units_per_trend"] = [UNIT_CAP]
    fixed = {**{k: v for k, v in values.items()}, "max_units_per_trend": UNIT_CAP, "use_monthly_high_entry": True, "MAX_TOTAL_EXPOSURE_PCT": int(exposure_pct)}
    mins = space["conditional"]["use_monthly_high_entry"]
    if "mh_min_age" in mins and "mh_max_age" in mins and min(mins["mh_min_age"]) >= max(mins["mh_max_age"]):
        errors.append("The smallest 'at least' age is not below the largest 'at most' age: no combination is valid.")
    return space, only, fixed, errors


# ---------------------------------------------------------------------------------------------------------------------------------- prices (disk cache)
def _cache_path(name: str, ticker: str) -> str:
    key = hashlib.md5(f"{ticker}|{td.START_DATE}".encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{name}_{key}.pkl")


def cache_age_hours(markets) -> float:
    """Hours since the OLDEST cached market of the list was saved (inf when something is missing)."""
    ages = []
    for m in markets:
        path = _cache_path(m, ALL_TICKERS[m])
        ages.append((time.time() - os.path.getmtime(path)) / 3600 if os.path.exists(path) else float("inf"))
    return max(ages) if ages else float("inf")


def load_prices(markets, refresh: bool = False) -> dict:
    """Daily prices of the markets: from the disk cache when saved within CACHE_HOURS hours, else downloaded (Yahoo, the engine's own cleaning) and saved."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    data, fetched = {}, []
    for m in markets:
        path = _cache_path(m, ALL_TICKERS[m])
        fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_HOURS * 3600 and not refresh
        if fresh:
            try:
                with open(path, "rb") as f:
                    data[m] = pickle.load(f)
                continue
            except Exception:
                pass
        got = _ORIGINAL_DOWNLOAD({m: ALL_TICKERS[m]}, td.START_DATE, td.END_DATE)
        if m in got:
            data[m] = got[m]
            fetched.append(m)
            with open(path, "wb") as f:
                pickle.dump(got[m], f)
    for name, df in data.items():
        df.attrs["asset"] = name
    print(f"Prices: {len(data) - len(fetched)} market(s) from the disk cache, {len(fetched)} downloaded" + (f" ({', '.join(fetched)})" if fetched else ""))
    return data


def clear_price_cache():
    if os.path.isdir(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(CACHE_DIR, f))


# ---------------------------------------------------------------------------------------------------------------------------------- the engine set-up for a run
def broker_profile():
    """The first futures broker profile of the brokers folder (Interactive Brokers) and its micro-contract sizing, or (None, None)."""
    for name, path in brokers.list_profiles().items():
        try:
            profile = brokers.load(path)
        except Exception:
            continue
        if profile.get("kind") == "futures":
            options = brokers.sizing_options(profile)
            return profile, (brokers.SIZING_MICRO if brokers.SIZING_MICRO in options else options[0])
    return None, None


def configure_account(s: dict, markets) -> list:
    """Sets the engine globals that describe the ACCOUNT (costs, sizing, caps, markets) for a backtest or a walk-forward. Returns notes about the cost model. Starts from the engine's own defaults every time."""
    notes = []
    for name, value in ENGINE_COSTS.items():
        setattr(td, name, value)
    for name, value in (("USE_PER_CONTRACT_FEES", False), ("FEE_SCHEDULE", {}), ("ASSET_COSTS", {}), ("SIZE_STEP", {}), ("USE_MARGIN_CALLS", False), ("MARGIN_SPEC", {})):
        setattr(td, name, copy.deepcopy(value))
    td.TAKE_MIN_ONE_UNIT = False
    td.MARK_TO_MARKET = True
    td.EXECUTION_TIMING = "next_open"
    td.ASSET_RISK_MULT, td.ASSET_RISK_NORMALIZE, td.STOP_SLIPPAGE_PCT = {}, False, 0.0
    td.CARRY_MODE, td.CARRY_RATE_SERIES, td.CARRY_RATE_FLAT, td.CARRY_DIVIDEND_YIELD = "off", None, 0.0, {}
    td.PORTFOLIO_STARTING_CAPITAL = float(s["capital"])
    td.USE_PORTFOLIO_RISK_CAP, td.MAX_PORTFOLIO_RISK_PCT = True, float(s["max_risk_pct"])
    td.USE_DRAWDOWN_SCALING = False
    if s["costs"] == COST_NONE:
        td.SLIPPAGE_PCT = td.SPREAD_PCT = td.COMMISSION_PCT = td.HOLDING_COST_PCT_PER_YEAR = 0.0
    elif s["costs"] == COST_IBKR:
        profile, sizing = broker_profile()
        if profile is None:
            notes.append("No futures broker profile found in the brokers folder: flat % costs were used instead.")
        else:
            compiled = brokers.compile_profile(profile, list(markets), sizing, margin_calls=bool(s.get("margin_calls")), margin_stress=1.0, stop_out_pct=None)
            for name, value in compiled["engine"].items():
                setattr(td, name, value)
            notes += list(compiled["notes"])
    td.TICKERS = {m: ALL_TICKERS[m] for m in markets}
    td.MH_CLEAN_MODE = True
    opt.ENABLE_S1, opt.ENABLE_S2 = True, False
    return notes


def configure_engine(s: dict, markets) -> list:
    """Sets every engine / optimizer / walk-forward global for a walk-forward run from the settings dict. Returns notes about the cost model."""
    notes = configure_account(s, markets)
    opt.SCORE_MODE = "portfolio"
    wf.SCORE_MODE = "portfolio"
    wf.TRAIN_YEARS, wf.TEST_YEARS, wf.ANCHORED_TRAIN = int(s["train_years"]), int(s["test_years"]), bool(s["anchored"])
    wf.N_TRIALS_PER_WINDOW = int(s["trials"])
    wf.RANK_METRIC = s["rank_metric"]
    wf.SEARCH_METHOD = s["search_method"]
    wf.PICK_MIDDLE_OF_TOP_N = int(s["top_n"]) if s["pick_middle"] else 1
    wf.SIM_WARMUP_YEARS = 3 if s["warm_start"] else None
    wf.USE_IBKR_FEES, wf.COST_OVERRIDES = False, None
    wf.RANDOM_SEED = int(s.get("seed", 42))
    wf.N_WORKERS = int(s["workers"]) if s.get("workers") else None
    wf.OUTPUT_DIR = OUTPUT_DIR
    space, only, fixed, _ = build_search(s["values"], s["searched"], s["candidates"], s["exposure_pct"])
    opt.set_search_space(space)
    wf.ONLY_OPTIMIZE, wf.FIXED_VALUES = only, fixed
    return notes


def restore_engine():
    td.MH_CLEAN_MODE = False
    td.TICKERS = dict(ALL_TICKERS)
    td.download_data = _ORIGINAL_DOWNLOAD
    opt.set_search_space(opt.DEFAULT_SEARCH_SPACE)
    wf.ONLY_OPTIMIZE = wf.FIXED_VALUES = None


# ---------------------------------------------------------------------------------------------------------------------------------- one background run at a time
LOCK = threading.Lock()
RUN = {"thread": None, "buf": io.StringIO(), "cancel": threading.Event(), "result": None, "error": None, "settings": None, "t0": None, "t1": None, "notes": []}


def is_running() -> bool:
    t = RUN["thread"]
    return bool(t is not None and t.is_alive())


def start_run(settings: dict) -> str:
    """Starts a walk-forward in a background thread. Returns '' when started, else the reason it did not."""
    if not LOCK.acquire(blocking=False):
        return "A run is still in progress."
    RUN.update(buf=io.StringIO(), cancel=threading.Event(), result=None, error=None, settings=settings, t0=time.time(), t1=None, notes=[])
    thread = threading.Thread(target=_run, args=(settings,), daemon=True)
    RUN["thread"] = thread
    thread.start()
    return ""


def cancel_run():
    RUN["cancel"].set()


def _run(settings: dict):
    try:
        with contextlib.redirect_stdout(RUN["buf"]):
            markets = list(settings["markets"])
            prices = load_prices(markets, refresh=bool(settings.get("refresh")))
            markets = [m for m in markets if m in prices]
            RUN["notes"] = configure_engine(settings, markets)
            td.download_data = lambda tickers, start, end: {k: v for k, v in prices.items() if k in tickers}     # the walk-forward asks for prices itself: hand it the cached ones
            result = wf.main(cancel_event=RUN["cancel"])
        RUN["result"] = summarize(result, settings, prices, RUN["buf"].getvalue())
    except Exception:
        RUN["error"] = traceback.format_exc()
    finally:
        restore_engine()
        RUN["t1"] = time.time()
        LOCK.release()


def progress() -> dict:
    """{'phase', 'done', 'total', 'elapsed', 'eta'} read from the walk-forward's own printed lines."""
    text = RUN["buf"].getvalue()
    elapsed = (RUN["t1"] or time.time()) - (RUN["t0"] or time.time())
    m = re.findall(r"\((\d+)/(\d+) windows finished, (\d+)s elapsed\)", text)
    total_m = re.search(r"Running (\d+) window", text)
    total = int(total_m.group(1)) if total_m else 0
    done = int(m[-1][0]) if m else 0
    if "WALK-FORWARD SUMMARY" in text:
        done = total
    if not text.strip() or "Downloading" in text and not total_m and "Prices:" not in text:
        phase = "Loading prices"
    elif not total_m:
        phase = "Preparing the windows"
    elif done < total:
        phase = f"Searching and testing windows ({done}/{total} done)"
    else:
        phase = "Summarizing"
    eta = (elapsed / done * (total - done)) if done and total > done else None
    return {"phase": phase, "done": done, "total": total, "elapsed": elapsed, "eta": eta}


# ---------------------------------------------------------------------------------------------------------------------------------- the summary of a finished run
def _equal_weight_buy_and_hold(prices: dict, capital: float, index: pd.DatetimeIndex) -> pd.Series:
    """Equal-weight buy & hold of the markets, on the dates of the strategy's curve, rebased to `capital` at its first date. Each market is held from its first bar at 1/N of the capital."""
    n = len(prices)
    parts = []
    for name, df in prices.items():
        daily = df["Close"].pct_change().reindex(index).fillna(0.0)
        parts.append((1.0 + daily).cumprod())
    avg = sum(parts) / n if parts else pd.Series(1.0, index=index)
    return capital * avg / avg.iloc[0]


def summarize(result: dict, s: dict, prices: dict, log: str) -> dict:
    capital = float(s["capital"])
    out = {"summary": result.get("summary", {}), "log": log, "settings": s, "windows": pd.DataFrame(), "curve": None, "bh": None, "stats": None, "bh_stats": None, "picks": pd.DataFrame(),
           "chart_data": result.get("window_oos_chart_data") or [], "window_params": {i: r.get("params") for i, r in enumerate(result.get("window_reports", []), start=1)}}
    rows = []
    for i, r in enumerate(result.get("window_reports", []), start=1):
        row = {"window": i, "status": r["status"], "train": f"{r['train_start'].date()} to {r['train_end'].date()}", "test": f"{r['test_start'].date()} to {r['test_end'].date()}",
               "IS Sharpe": r.get("is_sharpe"), "OOS Sharpe": r.get("oos_sharpe"), "OOS CAGR %": None if r.get("oos_cagr") is None else r["oos_cagr"] * 100,
               "OOS max DD %": None if r.get("oos_max_dd") is None else r["oos_max_dd"] * 100, "OOS trades": r.get("oos_trades")}
        for key, short in PARAM_NAMES.items():
            if r.get("params") and key in r["params"]:
                row[short] = r["params"][key]
        rows.append(row)
    out["windows"] = pd.DataFrame(rows)
    curves = result.get("stitched_oos_curves") or {}
    curve = curves.get("PORTFOLIO")
    if curve is None and curves:
        curve = next(iter(curves.values()))
    if curve is not None and len(curve) > 5:
        curve = curve.dropna()
        curve = capital * curve / curve.iloc[0]
        out["curve"] = curve
        out["stats"] = td.performance_stats(curve, capital)
        try:
            bh = _equal_weight_buy_and_hold(prices, capital, curve.index)
            out["bh"], out["bh_stats"] = bh, td.performance_stats(bh, capital)
        except Exception:
            pass
    ok = out["windows"][out["windows"]["status"] == "ok"] if len(out["windows"]) else out["windows"]
    if len(ok):
        picks = []
        for short in PARAM_NAMES.values():
            if short in ok.columns:
                col = ok[short].dropna()
                if len(col):
                    top = col.value_counts().iloc[0]
                    picks.append({"setting": short, "most chosen": col.value_counts().index[0], "in windows": f"{int(top)}/{len(col)}", "lowest": col.min(), "median": col.median(), "highest": col.max()})
        out["picks"] = pd.DataFrame(picks)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- settings as text (copy / paste / send between tabs)
_KEY_ALIASES = {
    "mh_sma_period": ("ema", "emalength", "emaperiod", "mhsmaperiod", "pricemustbeabovethisema"),
    "mh_min_age": ("minage", "mhminage", "minimumage", "monthlyhighminage"),
    "mh_max_age": ("maxage", "mhmaxage", "maximumage", "monthlyhighmaxage"),
    "entry_buffer_n_mult": ("buffer", "entrybuffer", "entrybuffernmult", "bufn"),
    "n_atr_period": ("slatr", "slatrperiod", "slperiod", "natrperiod", "sllen"),
    "stop_n_mult": ("slx", "slmult", "slmultiplier", "stopnmult", "stopmult"),
    "tp_atr_period": ("tpatr", "tpatrperiod", "tpperiod", "tplen"),
    "tp_n_mult": ("tpx", "tpmult", "tpmultiplier", "tpnmult"),
    "risk_per_trade_pct": ("risk", "riskperunit", "riskpertradepct", "riskpct"),
}
_ALIAS_TO_KEY = {alias: key for key, aliases in _KEY_ALIASES.items() for alias in aliases + (key.replace("_", "").replace("n", "n"),)}
_ALIAS_TO_KEY.update({re.sub(r"[^a-z]", "", key): key for key in _KEY_ALIASES})


def format_settings(values: dict) -> str:
    """The strategy's numbers as one JSON line (what the app copies and pastes)."""
    import json
    keys = [k for k, *_ in PARAMS] + ["risk_per_trade_pct"]
    return json.dumps({k: values[k] for k in keys if k in values})


def format_for_tradingview(values: dict) -> str:
    """The same numbers in the words of the TradingView script's inputs, to type in there."""
    return (f"Monthly high min age {values['mh_min_age']} / max age {values['mh_max_age']} | EMA {values['mh_sma_period']} | Entry buffer {values['entry_buffer_n_mult']:g} | "
            f"SL ATR period {values['n_atr_period']} x multiplier {values['stop_n_mult']:g} | TP ATR period {values['tp_atr_period']} x multiplier {values['tp_n_mult']:g} | "
            f"Risk per unit {values.get('risk_per_trade_pct', RISK_DEFAULT):g}%")


def parse_settings(text: str) -> tuple:
    """(values, problems) from pasted text: the app's own JSON line, or any 'name = value' / 'name: value' pairs (names like EMA, min age, buffer, SL ATR, SL x, TP ATR, TP x, risk, or the
    engine's own keys). Only the settings found are returned; values are checked against the lowest each setting accepts."""
    import json
    found, problems = {}, []
    text = (text or "").strip()
    if not text:
        return {}, ["Nothing to apply."]
    pairs = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            pairs = list(obj.items())
    except Exception:
        pass
    if not pairs:
        # the TradingView-words line ("SL ATR period 185 x multiplier 6") becomes plain pairs first
        text = re.sub(r"(?i)\b(SL|TP) ATR period\s*(\d+(?:\.\d+)?)\s*x\s*multiplier\s*(\d+(?:\.\d+)?)", r"\1 ATR period \2, \1 x \3", text)
        pairs = re.findall(r"([A-Za-z_][A-Za-z_ ]*?)\s*(?:[:=]|\s)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    for name, raw in pairs:
        key = _ALIAS_TO_KEY.get(re.sub(r"[^a-z]", "", str(name).lower()))
        if key is None:
            problems.append(f"Unknown setting '{name}' ignored.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            problems.append(f"'{name}': '{raw}' is not a number.")
            continue
        default = DEFAULTS.get(key, RISK_DEFAULT)
        lowest = next((low for k, _, _, _, _, low in PARAMS if k == key), 0.05)
        if value < lowest:
            problems.append(f"'{name}' must be at least {lowest:g}.")
            continue
        found[key] = int(round(value)) if isinstance(default, int) and not isinstance(default, bool) else float(value)
    if not found and not problems:
        problems.append("No settings recognised.")
    return found, problems


# ---------------------------------------------------------------------------------------------------------------------------------- one backtest
def _exit_reason_pnl(trades: list) -> dict:
    out = {}
    for t in trades:
        for u in t.get("unit_details") or []:
            out[u["exit_reason"]] = out.get(u["exit_reason"], 0.0) + u["pnl"]
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def run_backtest(s: dict) -> dict:
    """A plain backtest of the strategy with the given numbers on the shared account, from s['start'] (indicators use all earlier history). Returns {'error': ...} or the result dict.
    Refuses while a walk-forward is running (both set the same engine globals)."""
    if not LOCK.acquire(blocking=False):
        return {"error": "A walk-forward is running: wait for it to finish (or stop it) before a backtest."}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prices = load_prices(list(s["markets"]), refresh=bool(s.get("refresh")))
            markets = [m for m in s["markets"] if m in prices]
            if not markets:
                return {"error": "No prices could be loaded for the chosen markets."}
            notes = configure_account(s, markets)
            params = {**s["values"], "max_units_per_trend": UNIT_CAP, "use_monthly_high_entry": True, "MAX_TOTAL_EXPOSURE_PCT": int(s["exposure_pct"])}
            opt.apply_params(params)                                   # the one clean system, through the same function the optimizer and walk-forward use
            start = pd.Timestamp(s["start"])
            features = {}
            for m in markets:
                f = td.build_features(prices[m]).loc[start:].copy()
                f.attrs["asset"] = m
                features[m] = f
            td.RECORD_HISTORY = True
            try:
                port = td.simulate_portfolio(features)
            finally:
                td.RECORD_HISTORY = False
        capital = float(s["capital"])
        eq = port["equity_curve"].dropna()
        out = {"settings": s, "notes": notes, "port": port, "curve": eq, "stats": td.performance_stats(eq, capital),
               "trades": {m: port["trades_by_asset"].get(m, []) for m in markets}}
        try:
            bh = _equal_weight_buy_and_hold({m: prices[m] for m in markets}, capital, eq.index)
            out["bh"], out["bh_stats"] = bh, td.performance_stats(bh, capital)
        except Exception:
            out["bh"], out["bh_stats"] = None, None
        all_trades = [t for ts in out["trades"].values() for t in ts]
        out["episode_stats"] = td.episode_stats(all_trades) if all_trades else {}
        out["exit_pnl"] = _exit_reason_pnl(all_trades)
        rows = []
        for m, ts in out["trades"].items():
            ep = td.episode_stats(ts) if ts else {}
            rows.append({"market": m, "trades": len(ts), "win rate %": None if not ts else round(ep["Win Rate"] * 100, 1), "P&L $": round(sum(t["pnl"] for t in ts), 0),
                         "avg R": None if not ts or not np.isfinite(ep["Avg R"]) else round(ep["Avg R"], 2),
                         "profit factor": None if not ts or not np.isfinite(ep["Profit Factor"]) else round(ep["Profit Factor"], 2)})
        out["assets"] = pd.DataFrame(rows).sort_values("P&L $", ascending=False)
        out["episodes"] = td.build_episode_dataframe({m: {"trades": ts} for m, ts in out["trades"].items() if ts}) if all_trades else pd.DataFrame()
        # the same structure a walk-forward window gives the Positions tab: one 'window' that spans the whole test, with the numbers used
        assets = {}
        for m in markets:
            idx = features[m].index
            exits = {name: [(idx[i], price, reason) for i, price, reason in markers] for name, markers in port["exit_markers_by_asset"].get(m, {}).items()}
            assets[m] = {"trades_by_system": port["trades_by_system"].get(m, {}), "exit_markers_by_system": exits}
        out["chart_data"] = [{"window": 1, "test_start": eq.index[0], "test_end": eq.index[-1], "params": params, "assets": assets}]
        return out
    except Exception:
        return {"error": traceback.format_exc()}
    finally:
        restore_engine()
        LOCK.release()
