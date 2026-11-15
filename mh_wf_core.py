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

import baseline_defaults as BD
import brokers
import ibkr_fees
import tv_prices
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
    ("mh_sma_period", "Price must be above this EMA (days)", "Initial entry and pyramiding", BD.MH_VALUES["mh_sma_period"], BD.MH_CANDIDATES["mh_sma_period"], 1),
    ("mh_min_age", "Monthly high at least this many days old", "Initial entry and pyramiding", BD.MH_VALUES["mh_min_age"], BD.MH_CANDIDATES["mh_min_age"], 1),
    ("mh_max_age", "Monthly high at most this many days old", "Initial entry and pyramiding", BD.MH_VALUES["mh_max_age"], BD.MH_CANDIDATES["mh_max_age"], 2),
    ("entry_buffer_n_mult", "Entry buffer: the close must be this many ATR above the monthly high", "Initial entry and pyramiding", BD.MH_VALUES["entry_buffer_n_mult"], BD.MH_CANDIDATES["entry_buffer_n_mult"], 0.0),
    ("n_atr_period", "SL ATR period (days)", "Stop loss", BD.MH_VALUES["n_atr_period"], BD.MH_CANDIDATES["n_atr_period"], 2),
    ("stop_n_mult", "SL ATR multiplier", "Stop loss", BD.MH_VALUES["stop_n_mult"], BD.MH_CANDIDATES["stop_n_mult"], 0.5),
    ("tp_atr_period", "TP ATR period (days)", "Take profit", BD.MH_VALUES["tp_atr_period"], BD.MH_CANDIDATES["tp_atr_period"], 2),
    ("tp_n_mult", "TP ATR multiplier", "Take profit", BD.MH_VALUES["tp_n_mult"], BD.MH_CANDIDATES["tp_n_mult"], 0.25),
]
PARAM_NAMES = {"mh_sma_period": "EMA", "mh_min_age": "min age", "mh_max_age": "max age", "entry_buffer_n_mult": "buffer", "n_atr_period": "SL ATR",
               "stop_n_mult": "SL x", "tp_atr_period": "TP ATR", "tp_n_mult": "TP x"}
DEFAULTS = {key: default for key, _, _, default, _, _ in PARAMS}
RISK_DEFAULT = BD.RISK_PER_UNIT_PCT
UNIT_CAP = 999                                          # no cap of its own: the exposure and open-risk limits still apply

COST_NONE = "No costs"
COST_FLAT = "Flat % costs (the engine's own, fractional sizes)"
COST_IBKR = "Interactive Brokers futures (per-contract fees, whole contracts)"
CONTRACT_SIZES = {brokers.SIZING_MICRO: "Micro (e.g. MNQ, $2 a point)", brokers.SIZING_NANO: "Nano (a tenth of a micro, modelled)", brokers.SIZING_MINI: "Standard (e.g. NQ, $20 a point)"}


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


SOURCE_YAHOO = "yahoo"
SOURCE_TV = "tradingview"
SOURCES = {SOURCE_YAHOO: "Yahoo Finance (as before)", SOURCE_TV: "TradingView files (longer history)"}


def default_history_start() -> str:
    """The baseline's start (1951-01-01) for the TradingView files, which go back that far; Yahoo's own history is the last 30 years."""
    return BD.HISTORY_START


FINGERPRINT_FILES = ("baseline_defaults.py", "brokers.py", "brokers_builtin.py", "ibkr_fees.py", "mh_walkforward_ui.py", "mh_wf_charts.py", "mh_wf_core.py", "monte_carlo.py",
                     "optimize_turtle_dual_system.py", "turtle_dual_system.py", "tv_prices.py", "walk_forward_turtle.py", "wf_fast.py")


def code_fingerprint() -> str:
    """A short fingerprint of the app's own code files (line endings ignored; a missing file counts). The same files give the same fingerprint on every machine, so a cloud app and a local one
    can be compared at a glance: a different fingerprint means a different version, or files that were not uploaded."""
    h = hashlib.md5()
    for name in FINGERPRINT_FILES:
        h.update(name.encode())
        path = os.path.join(ROOT, name)
        h.update(open(path, "rb").read().replace(b"\r\n", b"\n") if os.path.exists(path) else b"MISSING")
    return h.hexdigest()[:8]


def setup_check(markets, costs: str, source: str) -> str:
    """One line for the sidebar: the code fingerprint, where the cost profile comes from, and how many TradingView files are here."""
    parts = [f"code {code_fingerprint()}"]
    if costs == COST_IBKR:
        paths = brokers.list_profiles()
        name = next((n for n, p in paths.items() if "futures" in n.lower()), None)
        parts.append("costs: Interactive Brokers, " + (f"{brokers.profile_source(paths[name])} profile" if name else "NO PROFILE"))
    if source == SOURCE_TV:
        want = [m for m in markets if m in tv_prices.MARKETS]
        parts.append(f"TradingView files {sum(1 for m in want if tv_prices.has(m))}/{len(want)}")
    return " · ".join(parts)


def data_summary(prices: dict, source: str) -> dict:
    """Where a run's prices came from and how far they reach: {'source', 'first', 'last'} (the earliest first bar and the latest last bar of the markets), {} when there are none."""
    firsts = [df.index[0] for df in prices.values() if len(df)]
    lasts = [df.index[-1] for df in prices.values() if len(df)]
    return {"source": source, "first": min(firsts), "last": max(lasts)} if firsts else {}


def data_caption(info: dict) -> str:
    """'prices: TradingView files, 1951-01-02 to 2026-11-13' for a result's caption ('' without information)."""
    if not info:
        return ""
    return f"prices: {'TradingView files' if info['source'] == SOURCE_TV else 'Yahoo Finance'}, {info['first'].date()} to {info['last'].date()}"


def load_prices(markets, refresh: bool = False, source: str = SOURCE_YAHOO, history_start: str = None) -> dict:
    """Daily prices of the markets. Yahoo: from the disk cache when saved within CACHE_HOURS hours, else downloaded (the engine's own cleaning) and saved. TradingView: read from the files of
    tv_prices.py (the engine's own cleaning too), from `history_start` (default the baseline's 1951-01-01). A market whose file is missing is downloaded from TradingView first, and if that fails the
    run stops with a clear error: a market with a TradingView symbol is never quietly taken from Yahoo (whose history is only the last 30 years)."""
    if source == SOURCE_TV:
        have = [m for m in markets if m in tv_prices.MARKETS]
        data = {}
        if have:
            data = _ORIGINAL_DOWNLOAD({m: ALL_TICKERS[m] for m in have}, history_start or td.START_DATE, td.END_DATE, SOURCE_TV)
        for name, df in data.items():
            df.attrs["asset"] = name
        rest = [m for m in markets if m not in tv_prices.MARKETS]          # no TradingView symbol at all: only these may come from Yahoo
        if rest:
            print(f"No TradingView symbol for {', '.join(rest)}: Yahoo is used for them.")
            data.update(load_prices(rest, refresh))
        info = data_summary(data, source)
        print(f"Prices: {len([m for m in markets if m in data and m in have])} market(s) from the TradingView files" + (f", {info['first'].date()} to {info['last'].date()}" if info else ""))
        return data
    os.makedirs(CACHE_DIR, exist_ok=True)
    data, fetched, stale = {}, [], []
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
        try:
            got = _ORIGINAL_DOWNLOAD({m: ALL_TICKERS[m]}, td.START_DATE, td.END_DATE)
        except Exception as e:                                  # e.g. Yahoo refusing a shared server: fall back to what is saved
            print(f"Download of {m} failed ({type(e).__name__}: {str(e)[:80]})")
            got = {}
        if m in got:
            data[m] = got[m]
            fetched.append(m)
            with open(path, "wb") as f:
                pickle.dump(got[m], f)
        elif os.path.exists(path):                              # the download gave nothing: use the saved copy however old it is
            try:
                with open(path, "rb") as f:
                    data[m] = pickle.load(f)
                stale.append(m)
            except Exception:
                pass
    for name, df in data.items():
        df.attrs["asset"] = name
    print(f"Prices: {len(data) - len(fetched) - len(stale)} market(s) from the disk cache, {len(fetched)} downloaded" + (f" ({', '.join(fetched)})" if fetched else "")
          + (f"; {len(stale)} saved copy(ies) used because the download failed ({', '.join(stale)})" if stale else ""))
    return data


def clear_price_cache():
    if os.path.isdir(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(CACHE_DIR, f))


# ---------------------------------------------------------------------------------------------------------------------------------- the engine set-up for a run
def broker_profile():
    """The first futures broker profile (Interactive Brokers: the brokers/ folder, or the copy written into brokers_builtin.py) and its micro-contract sizing, or (None, None)."""
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
    td.ASSET_MARGIN_SLICE = "equal" if s.get("margin_slice") else None            # an equal slice of equity per market, used as margin (a cap on the normal sizing)
    td.CARRY_MODE, td.CARRY_RATE_SERIES, td.CARRY_RATE_FLAT, td.CARRY_DIVIDEND_YIELD = "off", None, 0.0, {}
    td.PORTFOLIO_STARTING_CAPITAL = float(s["capital"])
    td.USE_PORTFOLIO_RISK_CAP, td.MAX_PORTFOLIO_RISK_PCT = True, float(s["max_risk_pct"])
    td.USE_DRAWDOWN_SCALING = False
    if s["costs"] == COST_NONE:
        td.SLIPPAGE_PCT = td.SPREAD_PCT = td.COMMISSION_PCT = td.HOLDING_COST_PCT_PER_YEAR = 0.0
    elif s["costs"] == COST_IBKR:
        profile, sizing = broker_profile()
        sizing = s.get("contract_size") or sizing
        if profile is None:
            raise RuntimeError("No futures broker profile found (neither the brokers/ folder nor brokers_builtin.py): the Interactive Brokers cost model cannot be built. "
                               "Upload brokers_builtin.py, or choose another trading-cost setting.")      # never quietly swap in a different cost model
        else:
            compiled = brokers.compile_profile(profile, list(markets), sizing, margin_calls=bool(s.get("margin_calls")), margin_stress=1.0, stop_out_pct=None)
            for name, value in compiled["engine"].items():
                setattr(td, name, value)
            notes += list(compiled["notes"])
            td.TAKE_MIN_ONE_UNIT = bool(s.get("take_min_one"))          # whole contracts: when the risk-based size rounds down to zero, open one contract anyway (if it fits the notional cap)
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


def configure_optimizer(s: dict, markets) -> tuple:
    """Sets the engine for an OPTIMIZER run: the same account as a backtest, every trial trading from s['start'] (the indicators use all the history before it), ranked by one metric.
    Returns (notes, only_optimize, fixed_values)."""
    notes = configure_account(s, markets)
    opt.SCORE_MODE = "portfolio"
    opt.RANK_METRIC = s["rank_metric"]
    opt.RANDOM_SEED = int(s.get("seed", 42))
    opt.FEATURE_START = pd.Timestamp(s["start"])
    space, only, fixed, _ = build_search(s["values"], s["searched"], s["candidates"], s["exposure_pct"])
    opt.set_search_space(space)
    return notes, only, fixed


def restore_engine():
    td.MH_CLEAN_MODE = False
    opt.FEATURE_START = None
    td.TICKERS = dict(ALL_TICKERS)
    td.download_data = _ORIGINAL_DOWNLOAD
    opt.set_search_space(opt.DEFAULT_SEARCH_SPACE)
    wf.ONLY_OPTIMIZE = wf.FIXED_VALUES = None


# ---------------------------------------------------------------------------------------------------------------------------------- one background run at a time
LOCK = threading.Lock()
RUN = {"thread": None, "buf": io.StringIO(), "cancel": threading.Event(), "result": None, "error": None, "settings": None, "t0": None, "t1": None, "notes": [], "opt": {}}


def is_running() -> bool:
    t = RUN["thread"]
    return bool(t is not None and t.is_alive())


def run_kind() -> str:
    """'optimize' or 'walkforward': what the last started run (finished or not) was."""
    return (RUN["settings"] or {}).get("kind", "walkforward")


def start_run(settings: dict) -> str:
    """Starts a walk-forward (or, with settings['kind'] == 'optimize', an optimizer search) in a background thread. Returns '' when started, else the reason it did not."""
    if not LOCK.acquire(blocking=False):
        return "A run is still in progress."
    RUN.update(buf=io.StringIO(), cancel=threading.Event(), result=None, error=None, settings=settings, t0=time.time(), t1=None, notes=[], opt={})
    thread = threading.Thread(target=_run_optimize if settings.get("kind") == "optimize" else _run, args=(settings,), daemon=True)
    RUN["thread"] = thread
    thread.start()
    return ""


def cancel_run():
    RUN["cancel"].set()


def _run(settings: dict):
    try:
        with contextlib.redirect_stdout(RUN["buf"]):
            markets = list(settings["markets"])
            prices = load_prices(markets, refresh=bool(settings.get("refresh")), source=settings.get("price_source", SOURCE_YAHOO), history_start=settings.get("history_start"))
            markets = [m for m in markets if m in prices]
            RUN["notes"] = configure_engine(settings, markets)
            td.download_data = lambda tickers, start, end: {k: v for k, v in prices.items() if k in tickers}     # the walk-forward asks for prices itself: hand it the cached ones
            result = wf.main(cancel_event=RUN["cancel"])
        RUN["result"] = summarize(result, settings, prices, RUN["buf"].getvalue())
        RUN["result"]["data_info"] = data_summary(prices, settings.get("price_source", SOURCE_YAHOO))
    except Exception:
        RUN["error"] = traceback.format_exc()
    finally:
        restore_engine()
        RUN["t1"] = time.time()
        LOCK.release()


def _run_optimize(settings: dict):
    """The optimizer: an Optuna search over the ticked settings on ONE stretch of history (from settings['start'] to the last bar), every trial a full backtest of the shared account."""
    try:
        with contextlib.redirect_stdout(RUN["buf"]):
            markets = list(settings["markets"])
            prices = load_prices(markets, refresh=bool(settings.get("refresh")), source=settings.get("price_source", SOURCE_YAHOO), history_start=settings.get("history_start"))
            markets = [m for m in markets if m in prices]
            raw = {m: prices[m] for m in markets}
            RUN["notes"], only, fixed = configure_optimizer(settings, markets)
            ath = {m: df["High"].cummax() for m, df in raw.items()}        # as the walk-forward does (the clean strategy does not use it)
            n_workers = int(settings["workers"]) if settings.get("workers") else max(1, (os.cpu_count() or 2) - 1)
            n_workers = max(1, min(n_workers, int(settings["trials"])))
            print(f"Optimizer: {settings['trials']} trials, ranked by {settings['rank_metric']}, {n_workers} worker process(es), {len(markets)} markets from {settings['start']}")
            t0 = time.time()

            def on_round(done, total, n_valid, n_bad):
                RUN["opt"] = {"done": done, "total": total, "valid": n_valid, "failed": n_bad}
                print(f"  ...{done}/{total} trials done ({time.time() - t0:.0f}s elapsed, {n_valid} valid, {n_bad} skipped)")

            results, n_failed = opt.search_optuna(raw, ath, int(settings["trials"]), n_workers, only_optimize=only, fixed_values=fixed, seed=int(settings.get("seed", 42)),
                                                  progress_cb=on_round, cancel_check=RUN["cancel"].is_set)
            if RUN["cancel"].is_set():
                print("Stopped: the trials finished so far are kept.")
        RUN["result"] = summarize_optimize(results, n_failed, settings, RUN["buf"].getvalue())
        RUN["result"]["data_info"] = data_summary(raw, settings.get("price_source", SOURCE_YAHOO))
    except Exception:
        RUN["error"] = traceback.format_exc()
    finally:
        restore_engine()
        RUN["t1"] = time.time()
        LOCK.release()


def progress() -> dict:
    """{'phase', 'done', 'total', 'elapsed', 'eta'} read from the walk-forward's own printed lines (or, for the optimizer, from its rounds)."""
    text = RUN["buf"].getvalue()
    elapsed = (RUN["t1"] or time.time()) - (RUN["t0"] or time.time())
    if run_kind() == "optimize":
        o = RUN["opt"] or {}
        done, total = int(o.get("done", 0)), int(o.get("total", 0))
        phase = "Loading prices and starting the search" if not total else (f"Searching ({done}/{total} trials done, {o.get('valid', 0)} valid)" if done < total else "Summarizing")
        return {"phase": phase, "done": done, "total": total, "elapsed": elapsed, "eta": (elapsed / done * (total - done)) if done and total > done else None}
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
               "IS Sharpe": r.get("is_sharpe"), "OOS Sharpe": r.get("oos_sharpe"), "IS Sortino": r.get("is_sortino"), "OOS Sortino": r.get("oos_sortino"), "OOS CAGR %": None if r.get("oos_cagr") is None else r["oos_cagr"] * 100,
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


def summarize_optimize(results: list, n_failed: int, s: dict, log: str, keep: int = 100) -> dict:
    """The optimizer's trials ranked by the metric it was asked to rank by: the best `keep` as a table (one row per trial) and as the full result dicts."""
    metric = s["rank_metric"]
    ranked = sorted((r for r in results if np.isfinite(r.get(metric, np.nan))), key=lambda r: r[metric], reverse=True)[:keep]
    rows = []
    for rank, r in enumerate(ranked, start=1):
        row = {"rank": rank, "Sharpe": r["sharpe"], "Sortino": r["sortino"], "Calmar": r["calmar"], "CAGR %": r["cagr"] * 100, "Max DD %": r["max_dd"] * 100, "trades": r["total_trades"]}
        for key, short in PARAM_NAMES.items():
            if key in r["params"]:
                row[short] = r["params"][key]
        rows.append(row)
    return {"kind": "optimize", "settings": s, "ranked": ranked, "table": pd.DataFrame(rows), "n_valid": len(results), "n_failed": int(n_failed), "log": log}


# ---------------------------------------------------------------------------------------------------------------------------------- returns by period
def _period_keys(index: pd.DatetimeIndex, period: str) -> list:
    if period == "year":
        return [index.year]
    if period == "quarter":
        return [index.year, (index.month - 1) // 3 + 1]
    if period == "month":
        return [index.year, index.month]
    raise ValueError(f"period must be 'year', 'quarter' or 'month', not {period!r}")


def period_returns(curve: pd.Series, period: str = "year") -> pd.Series:
    """The account's compounded return in each calendar year / quarter / month, from its equity curve (the last value of each period against the last value of the one before; the first period
    against the curve's first value). The index is the year, or (year, quarter), or (year, month). The first and last periods may be incomplete."""
    curve = curve.dropna()
    if len(curve) < 2:
        return pd.Series(dtype=float)
    ends = curve.groupby(_period_keys(curve.index, period)).last()
    before = ends.shift(1)
    before.iloc[0] = curve.iloc[0]
    return ends / before - 1.0


def yearly_table(curve: pd.Series, bh: pd.Series = None) -> pd.DataFrame:
    """One row per calendar year: the strategy's return, the equal-weight buy & hold return (when given), the worst drawdown inside the year (from the previous year-end value or the year's own peak,
    whichever is higher), the equity at the year's end and whether the year is incomplete (the curve starts after 10 January or ends before 20 December)."""
    curve = curve.dropna()
    if len(curve) < 2:
        return pd.DataFrame(columns=["year", "return", "bh_return", "max_dd", "end_equity", "partial"])
    returns = period_returns(curve, "year")
    bh_returns = period_returns(bh, "year") if bh is not None and len(bh.dropna()) > 1 else None
    rows, previous_end = [], float(curve.iloc[0])
    for year, seg in curve.groupby(curve.index.year):
        path = np.concatenate([[previous_end], seg.to_numpy(float)])
        dd = float((path / np.maximum.accumulate(path) - 1.0).min())
        first, last = seg.index[0], seg.index[-1]
        partial = bool((year == curve.index[0].year and first > pd.Timestamp(year, 1, 10)) or (year == curve.index[-1].year and last < pd.Timestamp(year, 12, 20)))
        rows.append({"year": int(year), "return": float(returns.loc[year]), "bh_return": None if bh_returns is None or year not in bh_returns.index else float(bh_returns.loc[year]),
                     "max_dd": dd, "end_equity": float(seg.iloc[-1]), "partial": partial, "to": last})
        previous_end = float(seg.iloc[-1])
    return pd.DataFrame(rows)


def period_matrix(curve: pd.Series, period: str = "month") -> pd.DataFrame:
    """Returns as a table: one row per year, one column per month (period='month': Jan..Dec) or quarter (Q1..Q4), plus a 'Year' column with the calendar-year return. Missing periods are NaN."""
    r = period_returns(curve, period)
    if not len(r):
        return pd.DataFrame()
    n = 12 if period == "month" else 4
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if period == "month" else ["Q1", "Q2", "Q3", "Q4"]
    table = r.unstack(level=1).reindex(columns=range(1, n + 1))
    table.columns = names
    table["Year"] = period_returns(curve, "year")
    table.index.name = "year"
    return table


def return_summary(curve: pd.Series, period: str) -> dict:
    """Best, worst and median period, and the share of positive periods, for the equity curve (incomplete first / last periods included)."""
    r = period_returns(curve, period)
    if not len(r):
        return {}
    return {"n": int(len(r)), "positive": int((r > 0).sum()), "best": float(r.max()), "best_at": r.idxmax(), "worst": float(r.min()), "worst_at": r.idxmin(), "median": float(r.median()), "mean": float(r.mean())}


def pnl_by_market_year(trades: dict) -> pd.DataFrame:
    """Realized profit and loss ($, after costs) of the trades by market and by the year each trade closed. trades = {market: [trade dicts with 'pnl' and 'exit_date']}. Totals in a last row and column."""
    rows = []
    for market, ts in trades.items():
        for t in ts or []:
            if t.get("exit_date") is not None:
                rows.append({"market": market, "year": pd.Timestamp(t["exit_date"]).year, "pnl": float(t["pnl"])})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).pivot_table(index="market", columns="year", values="pnl", aggfunc="sum").fillna(0.0)
    table.columns = [str(c) for c in table.columns]                      # the year columns are text, like the 'Total' column next to them
    table["Total"] = table.sum(axis=1)
    table = table.sort_values("Total", ascending=False)
    table.loc["All markets"] = table.sum(axis=0)
    return table


def trial_settings(params: dict, risk: float) -> dict:
    """One trial's numbers as the strategy panel takes them: whole numbers where the setting is a whole number, plus the risk %."""
    out = {key: (int(round(float(params[key]))) if isinstance(DEFAULTS.get(key), int) else float(params[key])) for key, *_ in PARAMS if key in params}
    out["risk_per_trade_pct"] = float(risk)
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
        return {"error": "A walk-forward or optimizer run is in progress: wait for it to finish (or stop it) before a backtest."}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prices = load_prices(list(s["markets"]), refresh=bool(s.get("refresh")), source=s.get("price_source", SOURCE_YAHOO), history_start=s.get("history_start"))
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
        out = {"settings": s, "notes": notes, "data_info": data_summary({m: prices[m] for m in markets}, s.get("price_source", SOURCE_YAHOO)), "port": port, "curve": eq, "stats": td.performance_stats(eq, capital),
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
