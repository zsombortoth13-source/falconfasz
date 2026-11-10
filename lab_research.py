"""Research harness for the mechanism lab: setup, leverage calibration to a drawdown cap, gap stress, paired comparison against a baseline,
trade analytics, a results ledger, and a parallel runner."""
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import brokers
import carry_data
import lab_data
import turtle_dual_system as td
from strategy_lab import Features, LabResult, run_design

PROJECT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(PROJECT, "research")
LEDGER = os.path.join(RESEARCH_DIR, "ledger.csv")
DD_CAPS = (50.0, 70.0)         # the worst drawdown a candidate may show, % (the search is run under each)
CAPITAL = 100_000.0


# ----------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------
def configure(markets=None, carry=True):
    """Costs like the app's default account (IBKR profile, fractional sizing, next-open fills) and, if `carry`, the T-bill / dividend model."""
    names = list(markets) if markets else list(lab_data.MARKETS)
    prof = brokers.load(os.path.join(PROJECT, "brokers", "ibkr.json"))
    for k, v in brokers.compile_profile(prof, names, "fractional")["engine"].items():
        setattr(td, k, v)
    td.MARK_TO_MARKET, td.EXECUTION_TIMING = True, "next_open"
    td.USE_MARGIN_CALLS = False
    if carry:
        td.CARRY_MODE = "futures"
        td.CARRY_RATE_SERIES = carry_data.tbill_series()
        td.CARRY_RATE_FLAT = 0.04
        td.CARRY_DIVIDEND_YIELD = carry_data.dividend_yields(1.0)
    else:
        td.CARRY_MODE, td.CARRY_RATE_SERIES, td.CARRY_DIVIDEND_YIELD = "off", None, {}


_FEATURES = {}


def market_features(name: str, period: str = "design") -> Features:
    """Cached Features for a market over its "design" years, its "sealed" years (after DESIGN_END) or "full" history."""
    key = (name, period)
    if key not in _FEATURES:
        df = lab_data.load_market(name)
        df = {"design": lab_data.design_slice, "sealed": lab_data.sealed_slice, "full": lambda d: d}[period](df)
        _FEATURES[key] = Features(df, name)
    return _FEATURES[key]


def stress_gaps(f: Features, threshold: float = -0.04) -> Features:
    """The same market with every day that closed `threshold` or worse (close to close) opening AT its close: the whole fall happens before
    anyone can trade, so an exit or stop filled at that open takes the full loss (the real 1987 open equalled the previous close)."""
    df = f.df.copy()
    ret = df["Close"].pct_change()
    bad = ret <= threshold
    df.loc[bad, "Open"] = df.loc[bad, "Close"]
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    return Features(df, f.asset)


# ----------------------------------------------------------------------------
# leverage calibration: the most CAGR a design can show inside a drawdown cap
# ----------------------------------------------------------------------------
def calibrate(make_design, f: Features, dd_cap: float, lo: float = 0.1, hi: float = 8.0, iters: int = 9):
    """Scale the design's exposure by L (make_design(L)) and find the largest L whose worst drawdown stays inside `dd_cap` percent.
    Returns (L, LabResult at that L). If even `lo` breaks the cap the result at `lo` is returned with L = nan."""
    def trial(L):
        res = run_design(make_design(L), f)
        return res, -res.stats()["MaxDD"]
    res_hi, dd_hi = trial(hi)
    if dd_hi <= dd_cap:
        return hi, res_hi
    res_lo, dd_lo = trial(lo)
    if dd_lo > dd_cap:
        return float("nan"), res_lo
    best_L, best_res = lo, res_lo
    a, b = lo, hi
    for _ in range(iters):
        mid = (a * b) ** 0.5            # geometric bisection: exposure spans a wide range
        res, dd = trial(mid)
        if dd <= dd_cap:
            a, best_L, best_res = mid, mid, res
        else:
            b = mid
    return best_L, best_res


def levered_buy_and_hold(f: Features, L: float, carry: bool = True) -> pd.Series:
    """Buy and hold rebalanced daily to a constant exposure L (futures accounting: L x (price return + dividend - T-bill) + T-bill)."""
    r = f.df["Close"].pct_change().fillna(0.0)
    if carry and td.CARRY_MODE == "futures":
        rates = pd.Series(td.carry_rates_for(f.dates), index=f.dates)
        days = pd.Series(np.r_[0.0, np.diff(f.dates.values).astype("timedelta64[D]").astype(float)], index=f.dates)
        div = td.CARRY_DIVIDEND_YIELD.get(f.asset, 0.0)
        ret = L * (r + (div - rates) * days / 365.0) + rates * days / 365.0
    else:
        ret = L * r
    return (1 + ret).cumprod() * CAPITAL


def calibrate_buy_and_hold(f: Features, dd_cap: float):
    lo, hi = 0.05, 8.0

    def dd(L):
        eq = levered_buy_and_hold(f, L)
        return -float((eq / eq.cummax() - 1).min() * 100)
    if dd(hi) <= dd_cap:
        return hi, levered_buy_and_hold(f, hi)
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        if dd(mid) <= dd_cap:
            lo = mid
        else:
            hi = mid
    return lo, levered_buy_and_hold(f, lo)


def walk_forward_leverage(factory, params: dict, df: pd.DataFrame, asset: str, dd_cap: float, min_train_years: int = 10, first_oos: str = None,
                          max_gap: float = None) -> dict:
    """Out-of-sample test of "the most CAGR inside a drawdown cap" for a design with fixed structure. For each calendar year Y after the first
    `min_train_years`: the leverage L is chosen using ONLY the years before Y (gap-stressed, the cap must hold there), the design runs through year Y with
    that L, and Y's daily returns are kept. The kept years are chained into one out-of-sample curve. Returns its stats, the cap breach if any, and the
    leverage used each year."""
    years = sorted(set(df.index.year))
    start_year = years[0] + min_train_years
    if first_oos:
        start_year = max(start_year, pd.Timestamp(first_oos).year)
    pieces, Ls = [], {}
    gap_L = None
    if max_gap:
        # a gap of max_gap in one day at the design's largest exposure (cap x L) costs about max_gap x that; keep it inside the drawdown cap whatever the history shows
        d1 = factory(1.0, **params)
        gap_L = (dd_cap / 100.0 / max_gap) / max(d1.cap, 1e-9)
    for y in years:
        if y < start_year:
            continue
        train = df.loc[:f"{y - 1}-12-31"]
        upto = df.loc[:f"{y}-12-31"]
        if len(train) < 252 * min_train_years // 2 or len(upto.loc[f"{y}-01-01":]) < 20:
            continue
        L, _ = calibrate(lambda L: factory(L, **params), stress_gaps(Features(train, asset)), dd_cap)
        if not (L == L):
            L = 0.1
        if gap_L is not None:
            L = min(L, gap_L)
        Ls[y] = L
        res = run_design(factory(L, **params), Features(upto, asset))
        eq = res.equity
        prev = eq.loc[:f"{y - 1}-12-31"]
        r = eq.pct_change().loc[f"{y}-01-01":f"{y}-12-31"]
        if len(prev):
            r.iloc[0] = eq.loc[f"{y}-01-01":].iloc[0] / prev.iloc[-1] - 1
        pieces.append(r)
    oos = pd.concat(pieces)
    curve = (1 + oos).cumprod() * CAPITAL
    s = curve_stats(curve)
    s["worst_year_L"] = max(Ls.values()) if Ls else np.nan
    s["mean_L"] = float(np.mean(list(Ls.values()))) if Ls else np.nan
    s["years"] = len(Ls)
    s["breach"] = max(0.0, -s["MaxDD"] - dd_cap)
    return {"stats": s, "curve": curve, "L": Ls}


def curve_stats(eq: pd.Series, a=None, b=None) -> dict:
    eq = (eq.loc[a:b] if a else eq).dropna()
    s = td.performance_stats(eq, float(eq.iloc[0]))
    return {"CAGR": s["CAGR"] * 100, "MaxDD": s["Max Drawdown"] * 100, "Sharpe": s["Sharpe"], "Calmar": s["Calmar"]}


def eras(f: Features, k: int = 3):
    """k equal-length slices of the market's years, as (label, start, end)."""
    idx = f.dates
    cuts = [idx[int(len(idx) * j / k)] for j in range(k)] + [idx[-1]]
    out = []
    for j in range(k):
        a, b = cuts[j], cuts[j + 1]
        out.append((f"{a.year}-{b.year}", a, b))
    return out


# ----------------------------------------------------------------------------
# comparing against a baseline
# ----------------------------------------------------------------------------
def annual_log_returns(eq: pd.Series) -> pd.Series:
    last = eq.resample("YE").last()
    first = pd.Series([eq.iloc[0]], index=[eq.index[0] - pd.Timedelta(days=1)])
    s = pd.concat([first, last]).sort_index()
    return np.log(s).diff().dropna()


def paired_delta(eq_a: pd.Series, eq_b: pd.Series, block: int = 3, n_boot: int = 4000, seed: int = 7) -> dict:
    """Candidate a against baseline b over their common years: the mean yearly log-return difference (about the CAGR gap), its t-statistic, a
    block-bootstrap 90% interval and the share of resamples above zero. Years are resampled in blocks (neighbouring years are not independent)."""
    ra, rb = annual_log_returns(eq_a), annual_log_returns(eq_b)
    return paired_delta_from_logs(ra, rb, block, n_boot, seed)


def paired_delta_from_logs(ra: pd.Series, rb: pd.Series, block: int = 3, n_boot: int = 4000, seed: int = 7) -> dict:
    common = ra.index.intersection(rb.index)
    d = (ra.loc[common] - rb.loc[common]).to_numpy()
    n = len(d)
    if n < 6:
        return {"years": n, "delta_cagr": np.nan, "t": np.nan, "lo90": np.nan, "hi90": np.nan, "p_pos": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    nblocks = int(np.ceil(n / block))
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, nblocks) if n >= block else rng.integers(0, n, nblocks)
        samp = np.concatenate([d[s:s + block] for s in starts])[:n]
        means[k] = samp.mean()
    se = means.std()
    return {"years": n, "delta_cagr": float((np.exp(d.mean()) - 1) * 100), "t": float(d.mean() / se) if se > 0 else np.nan,
            "lo90": float((np.exp(np.percentile(means, 5)) - 1) * 100), "hi90": float((np.exp(np.percentile(means, 95)) - 1) * 100),
            "p_pos": float((means > 0).mean())}


def t_threshold(n_trials: int) -> float:
    """How large a t-statistic the best of n null trials is expected to reach, sqrt(2 ln n): what a claim must beat after n looks."""
    return float(np.sqrt(2.0 * np.log(max(n_trials, 2))))


def _annual_from_row(row) -> pd.Series:
    return pd.Series({int(k): v for k, v in json.loads(row["annual"]).items()})


def gains_vs(df: pd.DataFrame, base: pd.DataFrame, n_boot: int = 800) -> pd.DataFrame:
    """Every row of `df` against the baseline row of the same market and drawdown cap: the CAGR gain (points), the paired t-statistic over the
    years, and how many of the market's equal-length eras improved. `base` has one row per (market, dd_cap)."""
    b_idx = base.set_index(["market", "dd_cap"])
    eracols = sorted(c for c in df.columns if c.startswith("era_"))
    rows = []
    for _, r in df.iterrows():
        b = b_idx.loc[(r["market"], r["dd_cap"])]
        ecols = [c for c in eracols if pd.notna(r[c]) and pd.notna(b[c])]
        rows.append({"market": r["market"], "cap": r["dd_cap"], "design": r["design"], "params": r["params"], "L": r["L"], "CAGR": r["CAGR"], "base": b["CAGR"],
                     "d": r["CAGR"] - b["CAGR"], "t": paired_delta_from_logs(_annual_from_row(r), _annual_from_row(b), n_boot=n_boot)["t"],
                     "eras": int(sum(r[c] > b[c] for c in ecols)), "n_eras": len(ecols), "episodes": r["episodes"], "dd_clean": r["MaxDD_clean"] - b["MaxDD_clean"]})
    return pd.DataFrame(rows)


def keep_table(gains: pd.DataFrame, min_gain: float = 0.3, min_t: float = 1.0) -> pd.DataFrame:
    """One row per (design, params): the screening rule from research/hypotheses.md (mean gain above min_gain; S&P 500 and Nasdaq gains >= 0 at
    both caps; at least 4 of the 6 S&P 500 era comparisons improved; S&P 500 paired t at least min_t)."""
    out = []
    for (design, params), g in gains.groupby(["design", "params"]):
        def pick(m, cap):
            x = g[(g.market == m) & (g.cap == cap)].d
            return x.mean() if len(x) else np.nan
        sp = g[g.market == "SP500"]
        out.append({"design": design, "params": params, "mean_d": g.d.mean(), "SP50": pick("SP500", 50.0), "SP70": pick("SP500", 70.0),
                    "NQ50": pick("Nasdaq", 50.0), "NQ70": pick("Nasdaq", 70.0), "GD50": pick("Gold", 50.0), "GD70": pick("Gold", 70.0),
                    "SP_t": sp.t.mean(), "SP_eras": int(sp.eras.sum()), "SP_eras_of": int(sp.n_eras.sum())})
    res = pd.DataFrame(out)
    res["keep"] = ((res.mean_d > min_gain) & (res.SP50 >= 0) & (res.SP70 >= 0) & (res.NQ50 >= 0) & (res.NQ70 >= 0) & (res.SP_eras >= 4) & (res.SP_t >= min_t))
    return res


# ----------------------------------------------------------------------------
# trade analytics
# ----------------------------------------------------------------------------
def analytics(res: LabResult) -> dict:
    """The trade-level autopsy: how the money is made and where it leaks."""
    f = res.f
    ep = res.episodes()
    out = {"summary": res.summary()}
    if ep.empty:
        return out
    c = f.c
    win, los = ep[ep.ret > 0], ep[ep.ret <= 0]
    out["winners"] = {"n": len(win), "avg_ret": win.ret.mean() if len(win) else np.nan, "avg_mfe": win["mfe"].mean() if len(win) else np.nan,
                      "kept_of_peak_pct": float(win.ret.sum() / win.mfe.sum() * 100) if len(win) and win.mfe.sum() > 0 else np.nan}
    out["losers"] = {"n": len(los), "avg_ret": los.ret.mean() if len(los) else np.nan, "avg_mae": los.mae.mean() if len(los) else np.nan,
                     "within_3pct": int((los.ret > -3).sum())}
    out["short_trades"] = {"under_10_bars": int((ep.bars < 10).sum()), "total_ret_pts": float(ep.ret[ep.bars < 10].sum())}
    big = ep[ep.ret > 20]
    out["concentration"] = {"trades_over_20pct": len(big),
                            "share_of_winner_profit_pct": float(big.ret.sum() / win.ret.sum() * 100) if len(win) and win.ret.sum() > 0 else np.nan}
    held = (res.exposure.shift(1).fillna(0) > 0).to_numpy()
    r = pd.Series(c).pct_change().fillna(0).to_numpy()
    yrs_in, yrs_out = held.sum() / 252, (~held).sum() / 252
    out["timing"] = {"in_pct": float(held.mean() * 100),
                     "market_in_ann_pct": float((np.exp(np.log1p(r[held]).sum() / yrs_in) - 1) * 100) if yrs_in > 0 else np.nan,
                     "market_out_ann_pct": float((np.exp(np.log1p(r[~held]).sum() / yrs_out) - 1) * 100) if yrs_out > 0 else np.nan}
    rr = res.equity.pct_change().fillna(0).to_numpy()
    up, dn = r > 0, r < 0
    out["capture"] = {"up_pct": float(rr[up].mean() / r[up].mean() * 100) if r[up].mean() else np.nan,
                      "down_pct": float(rr[dn].mean() / r[dn].mean() * 100) if r[dn].mean() else np.nan}
    ep = ep.sort_values("entry_i").reset_index(drop=True)
    gaps = []
    for k in range(1, len(ep)):
        px_out = f.o[min(ep.loc[k - 1, "exit_i"], f.n - 1)]
        px_in = f.o[ep.loc[k, "entry_i"]]
        gaps.append(px_in / px_out - 1)
    out["whipsaw_tax_pct"] = float((np.prod(1 + np.array(gaps)) - 1) * 100) if gaps else 0.0
    return out


# ----------------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------------
def ledger_append(rows: list):
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER, mode="a", header=not os.path.exists(LEDGER), index=False)


def ledger_trials() -> int:
    """Number of distinct trials in the ledger (a run repeated by accident counts once); this feeds the multiple-testing threshold."""
    if not os.path.exists(LEDGER):
        return 0
    df = pd.read_csv(LEDGER, usecols=lambda c: c in ("stage", "design", "params", "market", "period", "stress", "dd_cap"))
    return int(len(df.drop_duplicates()))


def ledger_read() -> pd.DataFrame:
    return pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame()


# ----------------------------------------------------------------------------
# a job = one design on one market under the drawdown caps; the parallel runner
# ----------------------------------------------------------------------------
_REGISTRY = {}


def register(name):
    """Decorator: register a design factory `factory(L, **params) -> Design` under a name (so jobs can be sent to worker processes by name)."""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def run_job(job: dict) -> list:
    """job: {"design": name, "params": {...}, "market": name, "period": "design"|"sealed"|"full", "dd_caps": (50, 70), "stress": bool, "stage": str}.
    Returns one row per drawdown cap, including the per-year log returns needed for paired comparisons."""
    import lab_designs   # noqa: F401  (registers the factories in this process)
    factory = _REGISTRY[job["design"]]
    f_clean = market_features(job["market"], job.get("period", "design"))
    use_stress = job.get("stress", True)
    f_cal = stress_gaps(f_clean) if use_stress else f_clean     # leverage is chosen so the cap holds on THIS path
    rows = []
    for cap in job.get("dd_caps", DD_CAPS):
        t0 = time.time()
        L, res = calibrate(lambda L: factory(L, **job.get("params", {})), f_cal, cap)
        sm = res.summary()
        ann = annual_log_returns(res.equity)
        clean = run_design(factory(L if L == L else 0.1, **job.get("params", {})), f_clean).stats() if use_stress else sm
        row = {"stage": job.get("stage", ""), "design": job["design"], "params": json.dumps(job.get("params", {}), sort_keys=True, default=str),
               "market": job["market"], "period": job.get("period", "design"), "stress": bool(use_stress), "dd_cap": cap, "L": L,
               "seconds": round(time.time() - t0, 2),
               **{k: sm.get(k) for k in ("CAGR", "MaxDD", "Sharpe", "Calmar", "in_mkt", "avg_expo", "episodes", "win_rate", "avg_win", "avg_loss",
                                         "short_trades_pct", "top10_share")},
               "CAGR_clean": clean["CAGR"], "MaxDD_clean": clean["MaxDD"],
               "annual": json.dumps({int(k.year): float(v) for k, v in ann.items()})}
        for label, a, b in eras(f_cal):
            row[f"era_{label}"] = round(res.stats(a, b)["CAGR"], 2)
        rows.append(row)
    return rows


def _init_worker(markets, carry):
    configure(markets, carry)


def run_jobs(jobs: list, n_workers: int = None, markets=None, carry: bool = True, ledger: bool = True) -> pd.DataFrame:
    """Run jobs in parallel; append every row to the ledger; return them as a DataFrame."""
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    out = []
    if n_workers == 1 or len(jobs) == 1:
        configure(markets, carry)
        for j in jobs:
            out.extend(run_job(j))
    else:
        with mp.Pool(processes=min(n_workers, len(jobs)), initializer=_init_worker, initargs=(markets, carry)) as pool:
            for rows in pool.imap_unordered(run_job, jobs, chunksize=1):
                out.extend(rows)
    df = pd.DataFrame(out)
    if ledger and len(df):
        ledger_append(out)
    return df
