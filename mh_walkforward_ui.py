"""Lean Streamlit app for the monthly-high strategy (the Strategy Lab's "Monthly high" mode / turtle_monthly_high.pine): a BACKTEST and a WALK-FORWARD, each with an equity chart, the positions on
the price charts and Monte Carlo. The strategy's numbers are shared between the two and can be copied, pasted, or sent from a walk-forward window into the backtest.
Run:  .venv\\Scripts\\python.exe -m streamlit run mh_walkforward_ui.py   (or double-click run_mh_walkforward.bat)
Fast because: the heavy work lives in mh_wf_core.py / mh_wf_charts.py (imported once per server process, not on every click), prices come from a disk cache instead of Yahoo, the walk-forward runs
in a background thread while the page stays usable, and only the view you are looking at is drawn."""
import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import mh_wf_charts as charts
import mh_wf_core as core
import monte_carlo

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LWC = True
except Exception:                                                  # the chart component is optional
    LWC = False

st.set_page_config(page_title="Monthly-high lab", page_icon=":material/timeline:", layout="wide")
ss = st.session_state

# ------------------------------------------------------------------------------------------------------------------------------------ settings sent from elsewhere (before the widgets exist)
pending = ss.pop("pending_values", None)
if pending:
    for key, value in pending.items():
        ss["p_v_risk" if key == "risk_per_trade_pct" else f"p_v_{key}"] = int(value) if isinstance(core.DEFAULTS.get(key), int) else float(value)
if "pending_mode" in ss:
    ss["mode"] = ss.pop("pending_mode")
if ss.pop("pending_clear_paste", False):
    ss["paste_text"] = ""

st.title("Monthly-high lab")
st.caption("Entry: close above its EMA and the first close above a monthly high (plus a buffer) that is min..max days old. Adds: every further such signal. Stop: ATR x multiplier under the newest unit, moved up by "
           "every add. Take profit: each unit at its entry + multiplier x ATR. Nothing else. Backtest it, or walk-forward it: re-optimize on a rolling window of the past and trade the winner on the next, unseen stretch.")


def fmt_pct(v, digits=1):
    return f"{v * 100:.{digits}f}%" if v is not None and np.isfinite(v) else "n/a"


def fmt_usd(v):
    return "n/a" if v is None or not np.isfinite(v) else (f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")


def is_dark() -> bool:
    try:
        return getattr(st.context.theme, "type", None) == "dark"
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def prices_for(asset: str) -> pd.DataFrame:
    return core.load_prices([asset])[asset]


# ------------------------------------------------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Markets")
    all_markets = list(core.ALL_TICKERS)
    markets = st.multiselect("Markets", all_markets, default=all_markets, key="w_markets", label_visibility="collapsed")
    st.subheader("Account")
    capital = st.number_input("Account size ($)", 1_000.0, 1_000_000_000.0, 100_000.0, 10_000.0, key="w_capital")
    costs = st.selectbox("Trading costs", [core.COST_IBKR, core.COST_FLAT, core.COST_NONE], key="w_costs",
                         help="Interactive Brokers: per-contract fees and whole micro contracts where IBKR has them (standard contracts otherwise; a contract that is too big for the account is skipped). "
                              "Flat %: the engine's own slippage, spread and commission with fractional sizes. The same choices as the Strategy Lab's Fees tab.")
    margin_calls = st.checkbox("Model margin calls", value=False, key="w_margin", disabled=costs != core.COST_IBKR)
    exposure = st.number_input("Max notional per market (% of equity)", 50, 2000, 300, 50, key="w_exp", help="New units are skipped above this.")
    max_risk = st.number_input("Max open risk of the whole account (%)", 1.0, 100.0, 20.0, 1.0, key="w_risk", help="New units are skipped above this (the sum of every open unit's risk at its stop).")
    st.subheader("Prices")
    age = core.cache_age_hours(markets) if markets else float("inf")
    st.caption("Saved on disk: " + ("not yet for every market (the first run downloads them)" if not np.isfinite(age) else f"oldest {age:.1f} h old (refreshed after {core.CACHE_HOURS} h)"))
    refresh = st.checkbox("Download fresh prices on the next run", value=False, key="w_refresh")

# ------------------------------------------------------------------------------------------------------------------------------------ the strategy (shared by the backtest and the walk-forward)
for key, _, _, default, default_text, _ in core.PARAMS:
    ss.setdefault(f"p_v_{key}", default)
    ss.setdefault(f"p_s_{key}", True)
    ss.setdefault(f"p_r_{key}", default_text)
ss.setdefault("p_v_risk", core.RISK_DEFAULT)
with st.expander("The strategy: numbers, what the walk-forward searches, copy / paste", expanded=True):
    st.caption("'Value' is what the backtest uses, and what the walk-forward holds a setting at when it is not ticked. 'Search it' and 'Values to try' only matter to the walk-forward.")
    head = st.columns([2.6, 0.7, 1.1, 2.6])
    head[1].caption("Search it")
    head[2].caption("Value")
    head[3].caption("Values to try")
    last_group = None
    for key, label, group, default, default_text, lowest in core.PARAMS:
        if group != last_group:
            st.markdown(f"**{group}**")
            last_group = group
        c0, c1, c2, c3 = st.columns([2.6, 0.7, 1.1, 2.6], vertical_alignment="center")
        c0.write(label)
        searched_here = c1.checkbox("Search", key=f"p_s_{key}", label_visibility="collapsed")
        if isinstance(default, int):
            c2.number_input("Value", min_value=int(lowest), max_value=5000, step=1, key=f"p_v_{key}", label_visibility="collapsed")
        else:
            c2.number_input("Value", min_value=float(lowest), max_value=200.0, step=0.25, format="%g", key=f"p_v_{key}", label_visibility="collapsed")
        c3.text_input("Values to try", key=f"p_r_{key}", label_visibility="collapsed", disabled=not searched_here)
    c0, c1 = st.columns([2.6, 1.8], vertical_alignment="center")
    c0.write("**Position size**: risk per unit (% of the account) - needed to trade, not a strategy rule. No limit on the number of units.")
    c1.number_input("Risk per unit (%)", min_value=0.05, max_value=10.0, step=0.25, format="%g", key="p_v_risk", label_visibility="collapsed")
    values = {key: ss[f"p_v_{key}"] for key, *_ in core.PARAMS}
    values["risk_per_trade_pct"] = float(ss["p_v_risk"])
    values = {k: (int(v) if isinstance(core.DEFAULTS.get(k), int) else float(v)) for k, v in values.items()}
    searched = {key: bool(ss[f"p_s_{key}"]) for key, *_ in core.PARAMS}
    candidates = {key: ss[f"p_r_{key}"] for key, *_ in core.PARAMS}
    _, _, _, setting_errors = core.build_search(values, searched, candidates, exposure)
    for message in setting_errors:
        st.warning(message)
    st.markdown("**Copy / paste the settings**")
    k1, k2 = st.columns(2)
    k1.caption("Copy this line (the copy button is at its top right); paste it in the box on the right, here or in another session.")
    k1.code(core.format_settings(values), language="json", wrap_lines=True)
    k1.caption("The same numbers in the words of the TradingView script's inputs:")
    k1.code(core.format_for_tradingview(values), language=None, wrap_lines=True)
    k2.text_area("Paste settings", key="paste_text", height=110, placeholder='{"mh_sma_period": 200, "mh_min_age": 20, ...}   or   EMA = 200, min age = 20, max age = 365, buffer = 0.5, '
                                                                          'SL ATR = 185, SL x = 6, TP ATR = 185, TP x = 30, risk = 1')
    if k2.button("Apply pasted settings", key="paste_apply"):
        found, problems = core.parse_settings(ss.get("paste_text", ""))
        for p in problems:
            k2.warning(p)
        if found:
            ss["pending_values"] = found
            ss["pending_clear_paste"] = True
            st.rerun()

mode = st.radio("Mode", ["Backtest", "Walk-forward"], horizontal=True, key="mode", label_visibility="collapsed")
running = core.is_running()


# ------------------------------------------------------------------------------------------------------------------------------------ shared pieces
def stat_tiles(stats, bh_stats):
    def delta(key, fmt):
        if bh_stats is None or not (np.isfinite(stats[key]) and np.isfinite(bh_stats[key])):
            return None
        return fmt(stats[key] - bh_stats[key]) + " vs B&H"
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("CAGR", fmt_pct(stats["CAGR"], 1), delta=delta("CAGR", lambda d: f"{d * 100:+.1f} pts"), border=True)
    t2.metric("Max drawdown", fmt_pct(stats["Max Drawdown"], 1), delta=delta("Max Drawdown", lambda d: f"{d * 100:+.1f} pts"), border=True)
    t3.metric("Sharpe", f"{stats['Sharpe']:.2f}", delta=delta("Sharpe", lambda d: f"{d:+.2f}"), border=True)
    t4.metric("Final equity", fmt_usd(stats["Final Equity"]), delta=delta("Final Equity", lambda d: f"{'+' if d >= 0 else '-'}${abs(d):,.0f}"), border=True)


def equity_view(curve, bh, name, key, csv_name):
    frame = pd.DataFrame({name: curve})
    if bh is not None:
        frame["Equal-weight buy & hold"] = bh
    weekly = frame.resample("W").last().dropna(how="all")
    weekly.index.name = "date"
    long = weekly.reset_index().melt("date", var_name="series", value_name="equity")
    log_scale = st.toggle("Log scale", value=False, key=f"{key}_log")
    st.altair_chart(alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None), y=alt.Y("equity:Q", title="Account value ($)", scale=alt.Scale(type="log" if log_scale else "linear", zero=False)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")), tooltip=["date:T", "series:N", alt.Tooltip("equity:Q", format=",.0f")]), width="stretch")
    under = ((curve / curve.cummax() - 1) * 100).resample("W").min().rename("Drawdown %")
    under.index.name = "date"
    st.caption("Drawdown of the strategy (%)")
    st.altair_chart(alt.Chart(under.reset_index()).mark_area(opacity=0.5, color="#d62728").encode(
        x=alt.X("date:T", title=None), y=alt.Y("Drawdown %:Q", title=None), tooltip=["date:T", alt.Tooltip("Drawdown %:Q", format=".1f")]).properties(height=150), width="stretch")
    st.download_button("Download the equity curve (CSV)", frame.to_csv().encode(), csv_name, "text/csv", key=f"{key}_dl")


def positions_view(chart_data, key, run_id, note):
    """The trades on one market's price chart: entry / add / take-profit / stop markers, the stepped position boxes and a line at every monthly high."""
    assets = charts.markets_with_trades(chart_data)
    if not assets:
        st.info("No trades were captured for any market.")
        return
    if not LWC:
        st.error("The chart component is not installed: pip install streamlit-lightweight-charts")
        return
    st.caption(note)
    c1, c2, c3, c4 = st.columns([1.4, 1.6, 1, 1.2], vertical_alignment="bottom")
    asset = c1.selectbox("Market", assets, key=f"{key}_asset")
    years = c2.number_input("Years of boxes and lines to draw (0 = all)", 0, 40, 3, key=f"{key}_years",
                            help="Each trade and monthly high is its own series on the chart; drawing 25 years of them is slow, so by default only the last years are.")
    show_boxes = c3.checkbox("Position boxes", value=True, key=f"{key}_boxes")
    show_lines = c4.checkbox("Monthly-high lines", value=True, key=f"{key}_lines")
    df = prices_for(asset)
    memo = ss.setdefault("pos_memo", {})
    ck = (run_id, key, asset, int(years), show_boxes, show_lines)
    if ck not in memo:
        if len(memo) > 12:
            memo.clear()
        trades, exits, wins = charts.asset_trades(chart_data, asset)
        since = None if not years else df.index[-1] - pd.DateOffset(years=int(years))
        memo[ck] = {"markers": charts.markers_for(trades, exits), "boxes": charts.stepped_boxes(df, trades, since) if show_boxes else [],
                    "lines": charts.monthly_high_lines(df, wins, since) if show_lines else [], "table": charts.trade_table(trades), "candles": charts.candles(df)}
    d = memo[ck]
    options = {"height": 650, "layout": charts.chart_layout(is_dark()), "timeScale": {"borderColor": "rgba(197, 203, 206, 0.8)", "timeVisible": False},
               "rightPriceScale": {"borderColor": "rgba(197, 203, 206, 0.8)"}, "grid": {"vertLines": {"color": "rgba(197, 203, 206, 0.3)"}, "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}}}
    series = [{"type": "Candlestick", "data": d["candles"], "options": {"upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False, "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"},
               "markers": d["markers"]}] + d["boxes"] + d["lines"]
    renderLightweightCharts([{"chart": options, "series": series}], key=f"{key}_chart_{asset}_{int(years)}_{int(show_boxes)}_{int(show_lines)}")
    if len(d["table"]):
        st.caption(f"Closed trades in {asset}")
        st.dataframe(d["table"], hide_index=True, width="stretch")


def _years(curve) -> float:
    try:
        y = (curve.index[-1] - curve.index[0]).days / 365.25
        return y if y > 0 else None
    except Exception:
        return None


def mc_series_view(curve, key):
    """Block Monte Carlo on the daily equity curve (blocks of days keep streaks and overlapping positions intact)."""
    returns = monte_carlo.equity_returns(curve)
    if len(returns) < 60:
        st.caption("Not enough daily history for a Monte Carlo test.")
        return
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Test", list(monte_carlo.SERIES_METHODS), format_func=monte_carlo.SERIES_METHODS.get, key=f"{key}_method")
    n_sims = c2.number_input("Simulations", 200, 5000, 1000, 200, key=f"{key}_n")
    block = c3.number_input("Block length (trading days)", 5, 504, 126, 1, key=f"{key}_block",
                            help="Consecutive days kept together. Trend following loses money in long stretches, so short blocks understate drawdowns; 126 days (about 6 months) matched the real drawdown best in the Lab's tests.")
    d1, d2, d3 = st.columns(3)
    skip = d1.slider("Share of blocks skipped", 0.02, 0.5, 0.1, 0.02, key=f"{key}_skip", disabled=method != "skip")
    drift = d2.checkbox("Allow for uncertainty in the true average return", value=True, key=f"{key}_drift", disabled=method == "shuffle",
                        help="Without it every simulated path assumes this history's average return is the true one, which makes the range of final results too narrow.")
    dd_limit = d3.slider("Count a path as a deep drawdown when worse than (%)", -90, -10, -50, 5, key=f"{key}_dd")
    st.caption({"shuffle": "Cuts the daily equity curve into blocks and puts them in a random order. The final result is the same in every shuffle, only the drawdown spreads.",
                "bootstrap": "Draws blocks at random WITH replacement: the same kind of trading in a luckier or unluckier mix. Shows the range of outcomes and the chance of losing money.",
                "skip": "Keeps the order but replaces a share of the blocks with flat stretches (missed periods). Shows how much depends on a few stretches."}[method])
    sig = (key, method, int(n_sims), int(block), float(skip), bool(drift), len(curve), float(curve.iloc[-1]))
    memo = ss.setdefault("mc_memo", {})
    if sig not in memo:
        if len(memo) > 20:
            memo.clear()
        memo[sig] = monte_carlo.simulate_series(returns, method, int(n_sims), block=int(block), skip_fraction=float(skip), years=_years(curve),
                                                drift_uncertainty=monte_carlo.DEFAULT_DRIFT_UNCERTAINTY if (drift and method != "shuffle") else 0.0)
    mc = memo[sig]
    s = monte_carlo.summarize(mc, dd_limit / 100)
    a1, a2, a3 = st.columns(3)
    a1.metric("Actual max drawdown", f"{mc['actual_max_dd'] * 100:.1f}%", border=True)
    a2.metric("Simulated median max DD", f"{s['dd_p50'] * 100:.1f}%", border=True, help=f"5th to 95th percentile: {s['dd_p5'] * 100:.1f}% to {s['dd_p95'] * 100:.1f}%.")
    a3.metric("Actual result's percentile", f"{mc['actual_max_dd_percentile']:.0f}%", border=True,
              help="Where the real drawdown ranks among the simulated ones: 50% is typical, 90%+ means the real path was an unusually bad draw, under 10% an unusually good one.")
    b1, b2, b3 = st.columns(3)
    b1.metric("Actual final result", f"{mc['actual_final_multiple']:.2f}x", border=True)
    b2.metric("Simulated median final", f"{s['final_p50']:.2f}x", border=True, help=f"5th to 95th percentile: {s['final_p5']:.2f}x to {s['final_p95']:.2f}x of the starting capital.")
    b3.metric("Chance of losing money", f"{s['p_loss'] * 100:.1f}%", border=True)
    e1, e2 = st.columns(2)
    e1.metric(f"Chance of a drawdown worse than {dd_limit}%", f"{s['p_dd_worse'] * 100:.1f}%", border=True)
    e2.metric("Chance of losing 90% or more", f"{s['p_ruin'] * 100:.1f}%", border=True)
    if "underwater_days" in mc:
        u1, u2 = st.columns(2)
        u1.metric("Actual longest time under water", f"{mc['actual_underwater_days'] / 252:.1f} years", border=True)
        u2.metric("Simulated median longest time under water", f"{np.median(mc['underwater_days']) / 252:.1f} years", border=True,
                  help=f"5th to 95th percentile: {np.percentile(mc['underwater_days'], 5) / 252:.1f} to {np.percentile(mc['underwater_days'], 95) / 252:.1f} years.")
    if "cagr" in mc and np.isfinite(mc["cagr"]).any():
        c5, c50, c95 = np.nanpercentile(mc["cagr"], [5, 50, 95]) * 100
        st.caption(f"Simulated CAGR: median {c50:.1f}% (5th to 95th percentile {c5:.1f}% to {c95:.1f}%)")
    h1, h2 = st.columns(2)
    with h1:
        st.caption("Max drawdown across simulations (%)")
        counts, edges = np.histogram(mc["max_dd"] * 100, bins=30)
        st.bar_chart(pd.Series(counts, index=[f"{edges[i]:.1f}" for i in range(len(counts))]))
    with h2:
        st.caption("Final result across simulations (x starting capital)")
        if np.ptp(mc["final_multiple"]) < 1e-9:
            st.caption("Identical in every shuffle: nothing to plot.")
        else:
            counts, edges = np.histogram(mc["final_multiple"], bins=30)
            st.bar_chart(pd.Series(counts, index=[f"{edges[i]:.2f}" for i in range(len(counts))]))


def mc_windows_view(windows: pd.DataFrame):
    """Bootstrap over the walk-forward windows: is the average out-of-sample result real?"""
    ok = windows[windows["status"] == "ok"] if len(windows) else windows
    if len(ok) < 3:
        st.caption("Needs at least 3 windows with an out-of-sample result.")
        return
    st.markdown("**Across windows: is the average out-of-sample result real?**")
    st.caption("Resamples the windows' own out-of-sample results 5,000 times, in runs of 3 neighbouring windows (good and bad market stretches span several windows, so single windows are not "
               "independent), and looks at the average each time. If a good share of resamples average zero or less, the positive average could be luck of which windows were in the run.")
    cols = st.columns(2)
    for col, (label, column, as_growth) in zip(cols, (("Sharpe", "OOS Sharpe", False), ("CAGR (compounded)", "OOS CAGR %", True))):
        vals = [float(v) for v in ok[column].dropna()]
        if as_growth:
            vals = [float(np.log1p(max(v / 100.0, -0.999999))) for v in vals]
        bs = monte_carlo.bootstrap_mean(vals, block=3)
        if bs is None:
            continue
        if as_growth:
            bs = dict(bs, mean=float(np.expm1(bs["mean"])), p5=float(np.expm1(bs["p5"])), p95=float(np.expm1(bs["p95"])))
        fmt = (lambda v: f"{v * 100:.2f}%") if as_growth else (lambda v: f"{v:.2f}")
        with col:
            st.metric(f"Mean out-of-sample {label}", fmt(bs["mean"]), border=True, help=f"Over {bs['n']} windows; 90% interval from resampling: {fmt(bs['p5'])} to {fmt(bs['p95'])}.")
            st.metric(f"Chance the true mean {label} is zero or less", f"{bs['p_le_zero'] * 100:.1f}%", border=True)
            st.caption(f"90% interval: {fmt(bs['p5'])} to {fmt(bs['p95'])}")


# ------------------------------------------------------------------------------------------------------------------------------------ BACKTEST
def backtest_mode():
    st.subheader("Backtest")
    b1, b2, b3 = st.columns([1.2, 1.7, 2.6], vertical_alignment="bottom")
    start = b1.date_input("Test from", value=dt.date(2005, 1, 1), min_value=dt.date(1990, 1, 1), key="bt_start", help="Trading starts on this date; the indicators use all the history before it.")
    go = b2.button("Run backtest", type="primary", icon=":material/play_arrow:", key="bt_run", disabled=running or not markets)
    if running:
        b3.caption("A walk-forward is running: the backtest waits for it.")
    if go:
        with st.spinner("Running the backtest..."):
            ss["bt_result"] = core.run_backtest({"markets": list(markets), "capital": float(capital), "costs": costs, "margin_calls": bool(margin_calls), "exposure_pct": float(exposure),
                                                  "max_risk_pct": float(max_risk), "start": str(start), "refresh": bool(refresh), "values": dict(values)})
    res = ss.get("bt_result")
    if res is None:
        st.info("Set the numbers above and press Run backtest (about 5 seconds for 15 markets). The first run downloads prices; after that they come from disk.")
        return
    if "error" in res:
        st.error("The backtest failed:")
        st.code(res["error"], language=None)
        return
    s = res["settings"]
    st.caption(f"{len(s['markets'])} markets from {s['start']} · {s['costs']} · ${s['capital']:,.0f} account · {core.format_for_tradingview(s['values'])}")
    for note in res["notes"][:3]:
        st.caption(note)
    stat_tiles(res["stats"], res["bh_stats"])
    ep = res["episode_stats"]
    if ep:
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Trades", f"{ep['Episodes']:,}", border=True)
        u2.metric("Win rate", fmt_pct(ep["Win Rate"], 0), border=True)
        u3.metric("Profit factor", "n/a" if not np.isfinite(ep["Profit Factor"]) else f"{ep['Profit Factor']:.2f}", border=True)
        u4.metric("Average R", "n/a" if not np.isfinite(ep["Avg R"]) else f"{ep['Avg R']:.2f}", border=True)
    view = st.radio("View", ["Equity", "Markets", "Trades", "Positions", "Monte Carlo"], horizontal=True, key="bt_view", label_visibility="collapsed")
    if view == "Equity":
        equity_view(res["curve"], res["bh"], "Strategy", "bt_eq", "mh_backtest_equity.csv")
    elif view == "Markets":
        st.dataframe(res["assets"], hide_index=True, width="stretch")
        st.caption("Profit and loss by market (the trades' own results, after costs).")
        st.bar_chart(res["assets"].set_index("market")["P&L $"])
        st.caption("Profit and loss by way of exit ($): scale_out = take-profit, SL = stop loss, end_of_data = still open on the last day")
        st.dataframe(pd.DataFrame([{"Exit": k, "P&L $": round(float(v), 0)} for k, v in res["exit_pnl"].items()]), hide_index=True, width="stretch")
    elif view == "Trades":
        if len(res["episodes"]):
            st.dataframe(res["episodes"], width="stretch")
            st.download_button("Download the trades (CSV)", res["episodes"].to_csv(index=False).encode(), "mh_backtest_trades.csv", "text/csv", key="bt_trades_dl")
        else:
            st.caption("No trades.")
    elif view == "Positions":
        positions_view(res["chart_data"], "bt_pos", id(res), "Every trade of the backtest on the market's price history. The monthly-high lines use the numbers above.")
    else:
        mc_series_view(res["curve"], "bt_mc")


# ------------------------------------------------------------------------------------------------------------------------------------ WALK-FORWARD
def settings_from_walkforward(res, choice: str) -> dict:
    """The numbers to take from a finished walk-forward: one window's, the latest window's, the median or the most chosen value of each setting."""
    params = {i: p for i, p in res["window_params"].items() if p}
    if choice.startswith("Window"):
        p = params.get(int(choice.split()[-1])) or {}
        out = {key: p[key] for key, *_ in core.PARAMS if key in p}
    else:
        table = pd.DataFrame(list(params.values()))
        out = {}
        for key, *_ in core.PARAMS:
            if key not in table.columns:
                continue
            col = table[key].dropna()
            if len(col) == 0:
                continue
            out[key] = col.iloc[-1] if choice == "Latest window" else (col.median() if choice == "Median of the windows" else col.value_counts().index[0])
    out = {k: (int(round(float(v))) if isinstance(core.DEFAULTS.get(k), int) else float(v)) for k, v in out.items()}
    out["risk_per_trade_pct"] = float(res["settings"]["values"]["risk_per_trade_pct"])
    return out


def walkforward_mode():
    st.subheader("Walk-forward")
    w1, w2, w3, w4 = st.columns(4)
    train_years = w1.number_input("Train years", 1, 20, 4, 1, key="w_train", help="Each window searches the best settings on this many past years ...")
    test_years = w2.number_input("Test years", 1, 10, 1, 1, key="w_test", help="... and trades them on the next stretch of this many years, which the search never saw.")
    trials = w3.number_input("Trials per window", 5, 2000, 40, 5, key="w_trials", help="How many settings combinations each window tries. More = slower and a better search; 20-60 is a quick look, 150+ is thorough.")
    rank_metric = w4.selectbox("Rank by", ["sharpe", "calmar", "cagr", "total_return"], key="w_rank", help="The number that picks each window's winner.")
    x1, x2, x3, x4 = st.columns(4)
    search_method = x1.selectbox("Search method", ["optuna", "random"], key="w_method", help="Optuna learns from the trials so far (better on the same budget); random is blind.")
    anchored = x2.checkbox("Expanding train window", value=False, key="w_anchored", help="Off: the train window slides. On: it always starts at the earliest data.")
    warm_start = x3.checkbox("Fast simulation start", value=True, key="w_warm", help="Simulates only from 3 years before each window instead of from the first bar of the data: much faster, same result.")
    pick_middle = x4.checkbox("Pick the middle of the top N", value=False, key="w_middle", help="Instead of the single best trial, take the one in the middle of the N best (less likely to be a lucky outlier).")
    y1, y2, y3, _ = st.columns(4)
    top_n = y1.number_input("N", 3, 100, 10, 1, key="w_topn", disabled=not pick_middle)
    workers = y2.number_input("Worker processes (0 = automatic)", 0, 64, 0, 1, key="w_workers", help="Windows run in parallel, one per process.")
    seed = y3.number_input("Random seed", 0, 100000, 42, 1, key="w_seed")
    b1, b2 = st.columns([1, 3])
    go = b1.button("Run walk-forward", type="primary", icon=":material/play_arrow:", key="w_run", disabled=running or not markets or bool(setting_errors))
    if not markets:
        b2.caption("Pick at least one market in the sidebar.")
    elif sum(searched.values()) == 0:
        b2.caption("No setting is ticked for search: every window will use the same numbers.")
    if go:
        problem = core.start_run({
            "markets": list(markets), "capital": float(capital), "costs": costs, "margin_calls": bool(margin_calls), "exposure_pct": float(exposure), "max_risk_pct": float(max_risk),
            "refresh": bool(refresh), "train_years": int(train_years), "test_years": int(test_years), "anchored": bool(anchored), "trials": int(trials), "rank_metric": rank_metric,
            "search_method": search_method, "pick_middle": bool(pick_middle), "top_n": int(top_n), "warm_start": bool(warm_start), "workers": int(workers), "seed": int(seed),
            "values": dict(values), "searched": dict(searched), "candidates": dict(candidates)})
        if problem:
            st.warning(problem)
        st.rerun()

    @st.fragment(run_every=1.0 if running else None)
    def live_panel():
        if not core.is_running():
            st.rerun()                                      # finished (or failed) since the last full draw: draw the results
            return
        p = core.progress()
        fraction = p["done"] / p["total"] if p["total"] else 0.0
        eta = f" · about {p['eta']:.0f} s left" if p["eta"] else ""
        st.progress(min(fraction, 1.0), text=f"{p['phase']} · {p['elapsed']:.0f} s{eta}")
        if st.button("Stop", key="w_stop", icon=":material/stop:", help="Ends the run; the windows already finished are kept."):
            core.cancel_run()
        with st.expander("Live log"):
            st.code(core.RUN["buf"].getvalue()[-3000:], language=None)

    if running:
        live_panel()
    if core.RUN["error"]:
        st.error("The run failed:")
        st.code(core.RUN["error"], language=None)
    res = core.RUN["result"]
    if res is None or running:
        if res is None and not running and not core.RUN["error"]:
            st.info("Set the strategy and the walk-forward above, then press Run. The first run downloads prices (about a minute); after that they come from disk.")
        return
    s = res["settings"]
    elapsed = (core.RUN["t1"] or 0) - (core.RUN["t0"] or 0)
    st.markdown("#### Result")
    st.caption(f"{len(s['markets'])} markets · {s['train_years']}y train / {s['test_years']}y test · {s['trials']} trials per window ({s['search_method']}, ranked by {s['rank_metric']}) · "
               f"{s['costs']} · ${s['capital']:,.0f} account · finished in {elapsed:.0f} s")
    stats, bh_stats, curve, bh, summary = res["stats"], res["bh_stats"], res["curve"], res["bh"], res["summary"]
    if stats is None:
        st.warning("No window produced an out-of-sample result (see the log).")
    else:
        stat_tiles(stats, bh_stats)
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Windows OK", f"{summary.get('windows_ok', 0)}/{summary.get('windows_total', 0)}", border=True, help="Windows that produced an out-of-sample result.")
        u2.metric("Positive windows", f"{summary.get('positive_oos_windows', 0)}/{summary.get('windows_ok', 0)}", border=True, help="Windows with a positive out-of-sample Sharpe.")
        u3.metric("WF efficiency", "n/a" if not np.isfinite(summary.get("wfe_cagr", np.nan)) else f"{summary['wfe_cagr']:.2f}", border=True,
                  help="Compounded out-of-sample CAGR divided by the average in-sample CAGR: how much of what the search found survived on unseen data. About 0.5 or more is reasonable; near 0 or below means the search found noise.")
        u4.metric("Sharpe in → out", f"{summary.get('mean_is_sharpe', float('nan')):.2f} → {summary.get('mean_oos_sharpe', float('nan')):.2f}", border=True,
                  help="Mean Sharpe in-sample, then out-of-sample. A big drop is the overfitting signature.")
    view = st.radio("View", ["Equity", "Windows", "Settings", "Positions", "Monte Carlo", "Log"], horizontal=True, key="wf_view", label_visibility="collapsed")
    if view == "Equity":
        if curve is not None:
            equity_view(curve, bh, "Strategy (out of sample)", "wf_eq", "mh_walkforward_oos_equity.csv")
    elif view == "Windows":
        if len(res["windows"]):
            st.dataframe(res["windows"], hide_index=True, width="stretch",
                         column_config={"IS Sharpe": st.column_config.NumberColumn(format="%.2f"), "OOS Sharpe": st.column_config.NumberColumn(format="%.2f"),
                                        "OOS CAGR %": st.column_config.NumberColumn(format="%.1f"), "OOS max DD %": st.column_config.NumberColumn(format="%.1f")})
            st.download_button("Download the windows (CSV)", res["windows"].to_csv(index=False).encode(), "mh_walkforward_windows.csv", "text/csv", key="wf_win_dl")
    elif view == "Settings":
        if len(res["picks"]):
            st.caption("The settings the search chose, window by window. Settings that keep jumping around from window to window carry little information; ones that settle are worth a look.")
            st.dataframe(res["picks"], hide_index=True, width="stretch")
            st.markdown("**Take settings from this walk-forward**")
            good = [i for i, p in res["window_params"].items() if p]
            choice = st.selectbox("Which", ["Latest window", "Median of the windows", "Most chosen value of each setting"] + [f"Window {i}" for i in good], key="wf_take")
            chosen = settings_from_walkforward(res, choice)
            t1, t2 = st.columns(2)
            t1.code(core.format_settings(chosen), language="json", wrap_lines=True)
            t2.code(core.format_for_tradingview(chosen), language=None, wrap_lines=True)
            if st.button("Use these numbers in the Backtest", key="wf_send", icon=":material/arrow_forward:", help="Puts them in the strategy panel at the top and switches to the Backtest."):
                ss["pending_values"] = chosen
                ss["pending_mode"] = "Backtest"
                st.rerun()
        else:
            st.caption("No window picked settings.")
    elif view == "Positions":
        positions_view(res["chart_data"], "wf_pos", id(res), "Every window's own out-of-sample trades on the market's price history. Each window traded its OWN chosen settings, so a change in style partway "
                                                           "through is expected; each window's monthly-high lines use that window's numbers.")
    elif view == "Monte Carlo":
        mc_windows_view(res["windows"])
        st.markdown("**Across days: the stitched out-of-sample equity curve**")
        if curve is not None:
            mc_series_view(curve, "wf_mc")
    else:
        st.code(res["log"][-12000:], language=None)


if mode == "Backtest":
    backtest_mode()
else:
    walkforward_mode()
