"""
Turtle Strategy Lab -- a Streamlit UI over turtle_dual_system.py, optimize_turtle_dual_system.py,
walk_forward_turtle.py, and regime_diagnostic.py.

Run with:  streamlit run turtle_ui.py
(needs all four of those files in the same folder, plus `pip install streamlit` if you don't
have it yet)

Parameter changes made in the UI apply for that session only, UNLESS you click "Save to
turtle_dual_system.py" on the Backtest tab, which writes your edited S1/S2 config directly
into the SYSTEMS block of that file (with a timestamped backup made first).
"""

import contextlib
import io
import multiprocessing as mp
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LIGHTWEIGHT_CHARTS_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_CHARTS_AVAILABLE = False

import turtle_dual_system as td
import optimize_turtle_dual_system as opt
import walk_forward_turtle as wf
import regime_diagnostic as rd
import ibkr_fees
import brokers
import carry_data
import monte_carlo
import robustness
import ui_glass
if not hasattr(ibkr_fees, "remember_engine_defaults"):
    # a long-running Streamlit server can still hold an older copy of this module in memory
    import importlib
    ibkr_fees = importlib.reload(ibkr_fees)

st.set_page_config(page_title="Turtle Strategy Lab", page_icon=":material/show_chart:", layout="wide")
if not (hasattr(td, "MH_CLEAN_MODE") and "MH_CLEAN_MODE" in opt._PORTFOLIO_CONFIG_ATTRS):
    # this server process loaded the engine before the monthly-high mode existed and keeps that old copy in memory (Streamlit re-runs this file but does not reload the engine modules)
    st.error("The engine files changed (monthly-high mode) but this running app still has the old copy in memory. Stop the app and start it again "
             "(close the terminal running `streamlit run turtle_ui.py` or press Ctrl+C there, then run it again).", icon=":material/restart_alt:")
    st.stop()
ui_glass.inject()
st.title("Turtle Strategy Lab")
st.caption("Long-only trend strategies: backtest them, search for better settings and check them out of sample, all on one shared account.")
with st.expander("New here? Quick start", icon=":material/rocket_launch:"):
    st.markdown(
        "1. Pick assets in the sidebar on the left. Under them, the **Setup** tabs hold the "
        "search settings, the per-market risk and the **Fees** every tab uses.\n"
        "2. Go to **Backtest**, leave the defaults, and click **Run backtest** to see how the "
        "strategy would have performed on one shared account. The results open on an **Overview** "
        "scorecard (against buy & hold), with the years, assets, trades, costs, robustness checks "
        "and a day-by-day replay on their own tabs next to it.\n"
        "3. Open **Trade chart** to see exactly where it entered/exited on the real price chart.\n"
        "4. **Optimizer** searches for better parameters; **Walk-forward** is the more honest "
        "version of that -- what you'd *actually* have gotten by re-optimizing on a schedule, "
        "not the best possible lookback-fitted numbers.\n"
        "5. **Regime diagnostic** shows what kind of market conditions this strategy tends to do "
        "best in, independent of anything above.\n\n"
        "Every number on every tab has a hover `(?)` or an expandable :material/info: section with the full "
        "explanation -- this is just the map, not the whole manual.")

# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

def ticker_picker(key_prefix: str) -> dict:
    """Checkbox list built from the FULL original ticker set (captured once at startup into
    st.session_state, never from td.TICKERS directly). This matters: several tabs used to do
    `td.TICKERS = chosen_tickers` to run against a subset, which permanently SHRANK the shared
    module-level dict every other tab (and this function) reads from -- any ticker you happened
    to have unchecked for one run would vanish everywhere, permanently, for the rest of the
    session. Reading from a preserved copy instead means nothing here ever mutates the full
    available list."""
    st.caption("Assets (edit the TICKERS dict in turtle_dual_system.py to add more choices)")
    all_tickers = st.session_state.all_tickers
    all_col, none_col = st.columns(2)
    # set before the checkboxes below exist in this pass, so the new state is what they render with
    with all_col:
        if st.button("All", key=f"{key_prefix}_all", width="stretch"):
            for name in all_tickers:
                st.session_state[f"{key_prefix}_tick_{name}"] = True
    with none_col:
        if st.button("None", key=f"{key_prefix}_none", width="stretch"):
            for name in all_tickers:
                st.session_state[f"{key_prefix}_tick_{name}"] = False
    chosen = {}
    with st.container(height=190, border=True, key=f"card_{key_prefix}"):   # compact, scrolls when there are many assets
        for name, symbol in all_tickers.items():
            if st.checkbox(name, value=True, key=f"{key_prefix}_tick_{name}", help=symbol):
                chosen[name] = symbol
    st.caption(f"{len(chosen)}/{len(all_tickers)} selected")
    return chosen


def chart_layout() -> dict:
    """Colours for the lightweight price charts, which do not pick up Streamlit's theme on their own."""
    dark = ui_glass.is_dark()
    # transparent when the see-through look is on, so the glass shows behind the candles
    background = "rgba(0, 0, 0, 0)" if ui_glass.ENABLED else ("#0e131b" if dark else "#ffffff")
    return {"textColor": "#e5e9f0" if dark else "#0f172a", "background": {"type": "solid", "color": background}}


# Fallback values used ONLY when a key is missing from the loaded turtle_dual_system.py --
# e.g. if your local file predates a feature that's since been added. The UI will show these
# defaults for the missing field instead of crashing; it won't silently invent values for
# anything that's actually present in your file.
_FALLBACK_DEFAULTS = {
    "breakout_lookback": 20, "exit_lookback": 15, "n_atr_period": 20, "entry_buffer_n_mult": 0.0,
    "risk_per_trade_pct": 2.0, "pyramid_spacing_n_mult": 1.0, "max_units_per_trend": 4, "stop_n_mult": 2.0,
    "pyramid_method": "spacing", "pyramid_breakout_lookback": 5, "swing_lookback": 15,
    "use_fixed_stop": True, "ratchet_stop_on_pyramid": True,
    "use_average_entry_stop": False,
    "tp_atr_period": 20, "tp_n_mult": 1.0, "use_scale_out": True, "use_time_stop": False,
    "time_stop_days": 10, "time_stop_min_r": 0.25, "use_n_day_low_trailing_stop": True,
    "trailing_stop_lookback_days": 5, "use_breakeven_stop": False, "breakeven_trigger_r": 1.0,
    "use_chandelier_exit": False, "chandelier_atr_mult": 3.0, "use_ema_filter": False, "ema_period": 200,
    "use_giveback_stop": False, "giveback_fraction": 0.4, "giveback_min_profit_r": 1.0,
    "use_cooldown": False, "cooldown_days": 10, "use_pyramid_adx_filter": False, "pyramid_max_adx": 35.0,
    "use_regime_scale_out": False, "regime_scale_out_adx_threshold": 35.0, "regime_scale_out_fraction": 0.5,
    "adx_period": 14, "use_choppiness_filter": False, "choppiness_period": 14, "choppiness_max": 38.2,
    "require_all_time_high": False,
    "use_ma_regime": False, "ma_regime_period": 200, "ma_regime_band_pct": 0.0,
    "use_ema_entry": False, "ema_entry_period": 50, "ema_entry_mode": "cross", "use_ema_exit": False, "ema_exit_period": 20,
    "use_monthly_high_entry": False, "mh_min_age": 20, "mh_max_age": 365, "mh_sma_period": 200, "mh_ma_type": "ema",
    "use_peak_trail": False, "peak_trail_atr_mult": 8.0,
}


def _g(p: dict, key: str):
    """Get a param value, falling back to the known default if this key is missing from the
    loaded file (e.g. an older turtle_dual_system.py that predates some feature)."""
    return p.get(key, _FALLBACK_DEFAULTS[key])


def system_param_editor(label: str, defaults: dict, key_prefix: str) -> dict:
    """Renders a form for one system's full parameter dict, organized into clearly labeled
    categories (tabs) instead of one long flat list. Returns the edited dict."""
    p = dict(defaults)
    missing = [k for k in _FALLBACK_DEFAULTS if k not in defaults]
    with st.expander(f"{label} parameters", expanded=False):
        if missing:
            st.caption(f":orange[Your turtle_dual_system.py is missing {len(missing)} newer setting(s) "
                       f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}) -- showing defaults "
                       f"for those. Re-download the latest turtle_dual_system.py to get them for real.]")

        tab_entry, tab_size, tab_pyr, tab_stop, tab_tp, tab_filt = st.tabs(
            ["Entry signal", "Position sizing", "Pyramiding", "Stops", "Take-profit / time exits", "Entry filters"])

        with tab_entry:
            entry_labels = {"breakout": "Entry 1 · N-day-high breakout (default)", "ema": "Entry 2 · EMA break",
                            "monthly": "Entry 3 · Monthly-high breakout", "ma": "Moving-average regime (older option)"}
            current_entry = ("monthly" if _g(p, "use_monthly_high_entry") else "ema" if _g(p, "use_ema_entry")
                             else "ma" if _g(p, "use_ma_regime") else "breakout")
            entry_type = st.radio(
                "Entry type", list(entry_labels), index=list(entry_labels).index(current_entry), format_func=entry_labels.get,
                key=f"{key_prefix}_etype", horizontal=True,
                help="How this system decides to buy. Entry 1: close above the highest close of the last N days (+ the entry buffer). "
                     "Entry 2: close breaks above an EMA. Entry 3: buys when a monthly high breaks out (each completed month's highest high, between a month and a year old; each breaks out once), "
                     "only while the close is above its 200-day EMA, and adds a unit every time another one breaks out. The moving-average regime buys while the close is above its average. One of them at a time.")
            p["use_monthly_high_entry"] = entry_type == "monthly"
            p["use_ema_entry"] = entry_type == "ema"
            p["use_ma_regime"] = entry_type == "ma"
            # always present in the dict, so the optimizer / walk-forward can pin them even if the entry type was never chosen
            for _k in ("ema_entry_period", "ema_entry_mode", "ema_exit_period", "mh_min_age", "mh_max_age", "mh_sma_period", "mh_ma_type"):
                p.setdefault(_k, _g(p, _k))
            c1, c2 = st.columns(2)
            with c1:
                p["breakout_lookback"] = st.number_input("Breakout lookback (days)", 2, 500, int(_g(p, "breakout_lookback")), key=f"{key_prefix}_bl",
                                                         disabled=entry_type != "breakout")
                p["exit_lookback"] = st.number_input("Exit lookback (days)", 2, 500, int(_g(p, "exit_lookback")), key=f"{key_prefix}_el",
                                                       disabled=bool(_g(p, "use_ema_exit")) or entry_type == "ma")
            with c2:
                p["n_atr_period"] = st.number_input("N (ATR) period", 2, 500, int(_g(p, "n_atr_period")), key=f"{key_prefix}_natr")
                p["entry_buffer_n_mult"] = st.number_input("Entry buffer (x N)", 0.0, 10.0, float(_g(p, "entry_buffer_n_mult")), 0.05,
                                                              key=f"{key_prefix}_buf", disabled=entry_type in ("ema", "ma"))
            if entry_type == "ema":
                e1, e2 = st.columns(2)
                with e1:
                    p["ema_entry_period"] = st.number_input("EMA entry length (days)", 2, 1000, int(_g(p, "ema_entry_period")), key=f"{key_prefix}_emap")
                with e2:
                    ema_modes = {"cross": "Cross: only the day it moves above", "above": "Above: whenever flat and above"}
                    current_mode = _g(p, "ema_entry_mode") if _g(p, "ema_entry_mode") in ema_modes else "cross"
                    p["ema_entry_mode"] = st.selectbox("EMA entry type", list(ema_modes), index=list(ema_modes).index(current_mode),
                                                       format_func=ema_modes.get, key=f"{key_prefix}_emam",
                                                       help="Cross waits for the next cross after any exit. Above buys straight back after an exit while the close is still above the EMA.")
                st.caption("The breakout lookback and the entry buffer are not used. Research so far: no robust edge (it won on the 15 original markets and lost on 27 unseen ones).")
            elif entry_type == "monthly":
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    p["mh_min_age"] = st.number_input("High at least this old (days)", 1, 1000, int(_g(p, "mh_min_age")), key=f"{key_prefix}_mhmin",
                                                      help="A monthly high only counts once the bar that set it is this many daily bars old. A monthly high that breaks out while it is still younger than this is used up without a trade.")
                with m2:
                    p["mh_max_age"] = st.number_input("High at most this old (days)", 2, 2000, int(_g(p, "mh_max_age")), key=f"{key_prefix}_mhmax",
                                                      help="A monthly high older than this many daily bars no longer counts. Between the minimum and the maximum age every monthly high is its own level.")
                with m3:
                    p["mh_sma_period"] = st.number_input("Close above its average (days, 0 = off)", 0, 1000, int(_g(p, "mh_sma_period")), key=f"{key_prefix}_mhsma",
                                                         help="The close (the price of the entry and of every add) must be above this moving average. 0 switches the filter off.")
                with m4:
                    ma_types = ["ema", "sma"]
                    cur_ma = str(_g(p, "mh_ma_type")).lower() if str(_g(p, "mh_ma_type")).lower() in ma_types else "ema"
                    p["mh_ma_type"] = st.selectbox("Average type", ma_types, index=ma_types.index(cur_ma), format_func=str.upper, key=f"{key_prefix}_mhma",
                                                   help="EMA = exponential average (like a 200 EMA line on a chart), SMA = simple average.")
                if int(p["mh_min_age"]) >= int(p["mh_max_age"]):
                    st.caption(":red[The minimum age must be smaller than the maximum age, otherwise this entry never fires.]")
                st.caption("Every completed calendar month has a monthly high; it counts while it is between min-age and max-age days old (days = daily bars). Each monthly high breaks out "
                           "ONCE: the first close above (that high + the entry buffer). If the high is old enough and the close is above its average, that breakout is a BUY, and while the "
                           "position is open every further breakout of that kind adds a unit (every time a new entry would have happened; no spacing rule). Either way the high is then used up. "
                           "The breakout lookback is not used.")
            elif entry_type == "ma":
                m1, m2 = st.columns(2)
                with m1:
                    p["ma_regime_period"] = st.number_input("Average length (days)", 5, 1000, int(_g(p, "ma_regime_period")), key=f"{key_prefix}_marp")
                with m2:
                    p["ma_regime_band_pct"] = st.number_input("Band (% either side)", 0.0, 20.0, float(_g(p, "ma_regime_band_pct")), 0.5, key=f"{key_prefix}_marb")
                st.caption("Enter when the close is above its simple average (plus the band) and leave when it closes below it (minus the band); the breakout and exit "
                           "lookbacks and the entry buffer are ignored. Built for a long-only account in a market that drifts up.")
            p["use_ema_exit"] = st.checkbox(
                "EMA exit instead of the exit lookback", value=bool(_g(p, "use_ema_exit")) and entry_type != "ma",
                key=f"{key_prefix}_emxe", disabled=entry_type == "ma",
                help="Leave when the close falls below an exponential moving average instead of below the lowest low of the exit lookback. "
                     "Independent of the entry: it works with every entry type except the moving-average regime. The stop, trims and caps still apply. Off by default.")
            if p["use_ema_exit"]:
                p["ema_exit_period"] = st.number_input("EMA exit length (days)", 2, 1000, int(_g(p, "ema_exit_period")), key=f"{key_prefix}_emxp")
                st.caption("The exit lookback above is ignored while the EMA exit is on.")
            if entry_type in ("ema", "monthly") or p["use_ema_exit"]:
                st.caption("Optimizer and Walk-forward tabs: these settings are shared by S1 and S2 there (taken from S1's panel, or S2's when only S2 is on) and held "
                           "at the values here unless you tick them in the sidebar list to have them searched.")

        with tab_size:
            p["risk_per_trade_pct"] = st.number_input("Risk per trade (%)", 0.05, 20.0,
                                                         float(_g(p, "risk_per_trade_pct")), 0.05, key=f"{key_prefix}_risk")
            st.caption("Graduated risk schedules (risk_per_trade_pct_schedule) aren't editable here -- "
                       "edit turtle_dual_system.py directly for that, this form just passes it through unchanged.")

        with tab_pyr:
            c1, c2 = st.columns(2)
            with c1:
                if entry_type == "monthly":
                    st.info("Entry 3 is this system's entry: it ADDS a unit every time a new entry would have happened (a monthly high breaks out for the first time, with the close above "
                            "the average), whatever method is picked below. The spacing / breakout / swing settings are not used while Entry 3 is on.")
                pyramid_method_options = ["spacing", "breakout", "swing_high", "monthly_high"]
                current_method = _g(p, "pyramid_method")
                p["pyramid_method"] = st.selectbox(
                    "Pyramid method", pyramid_method_options,
                    index=pyramid_method_options.index(current_method) if current_method in pyramid_method_options else 0,
                    key=f"{key_prefix}_pyrm")
                if p["pyramid_method"] == "breakout":
                    p["pyramid_breakout_lookback"] = st.number_input(
                        "Pyramid breakout lookback (days)", 1, 200,
                        int(_g(p, "pyramid_breakout_lookback")), key=f"{key_prefix}_pyrbl")
                    st.caption("Serves two roles: how many days define a 'fresh high' for an add, "
                               "AND the minimum days that must pass since the last add regardless "
                               "of price. The second part is what actually stops rapid-fire adds "
                               "during one uninterrupted thrust -- a longer lookback alone can't, "
                               "since every day of a sustained climb is trivially a fresh N-day "
                               "high no matter how large N is.")
                    p["pyramid_spacing_n_mult"] = _g(p, "pyramid_spacing_n_mult")   # unused but kept valid
                    p["swing_lookback"] = _g(p, "swing_lookback")   # unused but kept valid
                elif p["pyramid_method"] == "monthly_high":
                    p["pyramid_spacing_n_mult"] = _g(p, "pyramid_spacing_n_mult")   # unused but kept valid
                    p["pyramid_breakout_lookback"] = _g(p, "pyramid_breakout_lookback")   # unused but kept valid
                    p["swing_lookback"] = _g(p, "swing_lookback")   # unused but kept valid
                    if entry_type == "monthly":
                        st.caption("Adds a unit every time a new entry would have happened (a monthly high breaks out for the first time, with the close above the average), "
                                   "using the Entry 3 settings. No spacing rule. The max units below still cap it.")
                    else:
                        st.caption("Adds a unit every time a monthly high breaks out for the first time with the close above the average (no spacing rule). "
                                   "A completed month's high counts while it is between min-age and max-age days old (days = daily bars). "
                                   "The entry type of this system is not Entry 3, so set those here.")
                        p["mh_min_age"] = st.number_input("High at least this old (days)", 1, 1000, int(_g(p, "mh_min_age")), key=f"{key_prefix}_pmhmin")
                        p["mh_max_age"] = st.number_input("High at most this old (days)", 2, 2000, int(_g(p, "mh_max_age")), key=f"{key_prefix}_pmhmax")
                        p["mh_sma_period"] = st.number_input("Close above its average (days, 0 = off)", 0, 1000, int(_g(p, "mh_sma_period")), key=f"{key_prefix}_pmhsma")
                        pm_types = ["ema", "sma"]
                        pm_cur = str(_g(p, "mh_ma_type")).lower() if str(_g(p, "mh_ma_type")).lower() in pm_types else "ema"
                        p["mh_ma_type"] = st.selectbox("Average type", pm_types, index=pm_types.index(pm_cur), format_func=str.upper, key=f"{key_prefix}_pmhma")
                        if int(p["mh_min_age"]) >= int(p["mh_max_age"]):
                            st.caption(":red[The minimum age must be smaller than the maximum age, otherwise nothing is ever added.]")
                elif p["pyramid_method"] == "swing_high":
                    p["swing_lookback"] = st.number_input(
                        "Swing lookback (bars on EACH side to confirm a pivot)", 2, 100,
                        int(_g(p, "swing_lookback")), key=f"{key_prefix}_swl")
                    st.caption("The sparsest, most structural of the three methods -- built for "
                               "riding months-long trends. Requires a GENUINELY CONFIRMED swing "
                               "high (real pullback, then a push past the previous pivot) before "
                               "adding again, not just a rolling window. Bigger = requires a "
                               "bigger, more genuine pullback before the next add can even become "
                               "possible. A confirmed pivot can only be known ~lookback days after "
                               "it happens (that lag is the whole point, not a flaw) -- pairs "
                               "naturally with a slower entry/exit, since this won't fire often.")
                    p["pyramid_spacing_n_mult"] = _g(p, "pyramid_spacing_n_mult")   # unused but kept valid
                    p["pyramid_breakout_lookback"] = _g(p, "pyramid_breakout_lookback")   # unused but kept valid
                else:
                    p["pyramid_spacing_n_mult"] = st.number_input("Pyramid spacing (x N)", 0.05, 20.0,
                                                                    float(_g(p, "pyramid_spacing_n_mult")), 0.05, key=f"{key_prefix}_sp")
                    p["pyramid_breakout_lookback"] = _g(p, "pyramid_breakout_lookback")   # unused but kept valid
                    p["swing_lookback"] = _g(p, "swing_lookback")   # unused but kept valid
                p["max_units_per_trend"] = st.number_input("Max units per trend", 1, 20,
                                                             int(_g(p, "max_units_per_trend")), key=f"{key_prefix}_mu")
            with c2:
                p["use_pyramid_adx_filter"] = st.checkbox("Block pyramid adds above ADX threshold",
                                                             value=_g(p, "use_pyramid_adx_filter"), key=f"{key_prefix}_padx")
                if p["use_pyramid_adx_filter"]:
                    p["pyramid_max_adx"] = st.number_input("Max ADX to allow an add", 0.0, 100.0,
                                                              float(_g(p, "pyramid_max_adx")), 1.0, key=f"{key_prefix}_padxv")
                st.caption("ADX period for this filter is set in the Take-Profit / Time Exits tab -- "
                           "shared with the regime scale-out toggle there.")

        with tab_stop:
            c1, c2 = st.columns(2)
            with c1:
                p["use_fixed_stop"] = st.checkbox("Fixed stop (the core ratcheting SL)",
                                                     value=_g(p, "use_fixed_stop"), key=f"{key_prefix}_ufs")
                if p["use_fixed_stop"]:
                    p["stop_n_mult"] = st.number_input("Stop (x N)", 0.1, 20.0, float(_g(p, "stop_n_mult")), 0.1, key=f"{key_prefix}_stop")
                    p["ratchet_stop_on_pyramid"] = st.checkbox(
                        "Move stop on each pyramid add (off = stays at first entry's level)",
                        value=_g(p, "ratchet_stop_on_pyramid"), key=f"{key_prefix}_ratchet")
                    if p["ratchet_stop_on_pyramid"]:
                        p["use_average_entry_stop"] = st.checkbox(
                            "Anchor to AVERAGE entry (not just the latest unit)",
                            value=_g(p, "use_average_entry_stop"), key=f"{key_prefix}_avgstop")
                        if p["use_average_entry_stop"]:
                            st.caption("Tighten-only: stop = max(current stop, average entry - stop x N). "
                                       "Only moves up as the position's average cost basis improves.")
                        else:
                            st.caption("Unconditional SET on every add: stop = latest add's entry - stop x N, "
                                       "full stop, every time -- not a ratchet-up-only max(). If N widens "
                                       "between adds this can move the stop DOWN, by design: the stop always "
                                       "sits stop x N below the newest add, not 'never loosens.'")
                    else:
                        p["use_average_entry_stop"] = _g(p, "use_average_entry_stop")   # unused but kept valid
                else:
                    st.caption(":orange[With the fixed stop off, downside protection relies entirely on "
                               "whichever of trailing stop / chandelier / exit channel are still enabled.]")
                    p["stop_n_mult"] = _g(p, "stop_n_mult")
                    p["ratchet_stop_on_pyramid"] = _g(p, "ratchet_stop_on_pyramid")
                    p["use_average_entry_stop"] = _g(p, "use_average_entry_stop")
            with c2:
                p["use_n_day_low_trailing_stop"] = st.checkbox("N-day-low trailing stop",
                                                                  value=_g(p, "use_n_day_low_trailing_stop"), key=f"{key_prefix}_tr")
                if p["use_n_day_low_trailing_stop"]:
                    p["trailing_stop_lookback_days"] = st.number_input("Trailing lookback (days)", 1, 500,
                                                                          int(_g(p, "trailing_stop_lookback_days")), key=f"{key_prefix}_trlb")
                    st.caption("This is a SEPARATE setting from the channel exit's own 'Exit "
                               "lookback' (in the Entry Signal tab) -- easy to mix up since both "
                               "are N-day-low mechanisms. Changing this number does NOT change "
                               "when the channel exit fires; that's controlled by Exit lookback.")
                p["use_breakeven_stop"] = st.checkbox("Breakeven stop", value=_g(p, "use_breakeven_stop"), key=f"{key_prefix}_be")
                if p["use_breakeven_stop"]:
                    p["breakeven_trigger_r"] = st.number_input("Breakeven trigger (R)", 0.05, 20.0,
                                                                  float(_g(p, "breakeven_trigger_r")), 0.05, key=f"{key_prefix}_ber")
                p["use_chandelier_exit"] = st.checkbox("Chandelier exit", value=_g(p, "use_chandelier_exit"), key=f"{key_prefix}_ch")
                if p["use_chandelier_exit"]:
                    p["chandelier_atr_mult"] = st.number_input("Chandelier (x ATR)", 0.1, 20.0,
                                                                  float(_g(p, "chandelier_atr_mult")), 0.1, key=f"{key_prefix}_chm")
                p["use_peak_trail"] = st.checkbox(
                    "Peak-trail stop (resting)", value=_g(p, "use_peak_trail"), key=f"{key_prefix}_pkt",
                    help="After every close, the stop for the NEXT bar is set this many ATRs (N) under the highest high "
                         "since entry, and it only ever rises. If the next bar's low touches it the position is sold there "
                         "(or at the open if the market gapped below it). Unlike the Chandelier exit, which measures the "
                         "same bar's own high and range, this is an order that could really have been resting in the "
                         "market. EXPERIMENTAL: in the strategy research a trail of about 8 ATR (N period 20) on top of "
                         "the 250-day regime rule gained about 3 CAGR points at equal drawdown in one sealed-market test "
                         "(mostly Bitcoin, the S&P 500 and the Nasdaq; the median market gained about half a point), and "
                         "a second, independent test of a tiered version found no gain. Treat it as unproven.")
                if p["use_peak_trail"]:
                    p["peak_trail_atr_mult"] = st.number_input("Peak trail (x ATR)", 0.5, 40.0, float(_g(p, "peak_trail_atr_mult")), 0.5,
                                                               key=f"{key_prefix}_pktm")
                p["use_giveback_stop"] = st.checkbox("Per-unit profit give-back stop",
                                                        value=_g(p, "use_giveback_stop"), key=f"{key_prefix}_gb",
                                                        help="Every other stop here is shared across the whole "
                                                             "episode -- one level, applied identically to every "
                                                             "pyramided unit. This one is per-UNIT: once a unit's "
                                                             "own profit reaches the R threshold below, ITS OWN "
                                                             "stop trails at entry + fraction x (that unit's own "
                                                             "peak price since its own entry - entry). An early, "
                                                             "deeply profitable unit can give back a lot and still "
                                                             "hold; a late, barely-profitable unit gets protected "
                                                             "almost immediately -- instead of both dying at the "
                                                             "same shared price when the group as a whole reverses.")
                if p["use_giveback_stop"]:
                    p["giveback_fraction"] = st.number_input("Give-back fraction", 0.05, 0.95,
                                                                float(_g(p, "giveback_fraction")), 0.05, key=f"{key_prefix}_gbf")
                    p["giveback_min_profit_r"] = st.number_input("Activate above (R)", 0.05, 20.0,
                                                                     float(_g(p, "giveback_min_profit_r")), 0.05, key=f"{key_prefix}_gbr")
                else:
                    p["giveback_fraction"] = _g(p, "giveback_fraction")
                    p["giveback_min_profit_r"] = _g(p, "giveback_min_profit_r")
            st.caption("Note: the channel exit (close < N-day low) has no toggle -- it's always active, "
                       "the original Turtle trend-reversal exit. Its lookback is the 'Exit lookback' "
                       "field in the Entry Signal tab, NOT the trailing stop setting above.")

        with tab_tp:
            c1, c2 = st.columns(2)
            with c1:
                p["tp_atr_period"] = st.number_input("TP's own ATR period", 2, 500, int(_g(p, "tp_atr_period")), key=f"{key_prefix}_tpatr")
                p["tp_n_mult"] = st.number_input("TP (x its own N)", 0.1, 50.0, float(_g(p, "tp_n_mult")), 0.1, key=f"{key_prefix}_tpm")
                p["use_scale_out"] = st.checkbox("Scale-out (partial TP at fixed R levels)", value=_g(p, "use_scale_out"), key=f"{key_prefix}_so")
                p["use_time_stop"] = st.checkbox("Time stop", value=_g(p, "use_time_stop"), key=f"{key_prefix}_ts")
                if p["use_time_stop"]:
                    p["time_stop_days"] = st.number_input("Time stop days", 1, 500, int(_g(p, "time_stop_days")), key=f"{key_prefix}_tsd")
                    p["time_stop_min_r"] = st.number_input("Time stop min R", -5.0, 5.0, float(_g(p, "time_stop_min_r")), 0.05, key=f"{key_prefix}_tsr")
            with c2:
                p["use_regime_scale_out"] = st.checkbox("Regime-triggered scale-out (ADX spike)",
                                                           value=_g(p, "use_regime_scale_out"), key=f"{key_prefix}_rso")
                if p["use_regime_scale_out"]:
                    p["regime_scale_out_adx_threshold"] = st.number_input("ADX threshold", 0.0, 100.0,
                        float(_g(p, "regime_scale_out_adx_threshold")), 1.0, key=f"{key_prefix}_rsoadx")
                    p["regime_scale_out_fraction"] = st.number_input("Fraction to take", 0.01, 1.0,
                        float(_g(p, "regime_scale_out_fraction")), 0.05, key=f"{key_prefix}_rsofrac")
                p["adx_period"] = st.number_input("ADX period (shared: regime scale-out above + "
                                                    "pyramid ADX filter in the Pyramiding tab)", 2, 500,
                                                    int(_g(p, "adx_period")), key=f"{key_prefix}_adxp")

        with tab_filt:
            c1, c2 = st.columns(2)
            with c1:
                p["use_ema_filter"] = st.checkbox("EMA trend filter", value=_g(p, "use_ema_filter"), key=f"{key_prefix}_ema")
                if p["use_ema_filter"]:
                    p["ema_period"] = st.number_input("EMA period", 2, 1000, int(_g(p, "ema_period")), key=f"{key_prefix}_emap")
                p["use_choppiness_filter"] = st.checkbox("Choppiness Index filter (must be trending to enter)",
                                                            value=_g(p, "use_choppiness_filter"), key=f"{key_prefix}_chop")
                if p["use_choppiness_filter"]:
                    p["choppiness_period"] = st.number_input("Choppiness Index period", 2, 200,
                                                                int(_g(p, "choppiness_period")), key=f"{key_prefix}_chopp")
                    p["choppiness_max"] = st.number_input("Max Choppiness Index to allow entry (lower = stricter)",
                                                             0.0, 100.0, float(_g(p, "choppiness_max")), 0.5, key=f"{key_prefix}_chopm")
            with c2:
                p["require_all_time_high"] = st.checkbox("Require entry to be a genuine all-time high (not just N-day)",
                                                            value=_g(p, "require_all_time_high"), key=f"{key_prefix}_ath")
                p["use_cooldown"] = st.checkbox("Cooldown after a loss", value=_g(p, "use_cooldown"), key=f"{key_prefix}_cd")
                if p["use_cooldown"]:
                    p["cooldown_days"] = st.number_input("Cooldown days", 1, 500, int(_g(p, "cooldown_days")), key=f"{key_prefix}_cdd")

        # fields the editor doesn't expose a widget for -- pass through unchanged
        p.setdefault("risk_per_trade_pct_schedule", defaults.get("risk_per_trade_pct_schedule"))
        p.setdefault("scale_out_levels", defaults.get("scale_out_levels"))
    return p


def clear_widget_keys_with_prefix(prefix: str):
    """Deletes every st.session_state key starting with `prefix` -- needed whenever code (not
    the user) changes a parameter dict like s1_params/s2_params out from under an
    already-rendered form. Streamlit widgets remember their OWN value under their `key=`
    once rendered once, and silently ignore the `value=` argument on later reruns -- so just
    updating the underlying dict (e.g. auto-applying an optimizer's winning config) has NO
    effect on what the form actually displays unless the widgets' own keys are cleared too,
    forcing them to re-initialize fresh from the updated dict on the next render."""
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]


def build_systems_from_ui(s1_params: dict, s2_params: dict, enable_s1: bool, enable_s2: bool) -> list:
    systems = []
    if enable_s1:
        s1 = dict(s1_params)
        s1["name"] = "S1"
        systems.append(s1)
    if enable_s2:
        s2 = dict(s2_params)
        s2["name"] = "S2"
        systems.append(s2)
    return systems


# ----------------------------------------------------------------------------
# MONTHLY-HIGH STRATEGY ("clean" mode) -- picked on the Setup panel's "Monthly high" tab
# ----------------------------------------------------------------------------
# One system that does exactly four things and nothing else:
#   INITIAL ENTRY  close above its EMA, and the first close above a monthly high that is min..max days old
#   PYRAMIDING     one more unit on every further signal of that same entry
#   SL             stop = N x ATR(SL period) under the newest unit's entry; each add moves the stop up to the new unit's stop
#   TP             every unit closes completely at its entry + multiplier x ATR(TP period)
# The engine side is td.MH_CLEAN_MODE / td.MH_CLEAN_SYSTEM_PARAMS (turtle_dual_system.py); this is the UI side.
MH_MODE_STANDARD = "Standard: the S1 / S2 settings"
MH_MODE_CLEAN = "Monthly-high strategy: only entry, pyramiding, SL and TP"
# (key, label, group, default value, default candidate values)
MH_PARAMS = [
    ("mh_sma_period", "Price must be above this EMA (days)", "Initial entry and pyramiding", 200, "100, 150, 200, 250, 300"),
    ("mh_min_age", "Monthly high at least this many days old", "Initial entry and pyramiding", 20, "10, 15, 20, 30, 40, 60"),
    ("mh_max_age", "Monthly high at most this many days old", "Initial entry and pyramiding", 365, "120, 180, 250, 365, 500"),
    ("entry_buffer_n_mult", "Entry buffer: the close must be this many ATR above the monthly high", "Initial entry and pyramiding", 0.0, "0, 0.25, 0.5, 1, 1.5, 2"),
    ("n_atr_period", "SL ATR period (days)", "Stop loss", 40, "14, 20, 30, 40, 60, 100"),
    ("stop_n_mult", "SL ATR multiplier", "Stop loss", 3.0, "1.5, 2, 3, 4, 5, 6, 8"),
    ("tp_atr_period", "TP ATR period (days)", "Take profit", 20, "10, 14, 20, 30, 40, 60"),
    ("tp_n_mult", "TP ATR multiplier", "Take profit", 6.0, "2, 3, 4, 6, 8, 10, 15"),
]
MH_DEFAULTS = {key: default for key, _, _, default, _ in MH_PARAMS}
MH_MIN = {"entry_buffer_n_mult": 0.0}          # the lowest value a number box accepts (0.1 for the others)
MH_RISK_DEFAULT = 1.0       # position size, not a strategy rule: % of the account risked per unit
MH_UNIT_CAP = 999           # pyramiding has no cap of its own (the exposure and open-risk limits on the Backtest tab still apply)


def mh_mode_active() -> bool:
    return st.session_state.get("mh_mode") == MH_MODE_CLEAN


def mh_fixed_values() -> dict:
    """The value of every monthly-high setting as typed in the tab (its default until edited), plus the position size."""
    ss = st.session_state
    out = {}
    for key, _, _, default, _ in MH_PARAMS:
        value = ss.get(f"mh_v_{key}")
        out[key] = default if value is None else (int(value) if isinstance(default, int) else float(value))
    risk = ss.get("mh_v_risk_per_trade_pct")
    out["risk_per_trade_pct"] = MH_RISK_DEFAULT if risk is None else float(risk)
    out["max_units_per_trend"] = MH_UNIT_CAP
    return out


def build_clean_system(values: dict = None) -> dict:
    """The single system a plain backtest runs in the monthly-high mode (the optimizer builds the same one through opt.params_to_systems)."""
    values = values or mh_fixed_values()
    system = {**td.DEFAULT_SYSTEM_PARAMS, **values, **td.MH_CLEAN_SYSTEM_PARAMS}
    system.update({"name": "S1", "breakout_lookback": 20, "exit_lookback": 10})     # placeholders: the monthly-high level replaces the breakout, and there is no channel exit
    return system


def apply_mh_mode_to_engine():
    """Puts the engine flags in line with the strategy picked on the Monthly high tab. Call inside a snapshot / restore."""
    on = mh_mode_active()
    td.MH_CLEAN_MODE = on
    if on:
        opt.ENABLE_S1, opt.ENABLE_S2 = True, False       # one system only


def mh_summary_rows() -> list:
    values = mh_fixed_values()
    rows = []
    for key, label, group, _, _ in MH_PARAMS:
        rows.append({"Mechanic": group, "Setting": label, "Value": f"{values[key]:g}"})
    rows.append({"Mechanic": "Position size", "Setting": "Risk per unit (% of the account)", "Value": f"{values['risk_per_trade_pct']:g}"})
    return rows


def run_captured(fn, *args, **kwargs):
    """Runs fn, capturing everything it prints, returning (return_value, captured_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# Lock registry lives in ui_locks.py, NOT as a plain global here and NOT in st.session_state.
# It has to be a properly `import`-ed module: Streamlit re-execs turtle_ui.py's own module
# namespace FRESH on every rerun (confirmed empirically -- id(_run_captured_live_locks) differed
# between two successive button clicks in the same process even guarded by
# `if "x" not in globals()`, so that guard never actually found a pre-existing dict and silently
# handed back a brand-new, already-unlocked one every time). A separately imported module is
# cached in sys.modules and reused across reruns within the same process -- same guarantee td/
# opt/wf already rely on for their own snapshot/restore state. This lock is what makes
# run_captured_live() below able to detect and refuse a SECOND overlapping run: Streamlit can
# abort the SCRIPT EXECUTION that started a background thread when a new interaction comes in,
# but it has no way to kill the thread itself -- the orphaned thread keeps running to completion.
# Without a *working* lock, a second click while a run is still active spins up a SECOND thread
# that ALSO redirects the process-wide sys.stdout, and prints from both threads land in whichever
# buffer happens to be "current" at that instant -- confirmed directly: a user's walk-forward log
# came back with windows printed out of order and duplicated, which was two separate wf.main()
# runs' output interleaved into two different buffers, not a real bug in the (still perfectly
# sequential) window loop itself. st.session_state was deliberately NOT used for this flag either
# -- writing to it from a background thread is a known Streamlit footgun.
import ui_locks


def run_captured_live(fn, log_placeholder, progress_bar=None, run_key="default", on_complete=None,
                       status_placeholder=None, *args, **kwargs):
    """Like run_captured, but runs fn in a background thread and streams its captured stdout
    into log_placeholder every ~0.5s while it's still running, instead of the page just sitting
    on a spinner with zero visibility for however long the whole thing takes -- wf.main()/
    rd.main() are reused as their own standalone scripts (not refactored to report progress
    natively), so this reads their EXISTING print()-based progress output live rather than only
    after the function returns. If progress_bar is given, it also tries to parse a "WINDOW X/Y"
    line (walk_forward_turtle.py's own per-window progress print) out of the latest captured
    text to drive a real progress bar; harmless no-op for output that doesn't contain that.

    run_key: identifies which "slot" this call competes for (e.g. "walk_forward" vs
    "regime_diagnostic") -- see the lock registry comment above for why this exists. Returns
    (None, None) without running anything if a run under the SAME run_key is still active.

    on_complete: called from INSIDE the background thread, right before the lock is released --
    NOT from the caller's own script execution. A caller that mutates shared engine globals
    (td.TICKERS, wf.TRAIN_YEARS, etc.) before starting the run and wants them restored afterward
    must pass the restore as on_complete rather than putting it in its own try/finally: Streamlit
    can abort the caller's script execution (e.g. a later click, or the user pressing Stop) while
    this function's polling loop is still blocked, which does NOT directly kill the background
    thread. A restore in the caller's own finally would fire the moment ITS script gets aborted,
    yanking those globals out from under the still-running thread for every window/trial it
    hasn't reached yet. Routing the restore through on_complete ties it to when the actual
    computation finishes, not when the caller's script happens to stop running.

    CANCELLATION: fn is always called with a cancel_event=<threading.Event> kwarg (fn must accept
    it, currently only wf.main() does). Streamlit can't directly kill the background thread when
    the caller's script execution is aborted (Stop button or an overlapping click) -- so instead,
    the polling loop below is wrapped in try/finally and sets this SAME event the instant that
    abort happens (the finally block is the one piece of code Python still runs while the
    aborting exception propagates through). fn is expected to check cancel_event.is_set() at
    reasonable checkpoints (walk_forward_turtle.py checks between windows AND after every trial
    within a window's search, terminate()-ing its worker Pool immediately if set) and return
    early with whatever partial result it has, instead of running to full completion unsupervised.

    NOTE: contextlib.redirect_stdout swaps sys.stdout process-wide, not per-thread -- fine as long
    as the lock above actually prevents two calls under the same run_key from overlapping; still a
    real caveat for anything ELSE printing concurrently in the same process while this runs."""
    import threading
    import re
    lock = ui_locks.get_lock(run_key)
    if not lock.acquire(blocking=False):
        log_placeholder.warning(
            f"A previous run (key={run_key!r}) is still finishing in the background -- most likely "
            f"an earlier click that's still going. Wait for it to finish before starting another; "
            f"running two at once would corrupt both runs' progress output (they'd share the same "
            f"redirected console output).")
        return None, None

    buf = io.StringIO()
    result_holder = {}
    error_holder = {}
    # A plain threading.Event, not a multiprocessing one -- fn (currently always wf.main) runs in
    # THIS thread's background _run(), and even its own Pool-based trial search lives in the SAME
    # thread (only the individual trial bodies run in separate worker processes), so this is
    # sufficient to reach every cancellation checkpoint without cross-process signaling.
    cancel_event = threading.Event()

    def _run():
        try:
            with contextlib.redirect_stdout(buf):
                result_holder["value"] = fn(*args, cancel_event=cancel_event, **kwargs)
        except Exception as e:
            error_holder["error"] = e
        finally:
            # Both run from INSIDE the background thread, not the polling loop below -- this must
            # happen even if the script execution driving that loop gets aborted by a later rerun.
            # Restore before releasing the lock, so a subsequent run never sees a free lock but
            # still-mutated globals.
            if on_complete is not None:
                on_complete()
            lock.release()

    thread = threading.Thread(target=_run, daemon=True)
    started_at = time.time()
    thread.start()
    try:
        while thread.is_alive():
            text = buf.getvalue()
            log_placeholder.code(text[-6000:], language=None)
            if progress_bar is not None:
                matches = re.findall(r"WINDOW (\d+)/(\d+)", text)
                if matches:
                    cur, total = matches[-1]
                    progress_bar.progress(min(int(cur) / int(total), 1.0))
                    if status_placeholder is not None:
                        elapsed = time.time() - started_at
                        done = int(cur)
                        eta = f", about {elapsed / done * (int(total) - done):.0f}s left" if done else ""
                        status_placeholder.write(f"window {cur}/{total} -- {elapsed:.0f}s elapsed{eta}")
            time.sleep(0.5)
    finally:
        # Reached on normal completion too (harmless no-op there, the thread's already done) --
        # but critically, also reached if THIS script execution gets torn down while the loop is
        # still running: pressing Streamlit's own Stop button, or a later click starting a new
        # run, both abort the current script by raising inside whatever Streamlit call is next
        # (log_placeholder.code() above), which skips every line after it in this function EXCEPT
        # finally blocks. Without this, the background thread (and anything it's spawned, e.g. a
        # walk-forward window's worker Pool) would keep running for however long the rest of the
        # computation takes, with nothing left watching it and no way for the user to tell it to
        # stop -- confirmed directly as the exact behavior before this fix.
        cancel_event.set()
    thread.join()
    text = buf.getvalue()
    log_placeholder.code(text[-6000:], language=None)
    if progress_bar is not None:
        progress_bar.progress(1.0)
    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("value"), text


def show_yearly_win_loss(equity_curve: pd.Series, starting_capital: float):
    """Bar chart of each year's return (green=winning year, red=losing year), plus summary
    stats: how many winning vs losing years, and the average magnitude of each."""
    yearly = td.compute_period_returns(equity_curve, "YE", starting_capital)
    if yearly.empty:
        st.caption("Not enough history for a yearly breakdown.")
        return
    pct = yearly["return_%"] * 100
    winning = pct[pct > 0]
    losing = pct[pct < 0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Winning years", f"{len(winning)}/{len(pct)}", border=True)
    m2.metric("Losing years", f"{len(losing)}/{len(pct)}", border=True)
    m3.metric("Avg winning year", f"{winning.mean():.1f}%" if len(winning) else "n/a", border=True)
    m4.metric("Avg losing year", f"{losing.mean():.1f}%" if len(losing) else "n/a", border=True)

    chart_df = pd.DataFrame({
        "Year": yearly.index.year.astype(str),
        "Return %": pct.values,
        "Result": np.where(pct.values > 0, "Winning year", "Losing year"),
    })
    import altair as alt
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Year:N", sort=None),
        y=alt.Y("Return %:Q"),
        color=alt.Color("Result:N", scale=alt.Scale(
            domain=["Winning year", "Losing year"], range=["#2ca02c", "#d62728"])),
        tooltip=["Year", "Return %"],
    ).properties(height=250)
    st.altair_chart(chart, width="stretch")


def cost_figures(port: dict) -> tuple:
    """(the three costs as (label, $), their total, interest and carry earned, gross profit before costs and carry)."""
    items = [("Slippage + spread paid", port.get("total_slippage_spread_cost", 0.0)),
             ("Commissions paid", port.get("total_commission_cost", 0.0)),
             ("Holding / swap paid", port.get("total_holding_cost", 0.0))]
    total = sum(v for _, v in items)
    carry = port.get("total_carry", 0.0)
    gross = port["final_equity"] - port["starting_capital"] + total - carry   # what the trading made before any cost (and before interest/carry)
    return items, total, carry, gross


def asset_contribution(port: dict, name: str, trades: list) -> pd.Series:
    """What one asset's trades added to the SHARED account, in dollars, on the account's own dates: the profit of every closed unit from the day it closed (a take-profit counts the day it
    happens, not when the whole trade ends), plus the open profit of what is still held that day. Costs that belong to the account as a whole (interest, holding cost) are not in it.
    Position sizes in a shared account are set against the whole account's equity, so this is measured against the account's starting capital, never a per-asset slice of it."""
    index = port["equity_curve"].index
    realized = {}
    for t in trades:
        parts = t.get("unit_details")
        for part in (parts if parts else [t]):
            realized[part["exit_date"]] = realized.get(part["exit_date"], 0.0) + part["pnl"]
    out = pd.Series(0.0, index=index)
    if realized:
        booked = pd.Series(realized).sort_index().cumsum()
        booked.index = pd.DatetimeIndex(booked.index)
        out = booked.reindex(index.union(booked.index)).ffill().fillna(0.0).reindex(index)
    held = port.get("positions_history")
    if held is not None and len(held):
        open_pnl = held[held["asset"] == name].groupby("date")["unrealized"].sum()
        out = out + open_pnl.reindex(index).fillna(0.0)
    return out


def asset_share_curve(port: dict, contribution: pd.Series) -> pd.Series:
    """The account as if ONLY this asset's trades had counted: every day the asset's dollar contribution is taken as a share of the account's equity the day before, and those daily shares are
    compounded from the account's starting capital. A shared account has no per-asset capital, and measuring a late profit or loss against the starting capital (or an equal slice of it) gives
    nonsense once the account has grown (a -450% 'drawdown'); as a share of the equity of the day it stays what it is. CAGR, drawdown and Sharpe of this curve do not depend on the base."""
    shares = (contribution.diff() / port["equity_curve"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-1.0)
    return float(port["starting_capital"]) * (1.0 + shares).cumprod()


def bh_share_curve(port: dict, prices: pd.DataFrame, share_capital: float) -> pd.Series:
    """The reference for asset_share_curve: buy and hold of this one asset at its equal share of the account (share / starting capital of it, kept at that weight), on the account's dates,
    compounded the same way so the two lines and their statistics are comparable."""
    index = port["equity_curve"].index
    weight = share_capital / float(port["starting_capital"])
    daily = prices["Close"].pct_change().reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(port["starting_capital"]) * (1.0 + weight * daily).cumprod()


def show_scorecard(strategy_equity: pd.Series, bh_equity, starting_capital: float, trades: list = None, port: dict = None, equity_label: str = "Final equity"):
    """The one place the statistics live: four headline tiles (each against buy & hold), the strategy and buy & hold
    curves, and a single table grouped as returns, risk, risk-adjusted, trades and costs. bh_equity is None when there is
    no price history to compare with; trades adds the trade rows, port the cost rows (a single asset has trades only)."""
    strat = td.performance_stats(strategy_equity, starting_capital)
    bh = None
    if bh_equity is not None:
        bh_equity = bh_equity.reindex(strategy_equity.index).ffill()
        bh = td.performance_stats(bh_equity, starting_capital)

    def fmt_pct(v, digits=2):
        return f"{v * 100:.{digits}f}%" if np.isfinite(v) else "n/a"

    def fmt_num(v, digits=2):
        return f"{v:.{digits}f}" if np.isfinite(v) else "n/a"

    def fmt_usd(v):
        if not np.isfinite(v):
            return "n/a"
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

    def fmt_days(v):
        return f"{v:.0f} days" if np.isfinite(v) else "n/a"

    def vs(key, fmt):
        """A tile's delta against buy & hold; None (no delta shown) when there is nothing to compare with."""
        if bh is None or not (np.isfinite(strat[key]) and np.isfinite(bh[key])):
            return None
        return fmt(strat[key] - bh[key]) + " vs B&H"

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("CAGR", fmt_pct(strat["CAGR"]), delta=vs("CAGR", lambda d: f"{d * 100:+.2f} pts"), border=True)
    t2.metric("Max drawdown", fmt_pct(strat["Max Drawdown"]), delta=vs("Max Drawdown", lambda d: f"{d * 100:+.2f} pts"), border=True)
    t3.metric("Sharpe", fmt_num(strat["Sharpe"]), delta=vs("Sharpe", lambda d: f"{d:+.2f}"), border=True)
    t4.metric(equity_label, fmt_usd(strat["Final Equity"]),
              delta=vs("Final Equity", lambda d: f"{'+' if d >= 0 else '-'}${abs(d):,.0f}"), border=True)

    if bh_equity is not None:
        st.line_chart(pd.DataFrame({"Strategy": strategy_equity, "Buy & Hold": bh_equity}))
    else:
        st.line_chart(strategy_equity)

    groups = [
        ("Returns", [("CAGR", "CAGR", fmt_pct), ("Total return", "Total Return", lambda v: fmt_pct(v, 1)),
                     (equity_label, "Final Equity", fmt_usd), ("Best month", "Best Month", lambda v: fmt_pct(v, 1)),
                     ("Worst month", "Worst Month", lambda v: fmt_pct(v, 1))]),
        ("Risk", [("Max drawdown", "Max Drawdown", fmt_pct), ("Longest drawdown", "Max DD Duration (days)", fmt_days),
                  ("Annual volatility", "Ann. Vol", fmt_pct)]),
        ("Risk-adjusted", [("Sharpe", "Sharpe", fmt_num), ("Sortino", "Sortino", fmt_num), ("Calmar", "Calmar", fmt_num)]),
    ]
    table = []
    for group, metrics in groups:
        for i, (label, key, fmt) in enumerate(metrics):
            table.append({"Group": group if i == 0 else "", "Metric": label, "Strategy": fmt(strat[key]),
                          "Buy & Hold": fmt(bh[key]) if bh is not None else ""})

    def strategy_only(group, rows):
        for i, (label, value) in enumerate(rows):
            table.append({"Group": group if i == 0 else "", "Metric": label, "Strategy": value, "Buy & Hold": "-" if bh is not None else ""})

    if trades is not None:
        ep = td.episode_stats(trades)
        strategy_only("Trades", [
            ("Trades", f"{ep['Episodes']:,}"), ("Total PnL", fmt_usd(sum(t["pnl"] for t in trades))),
            ("Win rate", fmt_pct(ep["Win Rate"], 1)), ("Profit factor", fmt_num(ep["Profit Factor"])),
            ("Avg R", fmt_num(ep["Avg R"])), ("Avg holding", fmt_days(ep["Avg Holding (days)"])),
            ("Best trade", fmt_usd(ep["Best Episode"])), ("Worst trade", fmt_usd(ep["Worst Episode"])),
            ("Most wins in a row", str(ep["Max Consecutive Wins"])), ("Most losses in a row", str(ep["Max Consecutive Losses"]))])
    if port is not None:
        _, total_cost, carry, gross = cost_figures(port)
        strategy_only("Costs", [
            ("Costs paid", fmt_usd(total_cost)),
            ("Costs as % of gross profit", f"{total_cost / gross * 100:.0f}%" if gross > 0 else "n/a"),
            ("Interest and carry", f"{carry:+,.0f}")])
    frame = pd.DataFrame(table)
    if bh is None:
        frame = frame.drop(columns="Buy & Hold")
    st.dataframe(frame, hide_index=True, width="stretch", height=min(36 * (len(frame) + 1), 900))
    show_underwater_chart(strategy_equity)


REPLAY_STEPS = {"1 day": 1, "1 week": 5, "1 month": 21, "1 quarter": 63}


def _thin(frame: pd.DataFrame, limit: int = 1800) -> pd.DataFrame:
    """Every k-th row (always keeping the last) so a long replay stays light to redraw."""
    if len(frame) <= limit:
        return frame
    keep = np.unique(np.append(np.arange(0, len(frame), len(frame) // limit + 1), len(frame) - 1))
    return frame.iloc[keep]


def show_portfolio_replay(port: dict, gen: int):
    """Scrub or play back the account day by day: equity, cash (realized) balance, unrealized profit,
    costs, margin, drawdown, and the positions open on that date. Needs the daily record the
    Backtest keeps (td.RECORD_HISTORY)."""
    import altair as alt
    hist = port.get("history")
    if hist is None or hist.empty:
        st.caption("Run the backtest again to record the day-by-day replay.")
        return
    ss = st.session_state
    positions = port.get("positions_history")
    start_capital = float(port["starting_capital"])
    margin_modelled = bool(port.get("margin_modelled"))
    n = len(hist)
    first, last = hist.index[0].date(), hist.index[-1].date()
    if ss.get("replay_gen") != gen:
        ss.replay_gen, ss.replay_date, ss.replay_playing = gen, last, False
    ss.setdefault("replay_step", "1 month")
    playing = bool(ss.get("replay_playing"))
    margin_calls = port.get("margin_calls", [])
    full_dd = (hist["equity"] / hist["equity"].cummax() - 1) * 100

    st.markdown("**Replay the account day by day**")
    st.caption("Press Play or drag the date. Everything up to the chosen date is drawn; the bars and table "
               "show what was open on that date.")

    def draw(pos: int, log_scale: bool):
        snap = hist.iloc[pos]
        date = hist.index[pos]
        seen = hist.iloc[:pos + 1]
        equity, cash, open_pnl = float(snap["equity"]), float(snap["realized_balance"]), float(snap["unrealized"])
        peak = float(seen["equity"].max())
        costs = float(snap["cum_slippage_spread"] + snap["cum_commission"] + snap["cum_holding"])
        st.markdown(f"##### {date:%A, %d %B %Y}")
        calls_seen = [e for e in margin_calls if pd.Timestamp(e["date"]) <= date]
        gross = float(snap["gross_notional"])
        used = float(snap["margin_initial"])
        tiles = [
            ("Account equity", f"${equity:,.0f}", None, None),
            ("Profit so far", f"${equity - start_capital:,.0f}", f"{(equity / start_capital - 1) * 100:,.1f}%", None),
            ("Realized (cash) balance", f"${cash:,.0f}", None,
             "Starting capital plus every closed trade, after commissions and holding costs."),
            ("Unrealized profit", f"${open_pnl:,.0f}", None,
             "Profit or loss of positions still open, at that day's close."),
            ("Drawdown from peak", f"{(equity / peak - 1) * 100:.1f}%", None, None),
            ("Costs paid so far", f"${costs:,.0f}", None,
             "Slippage and spread, commissions and holding cost since the start."),
            ("Open positions", f"{int(snap['open_positions'])}", None, None),
            ("Gross exposure", f"${gross:,.0f}", f"{gross / equity:.2f}x equity" if equity > 0 else None, None),
        ]
        if margin_modelled:
            tiles += [
                ("Margin used", f"${used:,.0f}", None, None),
                ("Free margin", f"${equity - used:,.0f}", None, None),
                ("Margin level", f"{equity / used * 100:,.0f}%" if used > 0 else "n/a", None,
                 "Equity divided by margin used. A margin call happens when equity falls below the "
                 "maintenance requirement shown on the margin chart."),
                ("Margin calls so far", f"{len(calls_seen)}", None, None),
            ]
        for row_start in range(0, len(tiles), 3):
            for col, (label, value, delta, help_text) in zip(st.columns(3), tiles[row_start:row_start + 3]):
                col.metric(label, value, delta, delta_color="off", help=help_text)
        if not margin_modelled:
            st.caption("Margin is not modelled for this profile.")

        plot = _thin(seen).reset_index()
        x = alt.X("date:T", scale=alt.Scale(domain=[hist.index[0], hist.index[-1]]), title=None)
        both = hist[["equity", "realized_balance"]]
        lo = float(max(min(both.min().min(), start_capital) * 0.9, 1.0))
        hi = float(max(both.max().max(), start_capital) * 1.1)
        y_scale = alt.Scale(type="log" if log_scale else "linear", domain=[lo, hi], nice=False)
        lines = plot.melt("date", ["equity", "realized_balance"], var_name="Series", value_name="usd")
        lines["Series"] = lines["Series"].map({"equity": "Account equity", "realized_balance": "Realized balance"})
        base = alt.Chart(lines).mark_line().encode(
            x=x, y=alt.Y("usd:Q", scale=y_scale, title="$"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Account equity", "Realized balance"],
                                                         range=["#1f77b4", "#8c8c8c"])),
            tooltip=["date:T", "Series:N", alt.Tooltip("usd:Q", format=",.0f", title="$")])
        start_rule = alt.Chart(pd.DataFrame({"y": [start_capital]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(y="y:Q")
        layers = [base, start_rule]
        if calls_seen:
            call_df = pd.DataFrame({"date": [pd.Timestamp(e["date"]) for e in calls_seen],
                                    "usd": [float(e["equity"]) for e in calls_seen]})
            layers.append(alt.Chart(call_df).mark_point(color="#d62728", size=90, filled=True).encode(
                x="date:T", y="usd:Q",
                tooltip=["date:T", alt.Tooltip("usd:Q", format=",.0f", title="Equity at margin call")]))
        st.altair_chart(alt.layer(*layers).properties(
            height=260, title="Account equity vs realized balance (red dots: margin calls)"),
            width="stretch")

        left, right = st.columns(2)
        with left:
            frame = plot[["date", "unrealized"]].rename(columns={"unrealized": "Unrealized profit ($)"})
            st.altair_chart(alt.Chart(frame).mark_area(opacity=0.6, color="#2ca02c", line=True).encode(
                x=x, y=alt.Y("Unrealized profit ($):Q"),
                tooltip=["date:T", alt.Tooltip("Unrealized profit ($):Q", format=",.0f")]
            ).properties(height=200, title="Unrealized profit of open positions"), width="stretch")
        with right:
            dd = (seen["equity"] / seen["equity"].cummax() - 1) * 100
            frame = _thin(dd.rename("Drawdown (%)").to_frame()).reset_index()
            floor = min(float(full_dd.min()) * 1.05, -1.0)
            st.altair_chart(alt.Chart(frame).mark_area(opacity=0.6, color="#d62728", line=True).encode(
                x=x, y=alt.Y("Drawdown (%):Q", scale=alt.Scale(domain=[floor, 0])),
                tooltip=["date:T", alt.Tooltip("Drawdown (%):Q", format=".1f")]
            ).properties(height=200, title="Drawdown from the equity peak"), width="stretch")

        if margin_modelled:
            mframe = plot[["date", "equity", "margin_initial", "margin_maintenance"]].rename(columns={
                "equity": "Account equity", "margin_initial": "Margin used",
                "margin_maintenance": "Maintenance requirement"})
            mlines = mframe.melt("date", var_name="Series", value_name="usd")
            top = float(max(hist["equity"].max(), hist["margin_initial"].max()) * 1.05)
            st.altair_chart(alt.Chart(mlines).mark_line().encode(
                x=x, y=alt.Y("usd:Q", scale=alt.Scale(type="symlog" if log_scale else "linear", domain=[0, top]),
                             title="$"),
                color=alt.Color("Series:N", scale=alt.Scale(
                    domain=["Account equity", "Margin used", "Maintenance requirement"],
                    range=["#1f77b4", "#ff7f0e", "#d62728"])),
                tooltip=["date:T", "Series:N", alt.Tooltip("usd:Q", format=",.0f", title="$")]
            ).properties(height=220, title="Margin: equity must stay above the maintenance requirement"),
                width="stretch")

        st.markdown("**Open on this date**")
        today = positions[positions["date"] == date] if positions is not None else None
        if today is None or today.empty:
            st.caption("No open positions.")
        else:
            table = today.drop(columns="date").rename(columns={
                "asset": "Asset", "shares": "Size (units)", "avg_entry": "Avg entry", "price": "Price",
                "unrealized": "Unrealized P&L ($)", "notional": "Value ($)", "margin": "Margin ($)"})
            if not margin_modelled:
                table = table.drop(columns="Margin ($)")
            bars, tbl = st.columns([1, 1.4])
            with bars:
                st.altair_chart(alt.Chart(table).mark_bar().encode(
                    x=alt.X("Asset:N", sort="-y"), y=alt.Y("Value ($):Q"),
                    color=alt.condition(alt.datum["Unrealized P&L ($)"] >= 0,
                                        alt.value("#2ca02c"), alt.value("#d62728")),
                    tooltip=["Asset", alt.Tooltip("Value ($):Q", format=",.0f"),
                             alt.Tooltip("Unrealized P&L ($):Q", format=",.0f")]
                ).properties(height=220, title="Position value by asset (green = in profit)"),
                    width="stretch")
            with tbl:
                st.dataframe(table.round(2), hide_index=True, width="stretch", height=220)

        closed = []
        for asset, trades in port["trades_by_asset"].items():
            for t in trades:
                if pd.Timestamp(t["exit_date"]) <= date:
                    closed.append({"Closed": pd.Timestamp(t["exit_date"]).date(), "Asset": asset,
                                   "System": t["system"], "Opened": pd.Timestamp(t["entry_dates"][0]).date(),
                                   "Exit reason": t["exit_reason"], "P&L ($)": round(float(t["pnl"]))})
        with st.expander(f"Trades closed up to this date ({len(closed)})"):
            if closed:
                st.dataframe(pd.DataFrame(closed).sort_values("Closed", ascending=False).head(25),
                             hide_index=True, width="stretch")
            else:
                st.caption("None yet.")

    def toggle_play():
        # a callback runs before the widgets are drawn, so it may set the slider's own state safely
        if not ss.get("replay_playing") and ss.get("replay_date", last) >= last:
            ss.replay_date = first   # replaying from the end starts over
        ss.replay_playing = not ss.get("replay_playing")
        ss.replay_restart_timer = True

    @st.fragment(run_every=0.4 if playing else None)
    def replay():
        if "replay_date" not in ss:
            ss.replay_date = last
        if ss.get("replay_playing"):
            pos = int(hist.index.searchsorted(pd.Timestamp(ss.replay_date), side="right")) - 1
            new_pos = pos + REPLAY_STEPS[ss.replay_step]
            if new_pos >= n - 1:
                ss.replay_playing, ss.replay_restart_timer = False, True
                new_pos = n - 1
            ss.replay_date = hist.index[new_pos].date()
        c1, c2, c3 = st.columns([1, 1.2, 4])
        with c1:
            st.button("Pause" if ss.get("replay_playing") else "Play", key="replay_play",
                      width="stretch", on_click=toggle_play)
        with c2:
            st.selectbox("Speed", list(REPLAY_STEPS), key="replay_step", label_visibility="collapsed",
                         help="How much time passes per tick while playing.")
        with c3:
            st.slider("Date", min_value=first, max_value=last, key="replay_date", label_visibility="collapsed")
        pos = max(int(hist.index.searchsorted(pd.Timestamp(ss.replay_date), side="right")) - 1, 0)
        log_scale = st.checkbox("Log scale", value=True, key="replay_log")
        draw(pos, log_scale)
        if ss.pop("replay_restart_timer", False):
            st.rerun()   # whole app, once everything is drawn, so the play timer starts or stops

    replay()


def show_cost_breakdown(port: dict):
    items, total, carry, gross = cost_figures(port)
    cols = st.columns(4)
    for col, (label, value) in zip(cols, items):
        col.metric(label, f"${value:,.0f}")
    if carry:
        st.caption(f"Interest on cash and carry on positions: {carry:+,.0f} (included in the final equity, not in the costs below).")
    cols[3].metric("Costs as % of gross profit", f"{total / gross * 100:.0f}%" if gross > 0 else "n/a",
                   help="All three costs divided by what the strategy made before costs. Over 100% means "
                        "costs ate all of the profit. The $ amounts grow with the account as it compounds, "
                        "so the share is the useful number.")


def show_account_realism(port: dict):
    """What the whole-contract sizing and margin rules did to this run, when either is on."""
    show_cost_breakdown(port)
    sized_in_steps = bool(st.session_state.get("bt_run_config", {}).get("td", {}).get("SIZE_STEP"))
    margin_on = bool(st.session_state.get("bt_run_config", {}).get("td", {}).get("USE_MARGIN_CALLS"))
    if not (sized_in_steps or margin_on):
        if not port.get("mark_to_market", True):
            st.caption("Equity counts closed trades only (mark to market is off), so drawdowns here "
                       "leave out the swings of open positions.")
        return
    cols = st.columns(4)
    if sized_in_steps:
        cols[0].metric("Signals skipped: size under 1 contract", f"{port.get('n_rounding_skips', 0):,}",
                       help="Times the strategy wanted to open or add to a position but the size its risk rule "
                            "asked for was less than one whole contract (or lot), so nothing was traded. "
                            "A run of consecutive days with the same signal counts once. Includes pyramid adds.")
    if margin_on:
        calls = port.get("margin_calls", [])
        cols[1].metric("Margin calls", f"{len(calls):,}",
                       help="Days the account fell below the maintenance margin and positions were "
                            "closed by the broker.")
        cols[2].metric("Signals cut by margin", f"{port.get('n_margin_blocked', 0):,}",
                       help="Entries or adds refused or made smaller because the account could not "
                            "pay the initial margin.")
        cols[3].metric("Peak margin use", f"{port.get('max_margin_use', 0) * 100:.0f}%",
                       help="Highest initial margin as a share of account equity at a daily close. "
                            "Above 100% means the account was over-committed.")
        if port.get("wiped_out"):
            st.error("The account was wiped out: equity reached zero.")
        if calls:
            with st.expander(f"Margin call log ({len(calls)})"):
                st.dataframe(pd.DataFrame([{
                    "Date": e["date"].date(), "Equity": round(float(e["equity"])),
                    "Margin needed": round(float(e["required"])), "Positions closed": e["units_closed"]}
                    for e in calls]), hide_index=True, width="stretch")


def show_underwater_chart(equity_curve: pd.Series):
    """Drawdown-from-peak over time (the 'underwater' chart) -- same formula performance_stats()
    already computes internally and discards (turtle_dual_system.py's own drawdown/running_max
    locals), recomputed fresh here rather than changing that function's return contract, since
    print_stats_table/optimizer/CSV consumers all assume every value in it is a scalar."""
    equity = equity_curve.dropna()
    if equity.empty:
        return
    drawdown = (equity / equity.cummax() - 1) * 100
    st.caption("Underwater chart -- % below the equity curve's own running peak")
    st.area_chart(drawdown)


# exit_reason strings that represent a downside-protective stop firing, vs. a profit-taking
# exit, vs. everything else (channel exit, time stop, exposure trim, end of data) -- these are
# genuinely different in character (SL: something went wrong; TP: something went right; Other:
# neither, just a mechanical/structural exit) so they get separate marker categories, not lumped
# together as one generic "exit"
_STOP_TYPE_REASONS = {"SL", "trailing_stop_low", "chandelier_exit", "giveback_stop"}
_TP_TYPE_REASONS = {"scale_out", "regime_scale_out", "target"}


def _memo(cache_key: tuple, compute_fn):
    """Tiny memoization helper for the Backtest/Trade Chart tabs' result-display blocks.
    Streamlit reruns the ENTIRE script on every widget interaction anywhere in the app (st.tabs()
    renders every tab body regardless of which one is visible), so without this, dragging an
    unrelated slider on the Optimizer tab forces every stat/dataframe on the Backtest tab to
    rebuild from scratch. cache_key should start with st.session_state.bt_result_gen (bumped only
    when a backtest actually runs, see 'Run backtest' below) so a genuinely new result still
    invalidates correctly -- switching pages/tabs/unrelated widgets becomes a cheap dict lookup
    instead of a recompute."""
    cache = st.session_state.setdefault("bt_stat_cache", {})
    if cache_key not in cache:
        cache[cache_key] = compute_fn()
    return cache[cache_key]


def _exit_reason_pnl(trades: list) -> dict:
    """$ PnL summed by exit reason -- episode_stats() only counts UNITS per reason, never sums
    the dollars each reason actually contributed. Works for both engines: Dual System trades have
    unit_details (a pyramided episode can exit via several different reasons across its units, so
    the breakdown has to be per-unit, not per-episode); Turtle Soup trades are single-shot, so the
    top-level trade's own exit_reason/pnl stand in for "unit_details" directly."""
    breakdown = {}
    for t in trades:
        units = t.get("unit_details") or [t]
        for u in units:
            breakdown[u["exit_reason"]] = breakdown.get(u["exit_reason"], 0.0) + u["pnl"]
    return dict(sorted(breakdown.items(), key=lambda kv: -abs(kv[1])))


def _curve_years(curve) -> float:
    """Length of an equity curve in years (None if it can't be worked out)."""
    try:
        years = (curve.index[-1] - curve.index[0]).days / 365.25
        return years if years > 0 else None
    except Exception:
        return None


_MC_EXPLANATIONS = {
    "shuffle": "Same trades, random order. Shows how much the Max Drawdown depended on which trade "
               "happened to come first or last. The final result is the same in every shuffle (order "
               "doesn't change the product of the returns), so only the drawdown spreads.",
    "bootstrap": "Draws the same number of trades at random WITH replacement from this run's trades -- "
                 "what you'd get if the same kind of trades came in a different, luckier or unluckier "
                 "mix. Shows the range of outcomes and the chance of ending up losing money.",
    "skip": "Keeps the original order but randomly drops a share of the trades (missed signals, "
            "rejected orders, trades the account couldn't afford). Shows how fragile the result is "
            "to missing some trades.",
}


_MC_SERIES_EXPLANATIONS = {
    "shuffle": "Cuts the daily equity curve into blocks of days and puts the blocks in a random order. "
               "Overlapping positions and swings during open trades are all kept, because the curve "
               "already contains them. The final result is the same in every shuffle (order doesn't "
               "change the product of the returns), so only the drawdown spreads.",
    "bootstrap": "Draws blocks of days at random WITH replacement from the equity curve -- the same "
                 "kind of market and trading, in a luckier or unluckier mix. Shows the range of "
                 "outcomes and the chance of ending up losing money.",
    "skip": "Keeps the original order but replaces a share of the blocks with flat, no-return "
            "stretches (periods you weren't trading or missed). Shows how much of the result "
            "depends on a few stretches of days.",
}


def render_monte_carlo(key_prefix: str, trades: list = None, equity_curve=None, years: float = None,
                       series_mode: bool = False):
    """The Monte Carlo test controls and results (no expander of its own, so callers can place it
    inside whatever container they already have).

    series_mode=True simulates from the DAILY equity curve (blocks of days) -- the right choice for
    a shared portfolio, where positions overlap in time. Otherwise it replays the closed trades'
    returns one after another, which suits one asset at a time."""
    if series_mode:
        returns = monte_carlo.equity_returns(equity_curve) if equity_curve is not None else []
        if len(returns) < 60:
            st.caption("Not enough daily history for a Monte Carlo test.")
            return
        methods, explanations, unit = monte_carlo.SERIES_METHODS, _MC_SERIES_EXPLANATIONS, "stretches of days"
    else:
        returns = monte_carlo.trade_returns(trades or [], equity_curve)
        if len(returns) < 10:
            st.caption("Not enough closed episodes with a usable entry-equity reference (need at "
                       "least 10) for a Monte Carlo test to mean anything.")
            return
        methods, explanations, unit = monte_carlo.METHODS, _MC_EXPLANATIONS, "trades"
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("Test", list(methods), format_func=methods.get, key=f"mc_method_{key_prefix}")
    with c2:
        n_sims = st.number_input("Simulations", 200, 5000, 1000, 200, key=f"mc_n_{key_prefix}")
    with c3:
        skip = st.slider(f"Share of {unit} skipped", 0.02, 0.5, 0.1, 0.02, key=f"mc_skip_{key_prefix}",
                         disabled=method != "skip")
    block = 126
    if series_mode:
        block = st.number_input("Block length (trading days)", 5, 504, 126, 1, key=f"mc_block_{key_prefix}",
                                help="How many consecutive days stay together. Trend following loses "
                                     "money in long stretches, so short blocks understate drawdowns: "
                                     "checked against 60 strategy variants, 21-day blocks put the real "
                                     "drawdown beyond the simulated 95th percentile in 20% of cases "
                                     "(5% would be right); 126 days (about 6 months) got it to 5%.")
    drift_unc = 0.0
    if series_mode:
        use_drift = st.checkbox(
            "Allow for uncertainty in the strategy's true average return", value=True,
            key=f"mc_drift_{key_prefix}", disabled=method == "shuffle",
            help="Without this every simulated path assumes the average return seen in this one history "
                 "is the true one, which makes the range of final results far too narrow: simulating "
                 "from the first half of history, the real second-half result landed inside the "
                 "simulated 90% range only about half the time. With it (2 standard errors of the "
                 "average return, from year-to-year variation) it landed inside 90% of the time. "
                 "Does not apply to reshuffling, whose final result is fixed by design.")
        drift_unc = monte_carlo.DEFAULT_DRIFT_UNCERTAINTY if (use_drift and method != "shuffle") else 0.0
    else:
        st.caption("Approximate: trades that overlapped in time are replayed one after another, so "
                   "absolute figures are indicative. For a whole portfolio use the equity-curve test.")
    dd_limit = st.slider("Count a path as a deep drawdown when its Max Drawdown is worse than (%)",
                         -90, -10, -50, 5, key=f"mc_dd_{key_prefix}")
    st.caption(explanations[method])
    if series_mode:
        mc = monte_carlo.simulate_series(returns, method, int(n_sims), block=int(block),
                                         skip_fraction=skip, years=years, drift_uncertainty=drift_unc)
    else:
        mc = monte_carlo.simulate_trades(returns, method, int(n_sims), skip_fraction=skip, years=years)
    s = monte_carlo.summarize(mc, dd_limit / 100)

    a1, a2, a3 = st.columns(3)
    a1.metric("Actual Max Drawdown", f"{mc['actual_max_dd']*100:.1f}%",
              help="This backtest's real, chronological Max Drawdown (computed from its trades).")
    a2.metric("Simulated median Max DD", f"{s['dd_p50']*100:.1f}%",
              help=f"Median over {int(n_sims):,} simulations. 5th-95th percentile: "
                   f"{s['dd_p5']*100:.1f}% to {s['dd_p95']*100:.1f}%.")
    a3.metric("Actual result's percentile", f"{mc['actual_max_dd_percentile']:.0f}%",
              help="Where the real drawdown ranks among the simulated ones: 50% is typical, 90%+ means "
                   "the real path was an unusually BAD draw, under 10% an unusually good one.")
    b1, b2, b3 = st.columns(3)
    b1.metric("Actual final result", f"{mc['actual_final_multiple']:.2f}x")
    b2.metric("Simulated median final", f"{s['final_p50']:.2f}x",
              help=f"5th-95th percentile: {s['final_p5']:.2f}x to {s['final_p95']:.2f}x of starting capital.")
    b3.metric("Chance of losing money", f"{s['p_loss']*100:.1f}%",
              help="Share of simulations that ended below the starting capital.")
    d1, d2 = st.columns(2)
    d1.metric(f"Chance of a drawdown worse than {dd_limit}%", f"{s['p_dd_worse']*100:.1f}%")
    d2.metric("Chance of losing 90% or more", f"{s['p_ruin']*100:.1f}%")
    if "underwater_days" in mc:
        u1, u2 = st.columns(2)
        u1.metric("Actual longest time under water", f"{mc['actual_underwater_days']/252:.1f} years",
                  help="The longest stretch the real equity curve spent below a previous peak.")
        u2.metric("Simulated median longest time under water", f"{np.median(mc['underwater_days'])/252:.1f} years",
                  help=f"5th-95th percentile: {np.percentile(mc['underwater_days'], 5)/252:.1f} to "
                       f"{np.percentile(mc['underwater_days'], 95)/252:.1f} years. At the default "
                       f"126-day blocks the real figure matched the simulated one closely (0.96x "
                       f"across 60 variants); short blocks understate it.")
    if "cagr" in mc and np.isfinite(mc["cagr"]).any():
        c5, c50, c95 = np.nanpercentile(mc["cagr"], [5, 50, 95]) * 100
        st.caption(f"Simulated CAGR: median {c50:.1f}% (5th-95th percentile {c5:.1f}% to {c95:.1f}%)")

    h1, h2 = st.columns(2)
    with h1:
        st.caption("Max Drawdown across simulations (%)")
        counts, edges = np.histogram(mc["max_dd"] * 100, bins=30)
        st.bar_chart(pd.Series(counts, index=[f"{edges[i]:.1f}" for i in range(len(counts))]))
    with h2:
        st.caption("Final result across simulations (x starting capital)")
        if np.ptp(mc["final_multiple"]) < 1e-9:
            st.caption("Identical in every shuffle -- nothing to plot.")
        else:
            counts, edges = np.histogram(mc["final_multiple"], bins=30)
            st.bar_chart(pd.Series(counts, index=[f"{edges[i]:.2f}" for i in range(len(counts))]))


def show_monte_carlo_section(key_prefix: str, trades: list = None, equity_curve=None, years: float = None,
                             title: str = "Monte Carlo robustness check", series_mode: bool = False):
    """The Monte Carlo test as an expander (collapsed by default -- a deeper-dive diagnostic)."""
    with st.expander(title, icon=":material/casino:"):
        if years is None and equity_curve is not None:
            years = _curve_years(equity_curve)
        render_monte_carlo(key_prefix, trades, equity_curve, years, series_mode)


def _position_line(start_date, end_date, price: float, color: str, title: str, dashed: bool = True):
    """One flat horizontal price line spanning exactly [start_date, end_date] -- built as a
    2-point Line series rather than lightweight-charts' own createPriceLine (which draws all the
    way across the WHOLE visible chart, not just a position's own date range). Used only for the
    entry-price reference line; SL/TP get their own shaded zones via _position_zone below."""
    return {
        "type": "Line",
        "data": [
            {"time": start_date.strftime("%Y-%m-%d"), "value": float(price)},
            {"time": end_date.strftime("%Y-%m-%d"), "value": float(price)},
        ],
        "options": {
            "color": color, "lineWidth": 1, "lineStyle": 2 if dashed else 0,
            "title": title, "priceLineVisible": False, "lastValueVisible": False,
            "crosshairMarkerVisible": False,
        },
    }


def _position_zone(start_date, end_date, entry_price: float, edge_price: float, is_profit: bool, title: str, rgb: str = None, alpha: float = 0.18):
    """The actual TradingView-position-tool look: a SHADED rectangle from entry_price to
    edge_price (SL or TP), bounded to [start_date, end_date] -- not just a boundary line. Built
    as a lightweight-charts 'Baseline' series: it fills the area between its own (flat) line and
    a fixed baseline price, using a DIFFERENT color depending on whether the line sits above or
    below that baseline. Anchoring the baseline at entry_price and the line flat at edge_price
    means only ONE side ever actually paints per series (top=profit zone when edge_price >
    entry_price, bottom=loss zone when edge_price < entry_price) -- so a TP call and an SL call
    together are what produce the two-tone boxed look, green above entry and red below, exactly
    like dragging TradingView's own long-position tool onto a chart."""
    green, red, clear = "rgba(76,175,80,", "rgba(244,67,54,", "rgba(0,0,0,0)"
    base = rgb or (green if is_profit else red)
    fill = f"{base}{alpha})"
    line = f"{base}0.7)"
    return {
        "type": "Baseline",
        "data": [
            {"time": start_date.strftime("%Y-%m-%d"), "value": float(edge_price)},
            {"time": end_date.strftime("%Y-%m-%d"), "value": float(edge_price)},
        ],
        "options": {
            "baseValue": {"type": "price", "price": float(entry_price)},
            "topLineColor": line if is_profit else clear,
            "topFillColor1": fill if is_profit else clear,
            "topFillColor2": fill if is_profit else clear,
            "bottomLineColor": clear if is_profit else line,
            "bottomFillColor1": clear if is_profit else fill,
            "bottomFillColor2": clear if is_profit else fill,
            "lineWidth": 1, "title": title,
            "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False,
        },
    }


def build_lightweight_chart_data(df: pd.DataFrame, result: dict):
    """Builds the (candlestick_data, markers, position_lines) needed for
    streamlit_lightweight_charts' renderLightweightCharts -- Entry / Pyramid Add / SL Exit / TP
    Exit / Other Exit MARKERS, plus a shaded entry/SL/TP position box per trade (see
    _position_zone/_position_line) -- the equivalent of dropping TradingView's own position tool
    onto each trade: a green zone from entry up to the take-profit level, a red zone from entry
    down to the stop, both bounded to just that trade's own entry-to-exit dates, showing where
    the risk/reward was actually defined, not just the marker dot where it fired.
    Reads turtle_dual_system.py's trades: per-unit `unit_details` (first unit = Entry, later
    units = Pyramid Add).

    IMPORTANT: exit events are read from exit_markers_by_system, NOT each trade's own top-level
    exit_reason/exit_date/exit_price -- a partial scale-out (TP) happens MID-episode and never
    becomes the episode's own final exit_reason (the episode might go on to close via SL, or
    channel exit, or whatever comes later), so relying on the trade-level field alone silently
    misses every partial profit-take. exit_markers_by_system records every exit event, partial
    or final, each with its own price/date/reason -- confirmed empirically: with the default
    config's use_scale_out=True active, trade-level exit_reason showed ZERO 'scale_out' entries
    across 81 real episodes, even though scale-outs were genuinely happening throughout.

    Marker shapes are constrained to what lightweight-charts actually supports: circle, square,
    arrowUp, arrowDown -- confirmed directly from the library's own README (no "star", which
    would have been a natural but wrong guess from memory alone).
    """
    candles = [
        {"time": ts.strftime("%Y-%m-%d"), "open": float(o), "high": float(h),
         "low": float(l), "close": float(c)}
        for ts, o, h, l, c in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"])
    ]

    markers = []
    position_lines = []
    equity_curve = result.get("equity_curve")
    # (sysname, exit_date_str, exit_reason) -> FIFO queue of {"r_multiple", "equity_pct"} dicts,
    # one entry per actual exit EVENT (a pyramided episode can have several -- scale-outs, then a
    # final close) -- filled below from unit_details (Dual) / the trade itself (Soup, single-
    # shot), then consumed when building the exit markers loop further down.
    #
    # Deliberately R-multiple + true equity-% here, NOT pnl/capital_deployed (an earlier version
    # of this showed that instead, and it's misleading): capital_deployed can be a large multiple
    # of equity once leverage/exposure settings allow it, so "return on capital deployed" makes a
    # position that only risked 3% of the ACCOUNT look like it returned some huge % -- true, but
    # not the number that answers "was this actually a good trade for MY account." R-multiple
    # (pnl / risk_amount, i.e. how many risk-units were made or lost) and equity-% (pnl / equity
    # at entry, i.e. this trade's real contribution to total account equity) are what actually
    # answer that, and both hold "risk was capped at 3%" and "this trade's real outcome" in the
    # same units so they're directly comparable.
    exit_enrichment = {}

    def _queue_enrichment(sysname, date_str, reason, r_multiple, equity_pct):
        exit_enrichment.setdefault((sysname, date_str, reason), []).append(
            {"r_multiple": r_multiple, "equity_pct": equity_pct})

    for sysname, trades in result["trades_by_system"].items():
        for t in trades:
            unit_details = t.get("unit_details")
            if unit_details:
                for u in unit_details:
                    if u.get("exit_date") is None:
                        continue   # still open -- no exit event to enrich yet
                    risk_amount = u.get("risk_amount") or 0.0
                    equity_at_entry = u.get("equity_at_entry") or 0.0
                    r_multiple = (u["pnl"] / risk_amount) if risk_amount else float("nan")
                    equity_pct = (u["pnl"] / equity_at_entry * 100) if equity_at_entry else float("nan")
                    _queue_enrichment(sysname, u["exit_date"].strftime("%Y-%m-%d"), u["exit_reason"],
                                        r_multiple, equity_pct)
                # DEDUPLICATE by entry_iloc first: a single pyramid unit that gets partially
                # scaled out (use_scale_out, at multiple R levels) produces MULTIPLE entries in
                # unit_details -- one per partial-exit event -- all sharing the exact same
                # entry_iloc/entry_date/entry_price, since close_unit() does dict(u) and only
                # overwrites the exit-side fields. Without deduplicating first, each of those
                # partial-exit fragments gets counted as its own separate pyramid add, which is
                # exactly what produced the "multiple Add markers stacked on the same date"
                # symptom -- confirmed directly: 252/310 "adds" in a real stress test were
                # actually scale-out fragments of an add that had already happened, not new adds
                # at all. Genuinely distinct adds always have distinct entry_iloc values.
                seen_ilocs = {}
                for u in unit_details:
                    seen_ilocs.setdefault(u["entry_iloc"], u)
                deduped_units = list(seen_ilocs.values())
                for idx, u in enumerate(sorted(deduped_units, key=lambda u: u["entry_date"])):
                    equity_at_entry = u.get("equity_at_entry") or 0.0
                    risk_pct = (u["risk_amount"] / equity_at_entry * 100) if equity_at_entry else None
                    risk_suffix = f" (risk {risk_pct:.1f}%)" if risk_pct is not None else ""
                    if idx == 0:
                        markers.append({"time": u["entry_date"].strftime("%Y-%m-%d"),
                                         "position": "belowBar", "color": "#2196F3",
                                         "shape": "arrowUp", "text": f"{sysname} Entry{risk_suffix}"})
                    else:
                        markers.append({"time": u["entry_date"].strftime("%Y-%m-%d"),
                                         "position": "belowBar", "color": "#90CAF9",
                                         "shape": "arrowUp", "text": f"{sysname} Add{risk_suffix}"})

                # Position box: SL/TP computed from the FIRST unit's own frozen entry/stop/tp
                # fields -- the position's ORIGINAL risk/reward definition, not every later
                # pyramid add's own ratcheted stop (which would draw a cluttered near-duplicate
                # line per add instead of one clean box per episode). stop_n_mult/n_value/
                # tp_n_mult/tp_n_value are all frozen on the unit at entry time (open_unit()), so
                # no external config lookup is needed and this can't go stale if SYSTEMS changes.
                first_unit = min(deduped_units, key=lambda u: u["entry_iloc"])
                entry_date, exit_date = first_unit["entry_date"], t["exit_date"]
                entry_price = first_unit["entry_price"]
                sl_price = entry_price - first_unit["stop_n_mult"] * first_unit["n_value"]
                if exit_date > entry_date:
                    # Green zone extends to the highest High actually reached while the position
                    # was open (max favorable excursion), NOT the frozen tp_n_mult target -- a
                    # trend-following exit (channel/trailing-stop/time-stop) routinely runs the
                    # price well past that original target before finally closing, and the box
                    # should show the whole run, not just the planned first-target slice of it.
                    tp_price = max(entry_price, float(df.loc[entry_date:exit_date, "High"].max()))
                    position_lines.append(_position_zone(entry_date, exit_date, entry_price, tp_price,
                                                            is_profit=True, title=f"{sysname} TP zone"))
                    position_lines.append(_position_zone(entry_date, exit_date, entry_price, sl_price,
                                                            is_profit=False, title=f"{sysname} SL zone"))
                    position_lines.append(_position_line(entry_date, exit_date, entry_price,
                                                            "#616161", f"{sysname} Entry", dashed=True))

    for sysname, exit_markers in result.get("exit_markers_by_system", {}).items():
        for marker in exit_markers:
            iloc, price, reason = marker[0], marker[1], marker[2]
            date_str = df.index[iloc].strftime("%Y-%m-%d")
            # Pull this specific exit event's R-multiple/equity-% off the FIFO queue built above
            # (one queue per (system, date, reason) -- handles the rare case of two exit events
            # on the same bar with the same reason by matching them in the same order they
            # occurred).
            queue = exit_enrichment.get((sysname, date_str, reason))
            info = queue.pop(0) if queue else None
            if info is not None:
                r, e = info["r_multiple"], info["equity_pct"]
                r_str = f" {r:+.2f}R" if np.isfinite(r) else ""
                e_str = f" ({e:+.1f}% equity)" if np.isfinite(e) else ""
                pnl_suffix = r_str + e_str
            else:
                pnl_suffix = ""
            if reason in _STOP_TYPE_REASONS:
                markers.append({"time": date_str, "position": "aboveBar", "color": "#F44336",
                                 "shape": "arrowDown", "text": f"{sysname} SL{pnl_suffix}"})
            elif reason in _TP_TYPE_REASONS:
                markers.append({"time": date_str, "position": "aboveBar", "color": "#4CAF50",
                                 "shape": "arrowDown", "text": f"{sysname} TP{pnl_suffix}"})
            else:
                short_reason = {"exit_channel": "Chan", "time_stop": "Time",
                                 "exposure_trim": "Trim", "end_of_data": "End"}.get(reason, reason[:4])
                markers.append({"time": date_str, "position": "aboveBar", "color": "#9E9E9E",
                                 "shape": "circle", "text": f"{sysname} {short_reason}{pnl_suffix}"})

    # lightweight-charts requires markers sorted by time (ascending) -- multiple markers can
    # share a date, which is fine, but the ORDER across different dates matters to the library
    markers.sort(key=lambda m: m["time"])

    return candles, markers, position_lines


def build_stepped_position_boxes(df: pd.DataFrame, result: dict, sysdefs: dict, since=None) -> list:
    """Position boxes like the TradingView script's: per trade (episode) and per system, a GREEN area from the average entry up to the close while the position is open (up to the exit
    price on the last day) and a RED area from the average entry down to the stop. The stop is drawn as STEPS: every add re-sets the stop (3 N under the newest unit, like the engine), so
    each step is its own band and the red area shows where the stop was at each point of the trade. Once the stop is above the average entry the band is TEAL (profit locked in).
    The stop path is rebuilt from each unit's own entry price / N / stop multiple with the system's ratchet rules (ratchet_stop_on_pyramid, use_average_entry_stop, never-loosen); other stop
    mechanics (breakeven, trailing, chandelier, daily-loss exit) are not drawn. `since`: only trades that end on or after this date."""
    series = []
    green, red, teal = "rgba(76,175,80,", "rgba(244,67,54,", "rgba(0,150,136,"
    close = df["Close"]
    for sysname, trades in result["trades_by_system"].items():
        sd = sysdefs.get(sysname, {})
        ratchet = sd.get("ratchet_stop_on_pyramid", True)
        average_stop = sd.get("use_average_entry_stop", False)
        for t in trades:
            details = t.get("unit_details")
            if not details:
                continue
            if since is not None and t["exit_date"] < since:
                continue
            seen = {}
            for unit in details:
                seen.setdefault(unit["entry_iloc"], unit)
            units = sorted(seen.values(), key=lambda x: x["entry_iloc"])
            sum_q = sum_pq = 0.0
            stop = None
            for k, unit in enumerate(units):
                q = float(unit.get("initial_shares") or unit.get("shares") or 0.0)
                sum_q += q
                sum_pq += q * float(unit["entry_price"])
                avg = sum_pq / sum_q if sum_q > 0 else float(unit["entry_price"])
                reach = float(unit["stop_n_mult"]) * float(unit["n_value"])
                if k == 0:
                    stop = float(unit["entry_price"]) - reach
                elif ratchet:
                    new_stop = (avg - reach) if average_stop else (float(unit["entry_price"]) - reach)
                    stop = max(stop, new_stop) if (average_stop or td.NEVER_LOOSEN_STOPS) else new_stop
                start = unit["entry_date"]
                end = units[k + 1]["entry_date"] if k + 1 < len(units) else t["exit_date"]
                if not end > start:
                    continue
                locked = stop > avg
                series.append(_position_zone(start, end, avg, stop, is_profit=locked, title=f"{sysname} stop" if k == 0 else "",
                                             rgb=teal if locked else red, alpha=0.30 if locked else 0.20))
                days = df.loc[start:end]
                if len(days) >= 2:
                    values = [max(avg, float(c)) for c in close.loc[days.index]]
                    if k + 1 == len(units) and t.get("exit_price") is not None:
                        values[-1] = max(avg, float(t["exit_price"]))                 # the last day: up to the real exit price
                    series.append({
                        "type": "Baseline",
                        "data": [{"time": d.strftime("%Y-%m-%d"), "value": v} for d, v in zip(days.index, values)],
                        "options": {"baseValue": {"type": "price", "price": float(avg)},
                                    "topLineColor": f"{green}0.5)", "topFillColor1": f"{green}0.16)", "topFillColor2": f"{green}0.16)",
                                    "bottomLineColor": "rgba(0,0,0,0)", "bottomFillColor1": "rgba(0,0,0,0)", "bottomFillColor2": "rgba(0,0,0,0)",
                                    "lineWidth": 1, "title": "", "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False},
                    })
    return series


def build_monthly_high_lines(df: pd.DataFrame, min_age: int, max_age: int, buffer_n_mult: float, n_period: int, since=None) -> list:
    """A line at EVERY monthly high (each completed calendar month's highest high), drawn from the bar that set it until its breakout (the first close above high + buffer x N), or until it is
    older than max_age bars, whichever comes first. Dotted while it is younger than min_age (not tradable yet), solid once it is old enough - the same lines the TradingView script draws.
    `since`: only highs whose line ends on or after this date."""
    top, pos, nxt = td.monthly_high_months(df.index, df["High"])
    n_series = np.asarray(td.compute_atr(df, int(n_period)), dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    last = len(df) - 1
    orange = "rgba(255,152,0,0.9)"
    out = []

    def seg(a, b, price, dotted):
        if b <= a:
            return
        out.append({"type": "Line",
                    "data": [{"time": df.index[a].strftime("%Y-%m-%d"), "value": float(price)}, {"time": df.index[b].strftime("%Y-%m-%d"), "value": float(price)}],
                    "options": {"color": orange, "lineWidth": 1, "lineStyle": 3 if dotted else 0, "title": "", "priceLineVisible": False,
                                "lastValueVisible": False, "crosshairMarkerVisible": False}})

    for g in range(len(top)):
        if np.isnan(top[g]):
            continue
        p0 = int(pos[g])
        end = min(p0 + int(max_age), last)
        t0 = int(nxt[g])
        if t0 <= end:
            broken = c[t0:end + 1] > top[g] + float(buffer_n_mult) * n_series[t0:end + 1]
            if broken.any():
                end = t0 + int(np.argmax(broken))                                   # its breakout: the line ends here
        if since is not None and df.index[end] < since:
            continue
        young_end = min(p0 + int(min_age), end)
        seg(p0, young_end, top[g], True)
        seg(young_end, end, top[g], False)
    return out


CHART_LABEL_CHOICES = ["Entry / add labels", "Exit labels", "Risk % on entries", "P&L on exits (R, % equity)",
                       "Position box names"]


def chart_label_controls(key_prefix: str) -> set:
    """Multiselect for which texts appear on a trade chart; returns the set of choices on."""
    return set(st.multiselect(
        "Text shown on the chart", CHART_LABEL_CHOICES, default=CHART_LABEL_CHOICES,
        key=f"{key_prefix}_chart_labels",
        help="Pick which labels are drawn. Entry / add and Exit labels are the text next to each "
             "marker; Risk % and P&L are the extra figures inside them; Position box names are the "
             "TP zone / SL zone / Entry captions on the shaded boxes. The markers and boxes "
             "themselves always stay."))


def filter_chart_labels(markers: list, position_lines: list, shown: set) -> tuple:
    """Returns copies of markers / position_lines with the unselected texts removed."""
    import re
    out_markers = []
    for m in markers:
        m = dict(m)
        text = m.get("text", "")
        if m.get("position") == "belowBar":      # entries and adds
            if "Entry / add labels" not in shown:
                text = ""
            elif "Risk % on entries" not in shown:
                text = re.sub(r" \(risk [^)]*\)", "", text)
        else:                                      # exits
            if "Exit labels" not in shown:
                text = ""
            elif "P&L on exits (R, % equity)" not in shown:
                text = re.sub(r" [+-]\d+\.\d+R.*$", "", text)
        m["text"] = text
        out_markers.append(m)
    out_lines = []
    for line in position_lines:
        if "Position box names" not in shown:
            line = {**line, "options": {**line["options"], "title": ""}}
        out_lines.append(line)
    return out_markers, out_lines


def compute_combined_equity(results: dict) -> pd.Series:
    """Sums every asset's own equity curve, dollar-for-dollar, aligned on the union of their
    dates (forward/back-filled for any asset that has fewer trading days than another)."""
    all_index = None
    for r in results.values():
        idx = r["equity_curve"].index
        all_index = idx if all_index is None else all_index.union(idx)
    combined = None
    for r in results.values():
        aligned = r["equity_curve"].reindex(all_index).ffill().bfill()
        combined = aligned if combined is None else combined + aligned
    return combined


def compute_combined_buy_and_hold(raw: dict, starting_capital_per_asset: float) -> pd.Series:
    """Same idea as compute_combined_equity, but for a buy & hold of every asset instead --
    gives an apples-to-apples portfolio-level comparison against the combined strategy curve."""
    curves = {name: td.compute_buy_and_hold(df, starting_capital_per_asset) for name, df in raw.items()}
    all_index = None
    for c in curves.values():
        all_index = c.index if all_index is None else all_index.union(c.index)
    combined = None
    for c in curves.values():
        aligned = c.reindex(all_index).ffill().bfill()
        combined = aligned if combined is None else combined + aligned
    return combined


@st.cache_data(show_spinner=False)
def cached_download_data(tickers_items: tuple, start: str, end):
    """Cached so re-running the backtest or optimizer with the same assets and date range
    doesn't re-download from Yahoo Finance every single time -- the single biggest source of
    avoidable lag when iterating on parameters. tickers_items is a tuple (hashable, so
    st.cache_data can key on it) instead of a dict."""
    return td.download_data(dict(tickers_items), start, end)


@st.cache_data(show_spinner=False)
def cached_fetch_all_time_highs(tickers_items: tuple):
    """Cached companion to cached_download_data, for the genuine full-history all-time-high
    reference (require_all_time_high) -- a separate, much longer download per asset, so caching
    this matters even more than the regular price data."""
    return td.fetch_all_time_highs(dict(tickers_items))


def run_trials_with_progress(raw_data, param_list, n_workers, progress_bar, status, update_every, start_time,
                               worker_module=opt, ath_references=None):
    """Runs every trial in param_list, updating progress_bar/status every `update_every` trials
    (not every single one -- redrawing a Streamlit widget has real overhead on long runs).

    Uses worker_module._init_worker / worker_module._run_trial_worker for BOTH the parallel and
    sequential paths -- those live in a real, separately-importable module (optimize_turtle_
    dual_system.py or optimize_turtle_soup.py), not in this UI file, which is what makes a real
    multiprocessing.Pool safe to create here: child processes only need to re-import that module
    to find the worker function, not re-execute this Streamlit script. If Pool creation itself
    fails for any reason, falls back to running the same trials sequentially in-process rather
    than leaving you with nothing.

    ath_references (optional): the full-history all-time-high reference dict, passed through to
    _init_worker as a second argument -- both optimizer modules' _init_worker accept this now
    (optimize_turtle_soup.py's simply ignores it, since Turtle Soup has no ATH filter), so this
    works identically regardless of which worker_module is passed in.

    Returns (results, n_failed, ran_parallel).
    """
    total = len(param_list)
    results = []
    n_failed = 0

    if n_workers > 1:
        # On Windows, Pool workers spawn fresh (no fork) and re-import worker_module from disk,
        # so a runtime-mutated module global like opt.SCORE_MODE would otherwise silently revert
        # to the file's own default the moment a real worker process picks up a trial -- passing
        # it through initargs (same mechanism raw_data/ath_references already use) is what makes
        # it actually reach every worker. optimize_turtle_soup.py has no such setting, so this is
        # simply omitted for that engine (hasattr guards it).
        pool_initargs = (raw_data, ath_references)
        if hasattr(worker_module, "SCORE_MODE"):
            pool_initargs += (worker_module.SCORE_MODE,)
            if hasattr(worker_module, "current_portfolio_config"):
                pool_initargs += (worker_module.current_portfolio_config(),)
        try:
            with mp.Pool(processes=n_workers, initializer=worker_module._init_worker,
                         initargs=pool_initargs) as pool:
                for i, result in enumerate(pool.imap_unordered(worker_module._run_trial_worker, param_list, chunksize=1), start=1):
                    if result is None:
                        n_failed += 1
                    else:
                        results.append(result)
                    if i % update_every == 0 or i == total:
                        progress_bar.progress(i / total)
                        status.write(f"{i}/{total} trials -- {len(results)} valid, {n_failed} skipped -- "
                                     f"{time.time()-start_time:.0f}s elapsed ({n_workers} workers)")
            return results, n_failed, True
        except Exception as e:
            status.write(f"Parallel execution failed to start ({e}) -- falling back to sequential.")
            results, n_failed = [], 0

    worker_module._init_worker(raw_data, ath_references)
    for i, params in enumerate(param_list, start=1):
        result = worker_module._run_trial_worker(params)
        if result is None:
            n_failed += 1
        else:
            results.append(result)
        if i % update_every == 0 or i == total:
            progress_bar.progress(i / total)
            status.write(f"{i}/{total} trials -- {len(results)} valid, {n_failed} skipped -- "
                         f"{time.time()-start_time:.0f}s elapsed (sequential)")
    return results, n_failed, False


def run_two_stage_search(raw_data, ath_references, n_trials, n_workers, worker_module, rank_metric,
                           stage1_fraction=0.6, top_k=20, narrow_window=1, stage2_blind_fraction=0.4,
                           only_optimize=None, fixed_values=None):
    """UI version of optimize_turtle_dual_system.py's own two-stage coarse-to-fine search --
    same logic (broad stage 1, narrow to the neighborhood of the top trials, hedged stage 2 that
    stays partly blind), just with Streamlit progress bars instead of print statements.

    IMPORTANT CONTEXT: tested rigorously across 6 seeds in optimize_turtle_dual_system.py before
    this was wired in here -- it did NOT reliably beat plain blind search (0 wins, 4 ties, 2
    losses; average Sharpe 0.713 vs 0.760 blind), even with the hedge. It's exposed here as an
    optional, clearly-labeled experiment, not a recommended default -- see the checkbox caption
    in the Optimizer tab for the actual numbers.

    Returns (all_results, n_failed, stage1_best_score_or_None).
    """
    rng = random.Random(worker_module.RANDOM_SEED)
    n_stage1 = max(10, int(n_trials * stage1_fraction))
    n_stage2 = n_trials - n_stage1
    key_fn = {"sharpe": lambda r: r["sharpe"], "calmar": lambda r: r["calmar"],
              "cagr": lambda r: r["cagr"], "total_return": lambda r: r["total_return"]}[rank_metric]

    st.markdown(f"**Stage 1: broad search** ({n_stage1} trials)")
    stage1_params = worker_module.sample_unique_param_sets(
        rng, n_stage1, only_optimize=only_optimize, fixed_values=fixed_values)
    p1 = st.progress(0.0)
    s1 = st.empty()
    t1 = time.time()
    stage1_results, stage1_failed, _ = run_trials_with_progress(
        raw_data, stage1_params, n_workers, p1, s1, max(1, n_stage1 // 40), t1,
        worker_module=worker_module, ath_references=ath_references)

    all_results = list(stage1_results)
    n_failed = stage1_failed
    stage1_best_score = None

    if stage1_results and n_stage2 > 0:
        stage1_sorted = sorted(stage1_results, key=key_fn, reverse=True)
        stage1_best_score = key_fn(stage1_sorted[0])
        top_k_params = [r["params"] for r in stage1_sorted[:min(top_k, len(stage1_sorted))]]
        narrowed_indep, narrowed_always, narrowed_toggles, toggle_bias = worker_module.narrow_param_space(
            top_k_params, window=narrow_window)

        n_stage2_blind = max(0, int(n_stage2 * stage2_blind_fraction))
        n_stage2_refined = n_stage2 - n_stage2_blind
        st.markdown(f"**Stage 2: refined + hedge** ({n_stage2_refined} narrowed + "
                    f"{n_stage2_blind} still fully blind)")
        stage2_params = worker_module.sample_unique_param_sets(
            rng, n_stage2_refined, independent_space=narrowed_indep, always_sampled=narrowed_always,
            conditional_toggles=narrowed_toggles, toggle_bias=toggle_bias,
            only_optimize=only_optimize, fixed_values=fixed_values)
        if n_stage2_blind > 0:
            stage2_params += worker_module.sample_unique_param_sets(
                rng, n_stage2_blind, only_optimize=only_optimize, fixed_values=fixed_values)
        p2 = st.progress(0.0)
        s2 = st.empty()
        t2 = time.time()
        stage2_results, stage2_failed, _ = run_trials_with_progress(
            raw_data, stage2_params, n_workers, p2, s2, max(1, len(stage2_params) // 40), t2,
            worker_module=worker_module, ath_references=ath_references)
        all_results.extend(stage2_results)
        n_failed += stage2_failed

    return all_results, n_failed, stage1_best_score


# ----------------------------------------------------------------------------
# session state defaults (so edits survive reruns within a session)
# ----------------------------------------------------------------------------
if "all_tickers" not in st.session_state:
    st.session_state.all_tickers = dict(td.TICKERS)   # captured ONCE, before anything can mutate td.TICKERS

# Bumped every time the optimizer applies a winning config, and folded into every S1/S2
# widget's OWN key (see system_param_editor) -- NOT just used to clear those
# keys from session_state. Confirmed directly (Streamlit 1.x on Windows): once a number_input has
# actually been mounted in the browser -- which every field here already is, since st.expander's
# content renders into the DOM unconditionally, collapsed or not -- deleting its session_state key
# does NOT reliably make the frontend forget its own last-displayed value; the widget can keep
# showing its old number even though the Python-side session_state.s1_params etc. really did
# update underneath it. Changing the widget's KEY forces Streamlit to treat it as a genuinely NEW
# element (a fresh frontend mount that has no stale value to fall back on), which is the actual
# reliable fix. clear_widget_keys_with_prefix() is still called too, mainly so the old
# generation's now-unreachable keys don't just pile up in session_state forever.
if "bt_widget_gen" not in st.session_state:
    st.session_state.bt_widget_gen = 0

# Bumped only when "Run backtest" actually completes (see that button's handler below) -- the
# _memo() cache key for every derived stat/dataframe in the results section starts with this, so
# an unrelated rerun (any other widget, anywhere in the app) is a cache hit, and a genuinely new
# result is the only thing that invalidates it.
if "bt_result_gen" not in st.session_state:
    st.session_state.bt_result_gen = 0
if "bt_run_history" not in st.session_state:
    st.session_state.bt_run_history = []

if "s1_params" not in st.session_state:
    base = dict(td.DEFAULT_SYSTEM_PARAMS)
    base["breakout_lookback"] = next((s["breakout_lookback"] for s in td.SYSTEMS if s["name"] == "S1"), 20)
    base["exit_lookback"] = next((s["exit_lookback"] for s in td.SYSTEMS if s["name"] == "S1"), 15)
    st.session_state.s1_params = base
if "s2_params" not in st.session_state:
    base = dict(td.s2_SYSTEM_PARAMS)
    base["breakout_lookback"] = next((s["breakout_lookback"] for s in td.SYSTEMS if s["name"] == "S2"), 300)
    base["exit_lookback"] = next((s["exit_lookback"] for s in td.SYSTEMS if s["name"] == "S2"), 20)
    st.session_state.s2_params = base
# Streamlit forbids setting a widget's own key once it exists in the same pass -- the opt-in checkboxes
# (optinc_dual_*, the shared Search setup panel rendered by render_opt_param_panel) already exist
# earlier in the SAME pass than the parameter-stability section's "uncheck non-converged" button, so the button can't set
# them directly. It stashes which param keys to uncheck here instead; this block, which runs
# before any tab/widget exists, is the only safe place to apply it.
if st.session_state.get("wf_pending_uncheck_params"):
    pending = st.session_state.wf_pending_uncheck_params
    for pname in pending.get("params", []):
        st.session_state[f"optinc_{pending.get('engine_key', 'dual')}_{pname}"] = False
    st.session_state.wf_pending_uncheck_params = None

# Same reason: the Optimizer's "send to the Backtest tab" button can't set the Backtest tab's
# exposure box (bt_exp) after that box exists in the pass, so it leaves the value here.
if "pending_bt_exposure" in st.session_state:
    st.session_state.bt_exp = st.session_state.pop("pending_bt_exposure")

# And the same for the per-market risk multipliers (Setup panel, "Risk per market" tab).
if "pending_market_risk" in st.session_state:
    for _asset, _mult in st.session_state.pop("pending_market_risk").items():
        st.session_state[f"riskmult_{_asset}"] = float(_mult)

# The Backtest tab's "Load the S&P 500 trend-regime setup" button leaves a flag instead of setting the widgets
# itself (they already exist in its pass). System S2 becomes a plain moving-average regime held at a fixed
# share of the account: one unit, a risk allowance large enough that the exposure cap sets the size, a wide
# stop, and every other exit and filter off (the file's S2 defaults switch several on, and each of them would
# sell the pullbacks this setup is built to sit through).
SP500_REGIME_S2 = {
    "use_ma_regime": True, "ma_regime_period": 250, "ma_regime_band_pct": 0.0,
    "use_peak_trail": False, "peak_trail_atr_mult": 8.0,
    "risk_per_trade_pct": 20.0, "risk_per_trade_pct_schedule": None, "stop_n_mult": 10.0, "n_atr_period": 20,
    "max_units_per_trend": 1, "use_fixed_stop": True, "ratchet_stop_on_pyramid": True,
    "use_scale_out": False, "use_time_stop": False, "use_n_day_low_trailing_stop": False, "use_breakeven_stop": False,
    "use_chandelier_exit": False, "use_giveback_stop": False, "use_regime_scale_out": False, "use_ema_filter": False,
    "use_choppiness_filter": False, "require_all_time_high": False, "use_cooldown": False,
    "use_pyramid_adx_filter": False, "use_average_entry_stop": False,
}
if st.session_state.pop("pending_sp500_regime", False):
    st.session_state.s2_params = {**st.session_state.s2_params, **SP500_REGIME_S2}
    clear_widget_keys_with_prefix("bt_s1")
    clear_widget_keys_with_prefix("bt_s2")
    st.session_state.bt_widget_gen += 1
    st.session_state.update({"bt_en_s1": False, "bt_en_s2": True, "bt_exp": 150, "bt_port_riskcap_on": False,
                             "wf_en_s1": False, "wf_en_s2": True, "_sp500_regime_part2": True, "carry_on": True})
    for _name in st.session_state.all_tickers:
        st.session_state[f"assets_tick_{_name}"] = (_name == "SP500")

# One shared asset picker in the sidebar, always visible regardless of which tab is active --
# replaces 4 separate ticker_picker() calls (Backtest/Optimizer/Walk-Forward/Regime, one each with
# their own checkbox set) that forced re-picking the same assets per tab. Streamlit forbids
# reusing one widget `key` across multiple st.checkbox() calls in the same rerun, and since every
# tab body below executes every rerun (st.tabs() renders all of them), the picker can only be
# rendered ONCE -- the sidebar is the natural single location for a cross-cutting filter like this.
with st.sidebar:
    st.session_state.selected_tickers = ticker_picker("assets")
    st.markdown("**Run status**")
    run_status_box = st.container()
    # the Setup panel (search parameters and ranges, per-market risk, fees) is written into this slot further down,
    # once its helper functions exist; collapsing the sidebar hides it without un-rendering any of its widgets
    setup_slot = st.container()

with run_status_box:
    for _label, _text in st.session_state.get("last_runs", {}).items():
        st.caption(f"Last {_label.lower()}: {_text}")


def run_progress_widgets(label: str):
    """A progress bar + status line in the left sidebar's Run status box -- always on screen while
    a run is going, whichever tab it was started from. Returns (bar, status)."""
    with run_status_box:
        st.caption(f"{label} running...")
        bar = st.progress(0.0)
        status = st.empty()
    return bar, status


def note_last_run(label: str, text: str):
    st.session_state.setdefault("last_runs", {})[label] = text


# ----------------------------------------------------------------------------
# UNIVERSAL ACCOUNT, BROKER AND COSTS -- one setting for the Backtest, Optimizer and Walk-Forward
# ----------------------------------------------------------------------------
CUSTOM_PROFILE = "Custom flat % costs"
DEFAULT_ACCOUNT = 100_000.0
TAKE_ONE_LABEL = "Take one contract anyway"
SKIP_LABEL = "Skip the trade"
NEXT_OPEN_LABEL = "Next bar's open (realistic)"
SAME_CLOSE_LABEL = "Same bar's close (optimistic)"
SIZING_LABELS = {
    brokers.SIZING_MICRO: "Micro futures (whole contracts)",
    brokers.SIZING_MINI: "Mini / standard futures (whole contracts)",
    brokers.SIZING_CFD: "CFD (whole lots)",
    brokers.SIZING_FRACTIONAL: "Fractional (any size, no rounding)",
}
_ENGINE_COSTS = ibkr_fees.remember_engine_defaults(td)


def broker_profiles() -> dict:
    return brokers.list_profiles()


def selected_broker():
    """(profile name, unedited profile or None for the custom flat-% model, sizing key)."""
    ss = st.session_state
    paths = broker_profiles()
    name = ss.get("broker_profile")
    if name not in paths:
        name = next(iter(paths), CUSTOM_PROFILE) if name != CUSTOM_PROFILE else CUSTOM_PROFILE
    if name == CUSTOM_PROFILE:
        return name, None, brokers.SIZING_FRACTIONAL
    profile = brokers.load(paths[name])
    options = brokers.sizing_options(profile)
    sizing = ss.get("sizing_mode", options[0])
    return name, profile, sizing if sizing in options else options[0]


def current_profile():
    """The chosen broker profile with the Fees tab's table edits applied: (profile or None, sizing)."""
    name, profile, sizing = selected_broker()
    if profile is None:
        return None, sizing
    assets = list(st.session_state.get("all_tickers", td.TICKERS))
    edited = (st.session_state.get(f"broker_table_{name}_{sizing}") or {}).get("edited_rows")
    return brokers.apply_edits(profile, assets, sizing, edited), sizing


@st.cache_data(show_spinner=False)
def cached_tbill_series():
    """US 3-month T-bill yield history for the carry model (a saved copy once downloaded); None when unavailable."""
    return carry_data.tbill_series()


def market_risk_key(asset: str) -> str:
    return f"riskmult_{asset}"


def manual_market_risk(asset: str) -> float:
    """The risk multiplier typed for a market in the Risk per market tab (1.0 = no tilt)."""
    try:
        return max(0.0, float(st.session_state.get(market_risk_key(asset), 1.0)))
    except (TypeError, ValueError):
        return 1.0


def apply_universal_costs():
    """Sets every engine setting that describes the account and broker -- account size, costs,
    position sizing, margin rules, execution timing -- from the Fees tab, once per script run, so
    every run started in this pass (Backtest, Optimizer, Walk-Forward) uses the same ones. Worker
    processes receive them through the portfolio config. Starts from the engine's own defaults every
    time so switching profile never leaves anything behind. Also sets the per-market risk multipliers
    typed in the Risk per market tab (a setting of the account, like the costs: every tab uses it)."""
    import copy
    ss = st.session_state
    for name, value in _ENGINE_COSTS.items():
        setattr(td, name, value)
    for name, value in (("USE_PER_CONTRACT_FEES", False), ("FEE_SCHEDULE", {}), ("ASSET_COSTS", {}),
                        ("SIZE_STEP", {}), ("USE_MARGIN_CALLS", False), ("MARGIN_SPEC", {})):
        setattr(td, name, copy.deepcopy(value))
    td.PORTFOLIO_STARTING_CAPITAL = float(ss.get("acct_capital", DEFAULT_ACCOUNT))
    td.MARK_TO_MARKET = bool(ss.get("acct_mtm", True))
    td.EXECUTION_TIMING = "same_close" if ss.get("exec_timing") == SAME_CLOSE_LABEL else "next_open"
    margin_on = bool(ss.get("margin_calls_on", False))
    stress = float(ss.get("margin_stress", 1.0))
    assets = list(ss.get("all_tickers", td.TICKERS))
    profile, sizing = current_profile()
    td.TAKE_MIN_ONE_UNIT = (sizing in (brokers.SIZING_MICRO, brokers.SIZING_MINI)
                            and ss.get("sizing_take_one", SKIP_LABEL) == TAKE_ONE_LABEL)
    ss.broker_notes = []
    td.ASSET_RISK_MULT = {asset: manual_market_risk(asset) for asset in assets
                          if abs(manual_market_risk(asset) - 1.0) > 1e-12}
    td.ASSET_RISK_NORMALIZE = bool(ss.get("riskmult_normalize", False))
    td.ALLOW_EXIT_ABOVE_ENTRY = bool(ss.get("allow_slow_exit", False))
    td.STOP_SLIPPAGE_PCT = float(ss.get("stop_slip_pct", 0.0))
    if ss.get("carry_on", False):
        td.CARRY_MODE = "futures"
        td.CARRY_RATE_SERIES = cached_tbill_series()
        td.CARRY_RATE_FLAT = float(ss.get("carry_rate_flat", 4.0)) / 100.0
        td.CARRY_DIVIDEND_YIELD = carry_data.dividend_yields(float(ss.get("carry_div_scale", 1.0)), assets)
    else:
        td.CARRY_MODE, td.CARRY_RATE_SERIES, td.CARRY_RATE_FLAT, td.CARRY_DIVIDEND_YIELD = "off", None, 0.0, {}
    if profile is None:
        td.SLIPPAGE_PCT = float(ss.get("fees_slippage", _ENGINE_COSTS["SLIPPAGE_PCT"]))
        td.SPREAD_PCT = float(ss.get("fees_spread", _ENGINE_COSTS["SPREAD_PCT"]))
        td.COMMISSION_PCT = float(ss.get("fees_commission", _ENGINE_COSTS["COMMISSION_PCT"]))
        td.HOLDING_COST_PCT_PER_YEAR = float(ss.get("fees_holding", _ENGINE_COSTS["HOLDING_COST_PCT_PER_YEAR"]))
        if margin_on:
            initial = min(1.0, stress / max(float(ss.get("margin_custom_leverage", 20.0)), 1.0))
            stop_out = float(ss.get("margin_stopout", 50.0)) / 100.0
            td.USE_MARGIN_CALLS = True
            td.MARGIN_SPEC = {asset: (initial, initial * stop_out) for asset in assets}
    else:
        compiled = brokers.compile_profile(profile, assets, sizing, margin_calls=margin_on, margin_stress=stress,
                                           stop_out_pct=ss.get("margin_stopout"))
        for name, value in compiled["engine"].items():
            setattr(td, name, value)
        ss.broker_notes = compiled["notes"]


def cost_model_label() -> str:
    ss = st.session_state
    name, _, sizing = selected_broker()
    parts = [name if name != CUSTOM_PROFILE else "custom % costs"]
    if sizing != brokers.SIZING_FRACTIONAL:
        parts.append(f"{sizing} sizing")
    if ss.get("margin_calls_on", False):
        parts.append("margin calls")
    if td.ASSET_RISK_MULT:
        parts.append("per-market risk" + (" (rescaled)" if td.ASSET_RISK_NORMALIZE else ""))
    if td.CARRY_MODE == "futures":
        parts.append("interest and carry")
    parts.append(f"${float(ss.get('acct_capital', DEFAULT_ACCOUNT)):,.0f} account")
    return ", ".join(parts)


apply_universal_costs()

# The five tabs take the full width; the Setup panel lives in the sidebar (setup_slot). Its parameter selection
# and ranges are shared by the Optimizer and Walk-Forward tabs.
# Streamlit has no sidebar-width setting: give the open sidebar room for the Setup tabs and the fee tables. A minimum
# (not a fixed width) keeps drag-to-resize working, and the rule only applies while it is expanded so it still collapses.
st.html("<style>section[data-testid='stSidebar'][aria-expanded='true']{min-width:min(27rem,90vw) !important;}</style>")
tab_backtest, tab_chart, tab_optimize, tab_walkforward, tab_regime = st.tabs(
    [":material/play_arrow: Backtest", ":material/candlestick_chart: Trade chart", ":material/tune: Optimizer",
     ":material/timeline: Walk-forward", ":material/query_stats: Regime diagnostic"])

# ----------------------------------------------------------------------------
# BACKTEST TAB
# ----------------------------------------------------------------------------
bt_robustness_slot = None   # a spot in the Backtest tab, filled in further down once its function exists

with tab_backtest:
    st.header("Run a backtest")
    st.caption("Every selected asset trades from ONE shared account: a portfolio-wide cap on total "
               "open risk, and drawdown scaling (if enabled) that reacts to the whole account. "
               "Trading costs come from the Fees tab in the sidebar.")

    chosen_tickers = st.session_state.selected_tickers
    st.caption(f"{len(chosen_tickers)} asset(s) selected -- pick assets in the sidebar (shared "
               f"across every tab).")
    if st.button("Load the S&P 500 trend-regime setup", key="bt_preset_sp500",
                 help="For a long-only account on the S&P 500. Selects only the S&P 500, turns S1 off, and makes "
                      "S2 a 250-day moving-average regime: in while the close is above the average, out when it "
                      "closes below, held at 150% of the account, with every other exit and filter off, and turns "
                      "on interest on cash and carry in the Fees tab (a rule that sits in cash part of the time is "
                      "unfairly judged without it). The Optimizer and Walk-Forward then search only the exposure "
                      "(100-200%). In tests on 1965-2026 the regime rule earned about what buy-and-hold did with "
                      "roughly half the drawdown. Keep the exposure at "
                      "or below about 200%: a 25% one-day gap at full exposure then costs at most half the account. "
                      "Leverage fitted to past drawdowns broke its limit on a crash the fitting data had not seen."):
        st.session_state.pending_sp500_regime = True
        st.rerun()

    run_dual = True
    with st.container(border=True, key="card_settings"):
        st.subheader("Strategy settings")
        mh_on = mh_mode_active()
        if mh_on:
            st.success("Monthly-high strategy (picked in the sidebar's Setup panel, Monthly high tab): one system with only the entry, pyramiding, SL and TP below. "
                       "The S1 / S2 settings are not used.", icon=":material/trending_up:")
            st.dataframe(pd.DataFrame(mh_summary_rows()), width="stretch", hide_index=True)
        # the S1 / S2 editors stay rendered inside an expander while the monthly-high strategy is on, so their values are kept for when you switch back
        with (st.expander("Standard S1 / S2 settings (not used while the monthly-high strategy is selected)", expanded=False) if mh_on else st.container()):
            c1, c2 = st.columns(2)
            with c1:
                enable_s1 = st.checkbox("Enable S1", value=True, key="bt_en_s1")
            with c2:
                enable_s2 = st.checkbox("Enable S2", value=True, key="bt_en_s2")
            gen = st.session_state.bt_widget_gen
            st.session_state.s1_params = system_param_editor("S1", st.session_state.s1_params, f"bt_s1_g{gen}")
            st.session_state.s2_params = system_param_editor("S2", st.session_state.s2_params, f"bt_s2_g{gen}")

        c3, c4 = st.columns(2)
        with c3:
            # initial value kept in session state (not passed as value=) so the Optimizer's "send to
            # the Backtest tab" can also set it without Streamlit complaining about two defaults
            if "bt_exp" not in st.session_state:
                st.session_state.bt_exp = int(td.MAX_TOTAL_EXPOSURE_PCT)
            max_exposure = st.number_input("Max total exposure (%)", min_value=10, max_value=20000,
                                             step=50, key="bt_exp")
        with c4:
            starting_capital = st.number_input("Reference capital per asset ($)", 100.0, 10_000_000.0,
                                                 float(td.STARTING_CAPITAL_PER_ASSET), 1000.0, key="bt_cap",
                                                 help="Only used for per-asset buy-and-hold comparisons and "
                                                      "chart scaling. The account the strategy trades with is "
                                                      "the 'Account size' in the Fees tab.")

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.caption(f"Account size: **${float(st.session_state.get('acct_capital', DEFAULT_ACCOUNT)):,.0f}** "
                       "(set in the Fees tab in the sidebar, shared by every tab).")
        with pc2:
            use_portfolio_risk_cap = st.checkbox("Cap total open risk", value=True, key="bt_port_riskcap_on")
        with pc3:
            max_portfolio_risk_pct = st.number_input(
                "Max portfolio risk (%, sum of every open position's risk_amount)",
                0.5, 100.0, float(td.MAX_PORTFOLIO_RISK_PCT), 0.5, key="bt_port_riskcap",
                disabled=not use_portfolio_risk_cap)
        st.caption("'Max total exposure' above still applies too, per asset (S1+S2 combined notional on "
                   "one asset), relative to the shared equity. Drawdown scaling (turned on in the "
                   "Optimizer's global toggles or in turtle_dual_system.py) reacts to the whole account. "
                   "To risk more on some markets than others, use the 'Risk' tab of the Setup panel.")

    run_col, save_col = st.columns(2)
    run_clicked = run_col.button("Run backtest", type="primary", icon=":material/play_arrow:", key="bt_run")
    save_clicked = save_col.button("Save this S1/S2 config to turtle_dual_system.py", key="bt_save",
                                     disabled=not run_dual)

    if save_clicked and run_dual:
        try:
            src_path = os.path.join(os.path.dirname(os.path.abspath(td.__file__)), "turtle_dual_system.py")
            backup_path = src_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy(src_path, backup_path)

            def fmt_dict(d):
                lines = ["{"]
                for k, v in d.items():
                    lines.append(f"    {k!r}: {v!r},")
                lines.append("}")
                return "\n".join(lines)

            src = open(src_path).read()
            import re
            src = re.sub(r"DEFAULT_SYSTEM_PARAMS = \{.*?\n\}", "DEFAULT_SYSTEM_PARAMS = " + fmt_dict(
                {k: v for k, v in st.session_state.s1_params.items() if k not in ("breakout_lookback", "exit_lookback", "name")}),
                src, count=1, flags=re.DOTALL)
            src = re.sub(r"s2_SYSTEM_PARAMS = \{.*?\n\}", "s2_SYSTEM_PARAMS = " + fmt_dict(
                {k: v for k, v in st.session_state.s2_params.items() if k not in ("breakout_lookback", "exit_lookback", "name")}),
                src, count=1, flags=re.DOTALL)
            src = re.sub(
                r'SYSTEMS = \[\s*\{"name": "S1".*?\},\s*\{"name": "S2".*?\},\s*\]',
                f'SYSTEMS = [\n'
                f'    {{"name": "S1", "breakout_lookback": {st.session_state.s1_params["breakout_lookback"]}, '
                f'"exit_lookback": {st.session_state.s1_params["exit_lookback"]}, **DEFAULT_SYSTEM_PARAMS}},\n'
                f'    {{"name": "S2", "breakout_lookback": {st.session_state.s2_params["breakout_lookback"]}, '
                f'"exit_lookback": {st.session_state.s2_params["exit_lookback"]}, **s2_SYSTEM_PARAMS}},\n]',
                src, count=1, flags=re.DOTALL)
            with open(src_path, "w") as f:
                f.write(src)
            st.success(f"Saved. Backup of the previous version: {os.path.basename(backup_path)}")
        except Exception as e:
            st.error(f"Couldn't save automatically: {e}. Nothing was overwritten.")

    if run_clicked:
        if not chosen_tickers:
            st.error("Pick at least one asset.")
        elif not mh_on and not (enable_s1 or enable_s2):
            st.error("Enable at least one of S1 / S2.")
        else:
            # NOTE: chosen_tickers is passed directly to download_data below -- td.TICKERS
            # itself is never reassigned, so the full available list stays intact for every
            # other tab regardless of what's checked here
            with st.spinner("Downloading data and running the backtest..."):
                raw_prices = cached_download_data(tuple(sorted(chosen_tickers.items())), td.START_DATE, td.END_DATE)
                td.MH_CLEAN_MODE = mh_on
                td.SYSTEMS = ([build_clean_system()] if mh_on else
                              build_systems_from_ui(st.session_state.s1_params, st.session_state.s2_params, enable_s1, enable_s2))
                _dd_before = td.USE_DRAWDOWN_SCALING
                if mh_on:
                    td.USE_DRAWDOWN_SCALING = False          # not one of the four mechanics (put back right after this run is recorded)
                td.MAX_TOTAL_EXPOSURE_PCT = max_exposure
                td.STARTING_CAPITAL_PER_ASSET = starting_capital
                td.USE_PORTFOLIO_RISK_CAP = use_portfolio_risk_cap
                td.MAX_PORTFOLIO_RISK_PCT = max_portfolio_risk_pct
                td.PORTFOLIO_STARTING_CAPITAL = float(st.session_state.get("acct_capital", DEFAULT_ACCOUNT))
                dual_ath_refs = {}
                if any(s.get("require_all_time_high") for s in td.SYSTEMS):
                    dual_ath_refs = cached_fetch_all_time_highs(tuple(sorted(chosen_tickers.items())))
                features = {name: td.build_features(df, full_history_ath=dual_ath_refs.get(name))
                             for name, df in raw_prices.items()}
                for name in features:
                    features[name].attrs["asset"] = name   # fee schedule lookup
                bt_bar, bt_status = run_progress_widgets("Backtest")
                bt_start = time.time()
                td.RECORD_HISTORY = True   # the replay below needs the daily record
                try:
                    portfolio_result = td.simulate_portfolio(
                        features, progress_cb=lambda frac: bt_bar.progress(min(frac, 1.0)))
                finally:
                    td.RECORD_HISTORY = False
                bt_bar.progress(1.0)
                bt_status.write(f"{len(features)} assets simulated in {time.time() - bt_start:.0f}s")
                import copy
                st.session_state.bt_run_config = {
                    "systems": copy.deepcopy(td.SYSTEMS), "td": robustness.capture_td_config(),
                    "ath": dual_ath_refs}
                td.USE_DRAWDOWN_SCALING = _dd_before         # the clean run switched it off for itself only
                note_last_run("Backtest", f"{len(features)} assets, {time.time() - bt_start:.0f}s, "
                                          f"{cost_model_label()} costs")

            # stored so switching between the combined/per-asset views below (a separate widget,
            # triggering its own rerun) doesn't re-run the backtest itself -- only the page
            # actually being VIEWED gets its charts built, which is what keeps this fast even
            # with many assets selected
            st.session_state.bt_portfolio_result = portfolio_result
            st.session_state.bt_portfolio_raw = raw_prices
            # New generation -- every _memo()'d stat/dataframe below is keyed off this, so it's
            # what makes a genuinely new result actually invalidate the cache (an unrelated rerun
            # elsewhere in the app reuses the same generation and hits cache instead).
            st.session_state.bt_result_gen += 1
            st.session_state.bt_stat_cache = {}

            if portfolio_result is not None:
                run_stats = td.performance_stats(portfolio_result["equity_curve"],
                                                 portfolio_result["starting_capital"])
                st.session_state.bt_run_history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Assets": len(chosen_tickers),
                    "Fees": cost_model_label(),
                    "S1": (f"monthly high {td.SYSTEMS[0]['mh_min_age']}-{td.SYSTEMS[0]['mh_max_age']}d, EMA {td.SYSTEMS[0]['mh_sma_period']}" if mh_on
                           else f"{st.session_state.s1_params['breakout_lookback']}/{st.session_state.s1_params['exit_lookback']}"),
                    "S2": ("-" if mh_on else f"{st.session_state.s2_params['breakout_lookback']}/{st.session_state.s2_params['exit_lookback']}"),
                    "Exposure %": max_exposure,
                    "Sharpe": round(run_stats["Sharpe"], 2),
                    "CAGR": f"{run_stats['CAGR']*100:.1f}%",
                    "Max DD": f"{run_stats['Max Drawdown']*100:.1f}%",
                    "Final Equity": f"${run_stats['Final Equity']:,.0f}",
                })
                st.session_state.bt_run_history = st.session_state.bt_run_history[-8:]
            st.success("Done.")

    if st.session_state.bt_run_history:
        with st.expander(f"Run history (last {len(st.session_state.bt_run_history)})"):
            st.caption("One row per 'Run backtest' click (combined/portfolio result) -- lets you "
                       "eyeball how the last few runs compared without losing the previous numbers.")
            st.dataframe(pd.DataFrame(st.session_state.bt_run_history), width="stretch",
                          hide_index=True)

    if st.session_state.get("bt_portfolio_result") is not None:
        port = st.session_state.bt_portfolio_result
        port_raw = st.session_state.get("bt_portfolio_raw", {})
        gen = st.session_state.bt_result_gen
        # per-asset share of the portfolio's starting capital -- used ONLY as the buy & hold
        # baseline below, so "Strategy vs Buy & Hold" compares against the same assumption the
        # existing per-asset-mode combined view already makes (equal capital split per asset),
        # not a claim about how the strategy itself allocates risk (it doesn't split evenly --
        # that's the whole point of a shared risk budget)
        per_asset_share_capital = port["starting_capital"] / len(port_raw) if port_raw else 0
        all_trades = [t for asset_trades in port["trades_by_asset"].values() for t in asset_trades]

        st.subheader(f"Results: shared portfolio ({len(port['trades_by_asset'])} assets, one account)")
        tab_ov, tab_yrs, tab_as, tab_tr, tab_cr, tab_rb, tab_rp = st.tabs(
            [":material/dashboard: Overview", ":material/calendar_month: Years", ":material/pie_chart: Assets",
             ":material/receipt_long: Trades", ":material/account_balance: Costs & risk", ":material/casino: Robustness",
             ":material/replay: Replay"])

        with tab_ov:
            combined_bh = (_memo((gen, "combined_bh"), lambda: compute_combined_buy_and_hold(port_raw, per_asset_share_capital))
                           if port_raw else None)
            show_scorecard(port["equity_curve"], combined_bh, port["starting_capital"], all_trades, port=port)
            st.caption(f"Portfolio starting capital: ${port['starting_capital']:,.0f} -- one shared account across every selected "
                       f"asset, not a private pot each. Buy & Hold puts the SAME capital, split evenly, into these assets (not how "
                       f"the strategy itself allocates risk): an apples-to-apples 'was trading worth it at all' comparison.")

        with tab_yrs:
            st.markdown("**Yearly performance -- winning vs. losing years (shared portfolio)**")
            show_yearly_win_loss(port["equity_curve"], port["starting_capital"])

        with tab_as:
            port_page_options = ["Combined Portfolio"] + list(port["trades_by_asset"].keys())
            port_selected_page = st.radio("View results for:", port_page_options, horizontal=True,
                                          key="bt_portfolio_page_selector",
                                          format_func=lambda option: "All assets" if option == "Combined Portfolio" else option)
            if port_selected_page == "Combined Portfolio":
                st.markdown("**Per-asset statistics: strategy contribution vs. that asset's own Buy & Hold**")
                st.caption("Trades/Win Rate/PnL/Profit Factor are the strategy's actual contribution from "
                           "that asset (there's no separate per-asset equity curve in shared-capital mode "
                           "to compute a per-asset Sharpe from -- capital is genuinely pooled). B&H columns "
                           "show what simply buying and holding that one asset alone would have returned "
                           "over the same window, at its equal capital share, for reference. Pick one "
                           "asset above to see its own full breakdown.")
                def _compute_per_asset_rows():
                    rows = []
                    attributed_returns = {}
                    for name, trades in port["trades_by_asset"].items():
                        ep = td.episode_stats(trades)
                        row = {
                            "Asset": name, "Trades": ep["Episodes"],
                            "Win Rate": f"{ep['Win Rate']*100:.1f}%" if np.isfinite(ep["Win Rate"]) else "n/a",
                            "Total PnL": f"${sum(t['pnl'] for t in trades):,.0f}",
                            "Avg R": f"{ep['Avg R']:.2f}" if np.isfinite(ep["Avg R"]) else "n/a",
                            "Profit Factor": f"{ep['Profit Factor']:.2f}" if np.isfinite(ep["Profit Factor"]) else "n/a",
                        }
                        if name in port_raw:
                            bh_curve = td.compute_buy_and_hold(port_raw[name], per_asset_share_capital)
                            bh_stats = td.performance_stats(bh_curve, per_asset_share_capital)
                            row["B&H CAGR"] = f"{bh_stats['CAGR']*100:.2f}%"
                            row["B&H Total Return"] = f"{bh_stats['Total Return']*100:.1f}%"
                            # Attribution proxy for correlation: this asset's own realized-PnL step
                            # curve (same construction as the per-asset drill-down below) -- there's
                            # no true isolated equity curve in shared-capital mode, but daily % moves
                            # of this attributed curve are still the honest answer to "how correlated
                            # were these assets' own contributions," which is what a correlation
                            # matrix is for here (not a claim about the raw underlying instruments).
                            if trades:
                                # cumulative dollars this asset added to the account (closed units plus open profit, day by
                                # day); turned into daily shares of the account's equity in _compute_corr below
                                attributed_returns[name] = asset_contribution(port, name, trades)
                        rows.append(row)
                    return rows, attributed_returns

                per_asset_rows, attributed_returns = _memo((gen, "port_per_asset_rows"), _compute_per_asset_rows)
                st.dataframe(pd.DataFrame(per_asset_rows), width="stretch")

                if len(attributed_returns) >= 2:
                    st.markdown("**Correlation matrix (attributed daily returns)**")
                    st.caption("How correlated each asset's OWN strategy contribution was to the others' "
                               "-- the daily profit and loss of each asset's trades (closed plus still open) as a "
                               "share of the account's equity, not the raw price series. High correlation across "
                               "your picks means less real diversification than the asset count alone suggests.")

                    def _compute_corr():
                        # each asset's daily profit or loss as a share of the account's equity the day before (a % change of
                        # "equal share + profit" was dominated by that tiny base once an asset's profit dwarfed it)
                        ret_df = (pd.DataFrame(attributed_returns).diff()
                                  .div(port["equity_curve"].shift(1), axis=0).replace([np.inf, -np.inf], np.nan).iloc[1:].dropna(how="all"))
                        return ret_df.corr()

                    corr = _memo((gen, "port_correlation"), _compute_corr)
                    try:
                        st.dataframe(corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
                                      width="stretch")
                    except ImportError:
                        st.dataframe(corr.round(2), width="stretch")

            else:
                name = port_selected_page
                trades = port["trades_by_asset"][name]
                st.subheader(name)

                ep = _memo((gen, name, "port_asset_ep"), lambda: td.episode_stats(trades))
                attributed_equity = None   # set below when the price history is there

                if name in port_raw:
                    df_asset = port_raw[name]

                    # There's no TRUE isolated equity curve for one asset in shared-capital mode -- real position sizes
                    # were set against the shared account's equity at the time, not a private pot, so re-simulating the
                    # asset alone would give different trade sizes. What CAN be shown honestly: what this asset's trades
                    # added to the account (asset_contribution: closed units from the day each closed, plus the open
                    # profit of what is still held), taken every day as a share of the account's equity (asset_share_curve).
                    attributed_equity = _memo((gen, name, "port_asset_attributed"),
                                              lambda: asset_share_curve(port, asset_contribution(port, name, trades)))

                    st.markdown(f"**{name}: its contribution to the account vs. Buy & Hold**")
                    st.caption("Strategy line = the account if ONLY this asset's trades had counted: each day this asset's "
                               "profit or loss (closed trades and open positions) is taken as a share of the account's equity "
                               "that day, and those shares are compounded from the account's starting capital. Buy & Hold = "
                               "just holding this one asset at its equal share of the account, compounded the same way. An "
                               "attribution view, not an isolated backtest of this asset (its position sizes came from the "
                               "whole account's equity), so the percentages are what this asset added to the ACCOUNT's growth, "
                               "not a return on a private pot of money. Interest and costs that belong to the account as a "
                               "whole are not in it.")
                    bh_curve = bh_share_curve(port, df_asset, per_asset_share_capital)
                    show_scorecard(attributed_equity, bh_curve, float(port["starting_capital"]), trades,
                                   equity_label="Account if only this asset counted")
                else:
                    st.caption("No price history for this asset, so only its trade statistics are shown.")

                st.markdown("**Full episode breakdown**")
                # Value MUST be uniformly str -- a column mixing plain ints (e.g. "Episodes": 206)
                # with formatted-float strings (e.g. "0.27") isn't Arrow-serializable and crashes
                # st.dataframe's pyarrow conversion (confirmed directly: "Could not convert '0.27'
                # with type str: tried to convert to int64").
                def _fmt_stat(v):
                    if isinstance(v, float):
                        return f"{v:.2f}" if np.isfinite(v) else "n/a"
                    return str(v)
                detail_rows = [{"Metric": k, "Value": _fmt_stat(v)} for k, v in ep.items()]
                st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

                st.markdown("**PnL by exit reason ($)**")
                pnl_breakdown = _memo((gen, name, "port_asset_exit_pnl"), lambda: _exit_reason_pnl(trades))
                st.dataframe(pd.DataFrame([{"Exit Reason": k, "PnL": f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"}
                                              for k, v in pnl_breakdown.items()]),
                              width="stretch", hide_index=True)

                show_monte_carlo_section(f"port_{name}", trades=trades,
                                           equity_curve=attributed_equity)

                asset_episode_df = _memo((gen, name, "port_asset_episode_df"),
                                           lambda: td.build_episode_dataframe({name: {"trades": trades}}))
                if not asset_episode_df.empty:
                    st.markdown(f"**{name} episodes**")
                    st.dataframe(asset_episode_df, width="stretch")
                else:
                    st.caption("No closed episodes for this asset.")


        with tab_tr:
            st.markdown("**PnL by exit reason ($)**")
            all_exit_pnl = _memo((gen, "port_exit_pnl"), lambda: _exit_reason_pnl(all_trades))
            st.dataframe(pd.DataFrame([{"Exit Reason": k, "PnL": f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"} for k, v in all_exit_pnl.items()]),
                         width="stretch", hide_index=True)
            episode_df = _memo((gen, "port_episode_df"), lambda: td.build_episode_dataframe(
                {name: {"trades": trades} for name, trades in port["trades_by_asset"].items()}))
            if not episode_df.empty:
                st.markdown("**All episodes (shared portfolio)**")
                st.dataframe(episode_df, width="stretch")

        with tab_cr:
            show_account_realism(port)
            st.markdown("**Data quality of the price history behind this run**")
            st.caption("The strategy relies on daily highs and lows (stops, N) and on opens (gap fills). Yahoo's "
                       "continuous futures often carry no real range or stale opens, which makes stops and "
                       "volatility look better than they would have been. Treat results from assets flagged "
                       "here with care.")
            st.dataframe(pd.DataFrame(_memo((gen, "data_quality"), lambda: td.data_quality(port_raw))),
                         hide_index=True, width="stretch")

        with tab_rb:
            show_monte_carlo_section("port_combined", equity_curve=port["equity_curve"], series_mode=True,
                                     title="Monte Carlo robustness check: whole portfolio")
            bt_robustness_slot = st.container()

        with tab_rp:
            show_portfolio_replay(port, gen)

# ----------------------------------------------------------------------------
# TRADE CHART TAB
# ----------------------------------------------------------------------------
with tab_chart:
    st.header("Trade chart")
    if not LIGHTWEIGHT_CHARTS_AVAILABLE:
        st.error("streamlit-lightweight-charts isn't installed. Run: "
                 "pip install streamlit-lightweight-charts --break-system-packages")
    else:
        # The shared-portfolio result's per-asset slices (trades_by_system[asset],
        # exit_markers_by_asset[asset]) are reshaped into the small dict build_lightweight_chart_data
        # needs for ONE asset.
        chart_sources = {}
        port = st.session_state.get("bt_portfolio_result")
        port_raw = st.session_state.get("bt_portfolio_raw", {})
        if port is not None:
            for name in port["trades_by_asset"]:
                if name not in port_raw:
                    continue
                chart_sources[name] = (
                    {"trades_by_system": port["trades_by_system"].get(name, {}),
                     "exit_markers_by_system": port["exit_markers_by_asset"].get(name, {})},
                    port_raw[name])

        if not chart_sources:
            st.info("Run a backtest on the Backtest tab first -- this tab reads from those results.")
        else:
            chosen_label = st.selectbox("Asset", list(chart_sources.keys()), key="chart_asset_pick")
            r, df_for_chart = chart_sources[chosen_label]
            candles, markers, position_lines = _memo(
                (st.session_state.bt_result_gen, chosen_label, "chart_data"),
                lambda: build_lightweight_chart_data(df_for_chart, r))
            shown_labels = chart_label_controls("chart")
            markers, position_lines = filter_chart_labels(markers, position_lines, shown_labels)

            box_style = st.radio(
                "Position boxes", ["Stepped stop (like the TradingView script)", "Original (first unit's stop and take-profit)"], horizontal=True,
                key="chart_box_style",
                help="Stepped: per trade a green area from the average entry up to the close (to the exit price at the end) and a red area from the average entry down to the stop, drawn in steps "
                     "so it shows where the stop was after every add (teal once the stop is above the average entry = profit locked in). Original: one box per trade from the FIRST unit's stop "
                     "and the highest high reached.")
            chart_years = st.number_input(
                "Years of position boxes and monthly highs to draw (0 = all)", 0, 40, 3, key="chart_box_years",
                help="Each trade and each monthly high is its own series on the chart; drawing 20 years of them can make it slow, so only the most recent years are drawn by default. Markers always cover everything.")
            since_ts = None if not chart_years else df_for_chart.index[-1] - pd.DateOffset(years=int(chart_years))
            _sess = st.session_state
            _run_systems = {s_["name"]: s_ for s_ in (_sess.get("bt_run_config") or {}).get("systems", [])}
            sys_params_now = _run_systems or {"S1": _sess.get("s1_params", {}), "S2": _sess.get("s2_params", {})}      # the systems as the backtest ran them
            sys_enabled = {n_: True for n_ in _run_systems} if _run_systems else {"S1": _sess.get("bt_en_s1", True), "S2": _sess.get("bt_en_s2", True)}
            mh_sets = {}
            for _name, _p in sys_params_now.items():
                _cfg = td.monthly_high_add_settings(_p) if sys_enabled.get(_name, True) else None
                if _cfg is not None:
                    mh_sets[(_cfg[0], _cfg[1], float(_p.get("entry_buffer_n_mult", 0.0)), int(_p.get("n_atr_period", 20)))] = _name
            show_mh_lines = st.checkbox(
                "Show a line at every monthly high (dotted = too young to trade, solid = old enough; each ends at its breakout)", value=bool(mh_sets), key="chart_show_mh_lines",
                disabled=not mh_sets,
                help="Only available when a system uses the monthly-high entry (Entry 3) or the monthly-high adds. The settings used are the ones currently set for that system on the Backtest tab.")
            show_position_lines = st.checkbox(
                "Show entry/SL/TP position boxes (like TradingView's position tool)", value=True,
                key="chart_show_position_lines",
                help="A shaded box per trade, bounded to just that trade's own entry-to-exit "
                     "dates -- green from entry up to the take-profit level, red from entry down "
                     "to the stop, with a thin gray dashed line at entry. For pyramided Dual "
                     "System trades, SL/TP reflect the FIRST unit's own frozen risk/reward "
                     "definition, not every later add's own ratcheted stop (which would clutter "
                     "the chart with a near-duplicate box per add).")

            with st.expander("ℹ️ How to read this chart"):
                st.caption("Drag to pan, scroll/pinch to zoom. Blue = Entry, light blue = Pyramid Add, "
                           "red = SL Exit, green = TP Exit, gray = Other Exit (channel/time-stop/"
                           "exposure-trim/end-of-data). Each label starts with the system it's from "
                           "(e.g. 'S1 Entry', 'S2 Add') -- if this is a [Dual] chart, S1 and S2 markers "
                           "are overlaid on the same candles, so check the label if something looks off. "
                           "Entry/Add labels show 'risk X%' -- that unit's risk as a % of equity at the "
                           "time it opened. Exit labels show +/-X'R' (P&L as a multiple of what was "
                           "risked -- +2R means it made twice what it risked) and the true +/-X% equity "
                           "impact (P&L as a % of TOTAL account equity at entry, not the position's own "
                           "capital -- a position can be leveraged well above its risked %, so 'P&L as a "
                           "% of the money put into the trade' can look dramatic without meaning much for "
                           "the actual account; this is the number that does).")

            chart_options = {
                "height": 650,
                "layout": chart_layout(),
                "timeScale": {"borderColor": "rgba(197, 203, 206, 0.8)", "timeVisible": False},
                "rightPriceScale": {"borderColor": "rgba(197, 203, 206, 0.8)"},
                "grid": {"vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
                          "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}},
            }
            candlestick_series = [{
                "type": "Candlestick",
                "data": candles,
                "options": {
                    "upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False,
                    "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
                },
                "markers": markers,
            }]
            if show_position_lines and box_style.startswith("Stepped"):
                stepped_boxes = _memo((_sess.bt_result_gen, chosen_label, "stepped_boxes", int(chart_years), str(sorted(str(v) for v in sys_params_now.values()))[:200]),
                                      lambda: build_stepped_position_boxes(df_for_chart, r, {k: v for k, v in sys_params_now.items()}, since_ts))
                box_series = filter_chart_labels([], stepped_boxes, shown_labels)[1]
            else:
                box_series = position_lines if show_position_lines else []
            mh_series = []
            if show_mh_lines and mh_sets:
                for (_mn, _mx, _buf, _np), _nm in mh_sets.items():
                    mh_series += build_monthly_high_lines(df_for_chart, _mn, _mx, _buf, _np, since_ts)
            all_series = candlestick_series + box_series + mh_series

            renderLightweightCharts([{"chart": chart_options, "series": all_series}],
                                      key=f"chart_{chosen_label}_{box_style[:4]}_{int(chart_years)}_{int(show_mh_lines)}")

# ----------------------------------------------------------------------------
# OPTIMIZER: per-parameter opt-in/opt-out catalog
# ----------------------------------------------------------------------------
# Maps every leaf parameter name in each optimizer's search space to (category, label) -- the
# categories mirror system_param_editor's own tabs so this panel reads as
# "the same parameters, just choosing which ones to search" rather than a separate vocabulary.
# A parameter left unchecked here is NOT sampled at all: every trial pins it at whatever the
# Backtest tab currently has it set to (see dual_fixed_values below) -- so
# "opt out" means "hold this exactly where I've already set it," not "use some other default."
DUAL_OPT_CATALOG = {
    # the entry types are grouped and numbered the same way as in the Backtest tab's "Entry type" picker; the exits and N apply to every entry type
    "Entry 1 · N-day-high breakout (default)": [
        ("S1_BREAKOUT_LOOKBACK", "Entry 1 · S1 breakout lookback (days)"), ("S2_BREAKOUT_LOOKBACK", "Entry 1 · S2 breakout lookback (days)"),
        ("entry_buffer_n_mult", "Entry 1 & 3 · Entry buffer (x N)"),
    ],
    "Entry 2 · EMA break": [
        ("use_ema_entry", "Entry 2 · EMA break instead (on/off)"),
        ("ema_entry_period", "Entry 2 · EMA length (days)"), ("ema_entry_mode", "Entry 2 · type (cross / above)"),
    ],
    "Entry 3 · Monthly-high breakout": [
        ("use_monthly_high_entry", "Entry 3 · Monthly-high breakout instead (on/off)"),
        ("mh_min_age", "Entry 3 · high at least this old (days)"), ("mh_max_age", "Entry 3 · high at most this old (days)"),
        ("mh_sma_period", "Entry 3 · close above its average of (days)"), ("mh_ma_type", "Entry 3 · average type (ema / sma)"),
    ],
    "Older entry · moving-average regime": [
        ("use_ma_regime", "Moving-average regime instead of breakouts (on/off)"),
        ("ma_regime_period", "Moving-average length (days)"), ("ma_regime_band_pct", "Moving-average band (%)"),
    ],
    "Exits and N (every entry)": [
        ("S1_EXIT_LOOKBACK", "S1 exit lookback (days)"), ("S2_EXIT_LOOKBACK", "S2 exit lookback (days)"),
        ("use_ema_exit", "EMA exit instead of the exit lookback (on/off)"), ("ema_exit_period", "EMA exit length (days)"),
        ("n_atr_period", "N (ATR) period"),
    ],
    "Position sizing": [
        ("risk_per_trade_pct", "Risk per trade (%)"), ("risk_per_trade_pct_schedule", "Graduated risk schedule"),
        ("MAX_TOTAL_EXPOSURE_PCT", "Max total exposure (%)"),
        ("USE_DRAWDOWN_SCALING", "Drawdown-based risk scaling (on/off)"),
        ("DRAWDOWN_SCALING_THRESHOLD", "Drawdown scaling threshold"),
        ("DRAWDOWN_SCALING_FACTOR", "Drawdown scaling factor"),
    ],
    "Pyramiding": [
        ("pyramid_method", "Pyramid method"), ("pyramid_spacing_n_mult", "Pyramid spacing (x N)"),
        ("pyramid_breakout_lookback", "Pyramid breakout lookback"), ("swing_lookback", "Swing lookback"),
        ("max_units_per_trend", "Max units per trend"),
        ("use_pyramid_adx_filter", "Block pyramid adds above ADX (on/off)"),
        ("pyramid_max_adx", "Max ADX to allow an add"),
    ],
    "Stops": [
        ("stop_n_mult", "Stop (x N)"), ("use_fixed_stop", "Fixed stop -- core ratcheting SL (on/off)"),
        ("ratchet_stop_on_pyramid", "Move stop on each pyramid add (on/off)"),
        ("use_average_entry_stop", "Anchor to average entry, not latest (on/off)"),
        ("use_n_day_low_trailing_stop", "N-day-low trailing stop (on/off)"),
        ("trailing_stop_lookback_days", "Trailing stop lookback"),
        ("use_breakeven_stop", "Breakeven stop (on/off)"), ("breakeven_trigger_r", "Breakeven trigger (R)"),
        ("use_chandelier_exit", "Chandelier exit (on/off)"), ("chandelier_atr_mult", "Chandelier (x ATR)"),
        ("use_peak_trail", "Peak-trail stop, resting (on/off)"), ("peak_trail_atr_mult", "Peak trail (x ATR)"),
        ("use_giveback_stop", "Per-unit profit give-back stop (on/off)"),
        ("giveback_fraction", "Give-back fraction"), ("giveback_min_profit_r", "Give-back activate above (R)"),
    ],
    "Take-profit / time exits": [
        ("tp_atr_period", "TP's own ATR period"), ("tp_n_mult", "TP (x its own N)"),
        ("use_scale_out", "Scale-out at fixed R levels (on/off)"),
        ("use_time_stop", "Time stop (on/off)"), ("time_stop_days", "Time stop days"),
        ("time_stop_min_r", "Time stop min R"),
        ("use_regime_scale_out", "Regime-triggered scale-out (on/off)"),
        ("regime_scale_out_adx_threshold", "Regime scale-out ADX threshold"),
        ("regime_scale_out_fraction", "Regime scale-out fraction"), ("adx_period", "ADX period"),
    ],
    "Entry filters": [
        ("use_ema_filter", "EMA trend filter (on/off)"), ("ema_period", "EMA period"),
        ("use_choppiness_filter", "Choppiness Index filter (on/off)"),
        ("choppiness_period", "Choppiness Index period"), ("choppiness_max", "Max Choppiness Index"),
        ("require_all_time_high", "Require genuine all-time high (on/off)"),
        ("use_cooldown", "Cooldown after a loss (on/off)"), ("cooldown_days", "Cooldown days"),
    ],
}

def _opt_inc_key(engine_key: str, param_key: str) -> str:
    return f"optinc_{engine_key}_{param_key}"


def _opt_fix_key(engine_key: str, param_key: str) -> str:
    return f"optfix_{engine_key}_{param_key}"


_FIXED_VALUE_CHOICES = {
    "pyramid_method": ["spacing", "breakout", "swing_high", "monthly_high"],
    "ema_entry_mode": ["cross", "above"],
    "mh_ma_type": ["ema", "sma"],
    "risk_per_trade_pct_schedule": [None, (0.5, 1.0), (0.5, 1.5), (1.0, 2.0),
                                      (0.25, 0.5, 1.0), (0.25, 0.5, 1.0, 1.5)],
}


def _fixed_value_label(value) -> str:
    return "flat (no schedule)" if value is None else str(value)


# The recommended Dual-System search setup, from the 3,000-trial sensitivity run and the
# walk-forward comparison: search the 10 parameters that drive the strategy, and pin everything
# else to a plain, stable value (every optional exit/filter/overlay off). It is the default state
# of the shared Search setup panel; "Reset to recommended" restores it.
RECOMMENDED_DUAL_SEARCH = {
    "S1_BREAKOUT_LOOKBACK", "S1_EXIT_LOOKBACK", "S2_BREAKOUT_LOOKBACK", "S2_EXIT_LOOKBACK",
    "n_atr_period", "stop_n_mult", "risk_per_trade_pct", "MAX_TOTAL_EXPOSURE_PCT",
    "max_units_per_trend", "pyramid_spacing_n_mult",
}
RECOMMENDED_DUAL_PINS = {
    "entry_buffer_n_mult": 0.5, "pyramid_method": "spacing", "risk_per_trade_pct_schedule": None,
    "use_fixed_stop": True, "ratchet_stop_on_pyramid": True,
    "use_chandelier_exit": False, "use_giveback_stop": False, "use_breakeven_stop": False,
    "use_n_day_low_trailing_stop": False, "use_scale_out": False, "use_time_stop": False,
    "use_regime_scale_out": False, "use_ema_filter": False, "use_choppiness_filter": False,
    "require_all_time_high": False, "use_cooldown": False, "USE_DRAWDOWN_SCALING": True,
    "use_average_entry_stop": False, "use_pyramid_adx_filter": False, "use_ma_regime": False, "use_peak_trail": False,
}
_NO_PIN = object()   # None is a real pin (flat risk schedule), so "no default" needs its own marker


def _default_included(engine_key: str, key: str) -> bool:
    """Starting tick state of a parameter's checkbox: the recommended set for the Dual System
    panel, everything ticked for any other panel."""
    return key in RECOMMENDED_DUAL_SEARCH if engine_key == "dual" else True


def _render_fixed_value_input(engine_key: str, key: str, backtest_value, default_pin=_NO_PIN):
    """Optional 'pin this to a specific value' input for an unchecked parameter. With no default
    pin it starts empty = hold at the Backtest tab's current value (the original behavior); with
    one it starts at that value. Types without a sensible free-form input (e.g. tuples outside the
    known schedule choices) are skipped."""
    widget_key = _opt_fix_key(engine_key, key)
    placeholder = f"Backtest tab: {_fixed_value_label(backtest_value)}"
    choices = _FIXED_VALUE_CHOICES.get(key)
    if choices is None and isinstance(backtest_value, bool):
        choices = [True, False]
    has_pin = default_pin is not _NO_PIN
    if choices is not None:
        start = choices.index(default_pin) if has_pin and default_pin in choices else None
        st.selectbox("Fixed value", range(len(choices)), index=start, key=widget_key + "_sel",
                     format_func=lambda i: _fixed_value_label(choices[i]),
                     placeholder=placeholder, label_visibility="collapsed")
    elif isinstance(backtest_value, int):
        st.number_input("Fixed value", value=default_pin if has_pin else None, step=1, key=widget_key,
                        placeholder=placeholder, label_visibility="collapsed")
    elif isinstance(backtest_value, float):
        st.number_input("Fixed value", value=float(default_pin) if has_pin else None, step=0.05,
                        format="%g", key=widget_key, placeholder=placeholder, label_visibility="collapsed")


def dual_fixed_overrides(engine_key: str = "dual") -> dict:
    """Values the user explicitly pinned in the opt-in panel for parameters left unchecked."""
    if mh_mode_active():
        return {}                    # the monthly-high tab pins its own values (see dual_fixed_values)
    overrides = {}
    for cat in DUAL_OPT_CATALOG.values():
        for key, _ in cat:
            if st.session_state.get(_opt_inc_key(engine_key, key), _default_included(engine_key, key)):
                continue
            widget_key = _opt_fix_key(engine_key, key)
            picked_idx = st.session_state.get(widget_key + "_sel")
            if picked_idx is not None:
                choices = _FIXED_VALUE_CHOICES.get(key, [True, False])
                overrides[key] = choices[picked_idx]
                continue
            picked = st.session_state.get(widget_key)
            if picked is None:
                continue
            overrides[key] = picked
    return overrides


def render_opt_param_panel(engine_key: str, catalog: dict, fixed_defaults: dict = None, n_cols: int = 2,
                             show_recommended_reset: bool = False):
    """Renders the categorized 'optimize this parameter?' checkboxes for one engine. Each
    checkbox's OWN key (via _opt_inc_key) is the single source of truth -- read directly at run
    time, no separate dict to keep in sync. Defaults to checked (= current, full-search
    behavior) so nothing changes for anyone who never opens this panel.

    fixed_defaults (optional): {param: Backtest-tab value}. When given, each UNCHECKED parameter
    also shows an input to pin it to a specific value instead of the Backtest tab's."""
    all_keys = [key for cat in catalog.values() for key, _ in cat]
    default_pins = RECOMMENDED_DUAL_PINS if engine_key == "dual" else {}

    def clear_pin_widgets():
        for key in all_keys:
            st.session_state.pop(_opt_fix_key(engine_key, key), None)
            st.session_state.pop(_opt_fix_key(engine_key, key) + "_sel", None)

    bcol1, bcol2 = st.columns(2) if n_cols == 1 else st.columns([1, 1, 4])[:2]
    with bcol1:
        if st.button("Optimize all", key=f"optinc_{engine_key}_all"):
            for key in all_keys:
                st.session_state[_opt_inc_key(engine_key, key)] = True
    with bcol2:
        if st.button("Optimize none", key=f"optinc_{engine_key}_none"):
            for key in all_keys:
                st.session_state[_opt_inc_key(engine_key, key)] = False
            clear_pin_widgets()
    if show_recommended_reset and st.button(
            "Reset to recommended setup", key=f"optinc_{engine_key}_recommended",
            help="Search the 10 parameters that drive the strategy (entry/exit lookbacks, N period, "
                 "stop, risk, exposure cap, max units, pyramid spacing) and pin everything else to "
                 "plain values with every optional exit, filter and overlay off."):
        for key in all_keys:
            st.session_state[_opt_inc_key(engine_key, key)] = _default_included(engine_key, key)
        clear_pin_widgets()
    n_included = sum(1 for key in all_keys if st.session_state.get(
        _opt_inc_key(engine_key, key), _default_included(engine_key, key)))
    st.caption(f"{n_included}/{len(all_keys)} parameters set to optimize -- the rest stay fixed "
               f"at the pinned value shown under them (or the Backtest tab's value if none).")
    for category, entries in catalog.items():
        with st.expander(category, expanded=False):
            cols = st.columns(n_cols)
            for idx, (key, label) in enumerate(entries):
                with cols[idx % n_cols]:
                    included = st.checkbox(label, value=_default_included(engine_key, key),
                                           key=_opt_inc_key(engine_key, key))
                    if not included and fixed_defaults is not None and key in fixed_defaults:
                        _render_fixed_value_input(engine_key, key, fixed_defaults[key],
                                                  default_pins.get(key, _NO_PIN))


def dual_only_optimize_set(engine_key: str = "dual") -> set:
    """engine_key: which opt-in panel's checkboxes to read -- "dual" is the shared Search setup
    panel used by both the Optimizer and Walk-Forward tabs (render_opt_param_panel renders one
    checkbox set per engine_key -- Streamlit would raise a duplicate-widget-id error if two
    panels tried to reuse the same key)."""
    if mh_mode_active():
        return {key for key, _, _, _, _ in MH_PARAMS if st.session_state.get(f"mh_s_{key}", True)}
    chosen = {key for cat in DUAL_OPT_CATALOG.values() for key, _ in cat
              if st.session_state.get(_opt_inc_key(engine_key, key), _default_included(engine_key, key))}
    if engine_key == "dual":
        chosen |= {td.ASSET_RISK_PARAM_PREFIX + asset for asset in searched_markets()}
    return chosen


def dual_fixed_values(enable_s1: bool = None, enable_s2: bool = None) -> dict:
    """Fixed value for any Dual parameter NOT opted into optimization -- sourced live from the
    Backtest tab's current config, so 'opt out' always means 'hold at what's on that tab right
    now,' not some separate hardcoded default.

    enable_s1/enable_s2 (optional): which system's config to source SHARED fixed params from (see
    the fallback logic below) -- None (the default) reads the Backtest tab's own bt_en_s1/bt_en_s2
    checkboxes, for every existing call site. Pass explicit values when the caller has its OWN
    independent S1/S2 toggle (e.g. the Walk-Forward tab's "Test S1"/"Test S2" checkboxes) so the
    fixed-value source matches THAT toggle, not whatever the Backtest tab happens to have --
    otherwise walk-forward testing "S2 only" could still silently source shared fixed params from
    S1's leftover config if the Backtest tab itself has S1 enabled, the exact bug this fallback
    was already fixed for once, just for a different caller."""
    if mh_mode_active():
        # the monthly-high strategy: every searchable name is either sampled (ticked on its tab) or held at the value typed there
        return {**mh_fixed_values(), "use_monthly_high_entry": True,
                "MAX_TOTAL_EXPOSURE_PCT": st.session_state.get("bt_exp", td.MAX_TOTAL_EXPOSURE_PCT)}
    s1, s2 = st.session_state.s1_params, st.session_state.s2_params
    fixed = {
        "S1_BREAKOUT_LOOKBACK": s1["breakout_lookback"], "S1_EXIT_LOOKBACK": s1["exit_lookback"],
        "S2_BREAKOUT_LOOKBACK": s2["breakout_lookback"], "S2_EXIT_LOOKBACK": s2["exit_lookback"],
        # bt_exp is the Backtest tab's live widget value -- NOT td.MAX_TOTAL_EXPOSURE_PCT, which
        # only gets written when "Run backtest" is actually clicked. Reading the widget directly
        # means changing this field and going straight to the Optimizer tab (without re-running
        # the backtest first) still pins it at what's actually shown on screen, not a stale value
        # left over from whenever the backtest last ran.
        "MAX_TOTAL_EXPOSURE_PCT": st.session_state.get("bt_exp", td.MAX_TOTAL_EXPOSURE_PCT),
        "USE_DRAWDOWN_SCALING": td.USE_DRAWDOWN_SCALING,
        "DRAWDOWN_SCALING_THRESHOLD": td.DRAWDOWN_SCALING_THRESHOLD,
        "DRAWDOWN_SCALING_FACTOR": td.DRAWDOWN_SCALING_FACTOR,
    }
    # everything else is shared between S1/S2 when fixed (same as when it's sampled) -- normally
    # S1's current config is the source of truth for it, EXCEPT when S1 is actually disabled on
    # the Backtest tab (only S2 active): sourcing from S1 there would silently apply S1's own
    # leftover config to the only system that's actually running, which is exactly backwards.
    # Confirmed as a real, reproducible bug: a user with S1 disabled, S2's pyramid_method set to
    # "spacing", and "Pyramid method" left unchecked in the opt-in panel still got a DIFFERENT
    # pyramid_method back, because this fallback ignored enable state and always read S1's.
    if enable_s1 is None:
        enable_s1 = st.session_state.get("bt_en_s1", True)
    if enable_s2 is None:
        enable_s2 = st.session_state.get("bt_en_s2", True)
    source = s2 if (not enable_s1 and enable_s2) else s1
    for key, _ in [entry for cat in DUAL_OPT_CATALOG.values() for entry in cat]:
        if key not in fixed and key in source:
            fixed[key] = source[key]
    if market_risk_search_on():
        # a market that is not searched keeps the multiplier typed for it
        for asset in st.session_state.get("selected_tickers", {}):
            fixed[td.ASSET_RISK_PARAM_PREFIX + asset] = manual_market_risk(asset)
    return fixed


# Every module-level global a trial's apply_params() (opt.apply_params) OR this tab's own
# portfolio settings handler can mutate on turtle_dual_system.
# optimize_turtle_dual_system.py's OWN standalone main() carefully snapshots and restores these
# around its search (see its own original_systems/original_config) -- but turtle_ui.py bypasses
# main() and drives sample_unique_param_sets/run_trials_with_progress/run_joint_trial directly,
# so it never inherited that restore. Confirmed as a REAL bug, not just a theoretical one: joint
# mode (run_joint_trial, called directly in a for-loop, NEVER via a worker pool) and any
# sequential fallback single-engine run mutate td.SYSTEMS/USE_DRAWDOWN_SCALING/etc. directly in
# THIS process on every trial, and whichever trial happened to run LAST left its own sampled
# config stuck as the live module state -- including for a toggle the user had explicitly turned
# OFF and then left OUT of the opt-in panel (meaning "hold this fixed"): the "fixed" value itself
# was being read from module state that a PRIOR run's last trial had already flipped back on.
_TD_GLOBAL_NAMES = ("SYSTEMS", "MAX_TOTAL_EXPOSURE_PCT", "USE_DRAWDOWN_SCALING",
                     "DRAWDOWN_SCALING_THRESHOLD", "DRAWDOWN_SCALING_FACTOR",
                     "PORTFOLIO_STARTING_CAPITAL", "USE_PORTFOLIO_RISK_CAP", "MAX_PORTFOLIO_RISK_PCT",
                     "ASSET_RISK_MULT", "ASSET_RISK_NORMALIZE", "MH_CLEAN_MODE")
# ENABLE_S1/ENABLE_S2 live on the optimizer module itself (optimize_turtle_dual_system.py), not
# td -- they default to True/True in the file and were NEVER synced from the Backtest tab's own
# "Enable S1"/"Enable S2" checkboxes, so the optimizer always searched/scored BOTH systems
# regardless of what's actually enabled there. Snapshotted/restored the same way as the td/ts
# globals above, for the same reason (a run must not leave this "stuck on" for a later run either).
_OPT_GLOBAL_NAMES = ("ENABLE_S1", "ENABLE_S2")


def snapshot_engine_globals() -> dict:
    return {
        "td": {name: getattr(td, name) for name in _TD_GLOBAL_NAMES},
        "opt": {name: getattr(opt, name) for name in _OPT_GLOBAL_NAMES},
    }


def restore_engine_globals(snapshot: dict):
    for name, value in snapshot["td"].items():
        setattr(td, name, value)
    for name, value in snapshot.get("opt", {}).items():
        setattr(opt, name, value)
    opt.set_search_space(opt.DEFAULT_SEARCH_SPACE)   # a run may have applied custom search ranges


def build_param_sensitivity_table(results: list, metric_key: str, params_field: str = "params") -> pd.DataFrame:
    """For every parameter that actually VARIED across a random search's own trials, buckets
    trials by that parameter's value and reports the average/median score per value -- a quick
    "which values of this parameter tended to score better" read straight off the search's own
    output, no extra runs needed. Purely correlational (every other parameter is free-varying at
    the same time in a random search, not held fixed), so this is a lead to investigate -- e.g.
    by fixing that value in the opt-in panel and re-running -- not a controlled-experiment proof."""
    if not results:
        return pd.DataFrame()
    all_names = set()
    for r in results:
        all_names.update(r[params_field].keys())
    rows = []
    for pname in sorted(all_names):
        buckets = {}
        for r in results:
            val = r[params_field].get(pname)
            if val is None:
                continue
            key = str(val) if isinstance(val, (list, tuple)) else val
            buckets.setdefault(key, []).append(r[metric_key])
        if len(buckets) < 2:
            continue   # never varied across these trials -- nothing to compare
        for val, scores in buckets.items():
            rows.append({"parameter": pname, "value": val, "n_trials": len(scores),
                          "avg_score": float(np.mean(scores)), "median_score": float(np.median(scores))})
    return pd.DataFrame(rows)


def show_param_sensitivity_section(results: list, metric_key: str, rank_metric: str,
                                     params_field: str = "params", key_prefix: str = ""):
    """Renders the parameter-sensitivity view as an expander: pick one parameter, see its values
    ranked by average score, with a bar chart. Collapsed by default -- deeper-dive diagnostic."""
    with st.expander("Parameter sensitivity (which values tended to score better?)", icon=":material/bar_chart:"):
        sens = build_param_sensitivity_table(results, metric_key, params_field)
        if sens.empty:
            st.caption("No parameter varied across these trials (likely everything was held fixed).")
            return
        st.caption(f"Every searched parameter that actually varied across the {len(results)} trials "
                   f"above, bucketed by value and averaged by {rank_metric}. Correlational, not "
                   f"causal -- every OTHER parameter was also free-varying at the same time, so a "
                   f"value that looks best here might just be riding along with some other good "
                   f"parameter choice. Use it as a lead (try fixing that value in the opt-in panel "
                   f"and re-running), not a final answer.")
        param_names = sorted(sens["parameter"].unique())
        chosen_param = st.selectbox("Parameter", param_names, key=f"{key_prefix}_sens_param")
        sub = sens[sens["parameter"] == chosen_param].sort_values("avg_score", ascending=False).copy()
        chart_df = sub.set_index(sub["value"].astype(str))[["avg_score"]]
        st.bar_chart(chart_df)
        sub["avg_score"] = sub["avg_score"].round(3)
        sub["median_score"] = sub["median_score"].round(3)
        st.dataframe(
            sub[["value", "n_trials", "avg_score", "median_score"]].rename(
                columns={"value": "Value", "n_trials": "# Trials",
                          "avg_score": f"Avg {rank_metric}", "median_score": f"Median {rank_metric}"}),
            width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# SEARCH SETUP PANEL (right column) -- shared by the Optimizer and Walk-Forward tabs
# ----------------------------------------------------------------------------
_CATEGORICAL_SPACE_KEYS = {"pyramid_method", "risk_per_trade_pct_schedule", "ema_entry_mode", "mh_ma_type"}


def _space_locations(space: dict) -> dict:
    """{param name: (dict name, parent toggle or None)} for every parameter that has a list of
    candidate values -- top-level parameters, and the sub-parameters under on/off toggles."""
    found = {}
    for dname in ("independent", "always"):
        for key in space[dname]:
            found[key] = (dname, None)
    for dname in ("conditional", "global"):
        for toggle, subs in space[dname].items():
            for key in subs:
                found[key] = (dname, toggle)
    return found


def _space_values(space: dict, key: str, loc: tuple) -> list:
    dname, toggle = loc
    return space[dname][key] if toggle is None else space[dname][toggle][key]


def _set_space_values(space: dict, key: str, loc: tuple, values: list):
    dname, toggle = loc
    if toggle is None:
        space[dname][key] = values
    else:
        space[dname][toggle][key] = values


def _fmt_space_value(value) -> str:
    return "flat (no schedule)" if value is None else str(value)


def _parse_range_text(text: str, original: list) -> list:
    """'10, 15, 20' -> [10, 15, 20]. Whole-number lists stay whole numbers (a lookback must be an
    integer); results are de-duplicated and sorted ascending (the search treats them as ordered)."""
    import re
    tokens = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    if not tokens:
        raise ValueError("enter at least one value")
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in original)
    must_be_positive = min(original) > 0
    values = []
    for token in tokens:
        x = float(token)
        if not np.isfinite(x):
            raise ValueError(f"'{token}' is not a number")
        if all_int and x != int(x):
            raise ValueError(f"'{token}' must be a whole number")
        if x < 0 or (must_be_positive and x <= 0):
            raise ValueError(f"'{token}' must be {'above' if must_be_positive else 'at least'} 0")
        values.append(int(x) if all_int or ("." not in token and x == int(x)) else x)
    return sorted(set(values))


def current_search_space() -> tuple:
    """The search ranges the Setup panel currently describes, as (space, errors): the
    defaults from optimize_turtle_dual_system.py with every edited parameter replaced. An entry
    that can't be parsed falls back to its default and is listed in errors."""
    import copy
    if mh_mode_active():
        return mh_search_space()
    defaults = opt.DEFAULT_SEARCH_SPACE
    space = copy.deepcopy(defaults)
    errors = []
    for key, loc in _space_locations(defaults).items():
        default_values = _space_values(defaults, key, loc)
        if key in _CATEGORICAL_SPACE_KEYS:
            picked = st.session_state.get(f"space_{key}_ms")
            if picked:
                _set_space_values(space, key, loc, [default_values[i] for i in picked])
            elif picked is not None:
                errors.append(f"{key}: pick at least one option")
        else:
            text = st.session_state.get(f"space_{key}")
            if text is None:
                continue
            try:
                _set_space_values(space, key, loc, _parse_range_text(text, default_values))
            except ValueError as e:
                errors.append(f"{key}: {e}")
    if market_risk_search_on():
        # every selected market is in the space: the searched ones are sampled, the others pinned to their typed value
        candidates, error = market_risk_candidates()
        if error:
            errors.append(error)
        space["asset_risk"] = {asset: list(candidates) for asset in st.session_state.get("selected_tickers", {})}
    return space, errors


def mh_search_space() -> tuple:
    """(space, errors) for the monthly-high strategy: ONLY its names. A name that is not ticked is pinned by dual_fixed_values; a ticked one is drawn from its candidate list."""
    space = {"independent": {"MAX_TOTAL_EXPOSURE_PCT": [int(st.session_state.get("bt_exp", td.MAX_TOTAL_EXPOSURE_PCT))]},
             "always": {}, "conditional": {"use_monthly_high_entry": {}}, "global": {}, "asset_risk": {}}
    errors = []
    values = mh_fixed_values()
    for key, label, _, default, default_range in MH_PARAMS:
        candidates = [values[key]]
        if st.session_state.get(f"mh_s_{key}", True):
            try:
                candidates = _parse_range_text(st.session_state.get(f"mh_r_{key}", default_range), [default])
            except ValueError as e:
                errors.append(f"{label}: {e} -- the default list is used instead")
                candidates = _parse_range_text(default_range, [default])
        if key.startswith("mh_"):
            space["conditional"]["use_monthly_high_entry"][key] = candidates
        else:
            space["always"][key] = candidates
    space["always"]["risk_per_trade_pct"] = [values["risk_per_trade_pct"]]
    space["always"]["max_units_per_trend"] = [values["max_units_per_trend"]]
    return space, errors


def render_monthly_high_panel():
    """The Setup panel's 'Monthly high' tab: pick the strategy, set its eight numbers, tick which of them the Optimizer / Walk-forward search."""
    ss = st.session_state
    pending = ss.pop("pending_mh_values", None)          # a trial sent over from the Optimizer / Walk-forward (the widgets did not exist yet when it was sent)
    if pending:
        for key, value in pending.items():
            ss[f"mh_v_{key}"] = int(value) if isinstance(MH_DEFAULTS.get(key), int) else float(value)
    # initial values live in session state (not passed as value=), so a trial can also be sent into these widgets without Streamlit complaining about two defaults
    for key, _, _, default, default_range in MH_PARAMS:
        ss.setdefault(f"mh_v_{key}", default)
        ss.setdefault(f"mh_s_{key}", True)
        ss.setdefault(f"mh_r_{key}", default_range)
    ss.setdefault("mh_v_risk_per_trade_pct", MH_RISK_DEFAULT)
    st.radio("Strategy the Backtest, Optimizer and Walk-forward tabs run", [MH_MODE_STANDARD, MH_MODE_CLEAN], key="mh_mode",
             help="Standard: the S1 / S2 settings with all their options. Monthly-high: ONE system with only the mechanics listed below; everything else (channel exit, trailing / "
                  "chandelier / break-even / time stops, filters, cooldown, drawdown scaling, S2) is off, and the Optimizer / Walk-forward search only the ticked numbers below.")
    if not mh_mode_active():
        st.caption("Pick the monthly-high strategy above to run only the four mechanics below. The numbers are kept while the standard strategy is selected.")
    else:
        st.success("Monthly-high strategy selected: the Backtest, Optimizer and Walk-forward tabs run only the mechanics below.", icon=":material/check_circle:")
    st.markdown(
        "**Initial entry**: the close is above the EMA, and it is the first time the close is above a monthly high (plus the entry buffer, in ATR) that is at least *min* and at most *max* days old.  \n"
        "**Pyramiding**: one more position on every further signal with the same criteria.  \n"
        "**SL**: ATR period x multiplier under the newest position's entry; every new position moves the stop up to its own stop.  \n"
        "**TP**: each position closes completely at its entry + multiplier x ATR(TP period).  \n"
        "Nothing else is used.")
    head = st.columns([0.9, 1.2, 2.6])
    head[0].caption("Search it")
    head[1].caption("Value")
    head[2].caption("Values to try")
    last_group = None
    for key, label, group, default, default_range in MH_PARAMS:
        if group != last_group:
            st.markdown(f"**{group}**")
            last_group = group
        st.caption(label)
        c1, c2, c3 = st.columns([0.9, 1.2, 2.6])
        with c1:
            searched = st.checkbox("Search", key=f"mh_s_{key}", label_visibility="collapsed")
        with c2:
            if isinstance(default, int):
                st.number_input("Value", min_value=1, max_value=5000, step=1, key=f"mh_v_{key}", label_visibility="collapsed")
            else:
                st.number_input("Value", min_value=MH_MIN.get(key, 0.1), max_value=100.0, step=0.5, format="%g", key=f"mh_v_{key}", label_visibility="collapsed")
        with c3:
            st.text_input("Values to try", key=f"mh_r_{key}", label_visibility="collapsed", disabled=not searched)
    st.markdown("**Position size** (needed to trade, not a strategy rule)")
    st.number_input("Risk per position (% of the account)", min_value=0.05, max_value=10.0, step=0.25, format="%g", key="mh_v_risk_per_trade_pct",
                    help="Every position (the first and each pyramided one) is sized so that its stop loss costs this much of the account. There is no limit on the number of "
                         "positions; the exposure and open-risk limits on the Backtest tab still apply.")
    st.caption("Ticked = the Optimizer / Walk-forward pick that number from the list to the right; unticked = held at the value in the middle. "
               "Total exposure and the open-risk cap come from the Backtest tab, costs from the Fees tab.")


def render_search_range_editor():
    """One input per searchable parameter, pre-filled with the default candidate values the search
    samples from: edit the list (e.g. '10, 15, 20, 30') to narrow, widen or move that parameter's
    search range. Pick-lists for the categorical ones. Reset puts every range back."""
    defaults = opt.DEFAULT_SEARCH_SPACE
    locations = _space_locations(defaults)
    if st.button("Reset all ranges to defaults", key="space_reset"):
        for key in locations:
            st.session_state.pop(f"space_{key}", None)
            st.session_state.pop(f"space_{key}_ms", None)
    st.caption("The values each parameter is searched over. Edit a list to narrow, widen or move "
               "its range; a single value fixes it. Only matters for parameters ticked in "
               "'Parameters'. Applies to both the Optimizer and Walk-Forward.")
    st.checkbox("Let a system's exit channel be longer than its entry channel", key="allow_slow_exit",
                help="Off (the Turtle pairing): the search only tries systems that leave on a shorter channel "
                     "than they enter on, e.g. in on a 55-day high, out on a 20-day low. On: any pair, e.g. in "
                     "on a 20-day high and out only on a 100-day low, which keeps a long-only account invested "
                     "through ordinary dips. Widen the exit lookback lists below to reach long exits (the "
                     "defaults stop at 35 days for S1 and 130 for S2). Also lifts the same limit in the "
                     "Backtest tab's robustness test.")
    for category, entries in DUAL_OPT_CATALOG.items():
        rows = [(key, label) for key, label in entries if key in locations]
        if not rows:
            continue
        with st.expander(category, expanded=False):
            for key, label in rows:
                default_values = _space_values(defaults, key, locations[key])
                if key in _CATEGORICAL_SPACE_KEYS:
                    st.multiselect(label, options=list(range(len(default_values))),
                                   default=list(range(len(default_values))), key=f"space_{key}_ms",
                                   format_func=lambda i, vals=default_values: _fmt_space_value(vals[i]))
                else:
                    text = st.text_input(label, value=", ".join(str(v) for v in default_values),
                                         key=f"space_{key}")
                    try:
                        _parse_range_text(text, default_values)
                    except ValueError as e:
                        st.caption(f":red[{e} -- the default range is used instead]")


DEFAULT_MARKET_RISK_CHOICES = "0.5, 1, 1.5, 2"


def market_search_key(asset: str) -> str:
    """The 'search this market's multiplier' tick box. It follows the optimizer's own tick-box key
    pattern so the Walk-Forward's 'uncheck the parameters that never converged' button reaches it too."""
    return _opt_inc_key("dual", td.ASSET_RISK_PARAM_PREFIX + asset)


def market_risk_search_on() -> bool:
    return bool(st.session_state.get("riskmult_search_on", False))


def market_risk_candidates() -> tuple:
    """(the multipliers the search chooses from, an error message or None)."""
    text = st.session_state.get("riskmult_choices", DEFAULT_MARKET_RISK_CHOICES)
    try:
        return [float(v) for v in _parse_range_text(text, [0.5, 1.0])], None
    except ValueError as e:
        fallback = [float(v) for v in _parse_range_text(DEFAULT_MARKET_RISK_CHOICES, [0.5, 1.0])]
        return fallback, f"Risk per market choices: {e} -- the default list is used instead"


def searched_markets() -> list:
    """Selected markets whose multiplier the Optimizer / Walk-Forward should search."""
    if not market_risk_search_on():
        return []
    return [asset for asset in st.session_state.get("selected_tickers", {})
            if st.session_state.get(market_search_key(asset), True)]


def _reset_market_risk():
    for asset in st.session_state.get("all_tickers", td.TICKERS):
        st.session_state[market_risk_key(asset)] = 1.0


def render_market_risk_panel():
    """One risk multiplier per market (every tab uses them), and the option to let the Optimizer and
    Walk-Forward search them."""
    ss = st.session_state
    assets = list(ss.get("all_tickers", td.TICKERS))
    selected = set(ss.get("selected_tickers", {}))
    st.caption("How much risk each market gets compared with the others. A signal's risk is the strategy's "
               "'Risk per trade' times the market's multiplier: 1.5 puts 1.5x as much at stake, 0.5 half "
               "as much, 0 never trades that market. The usual limits (exposure caps, portfolio risk cap, "
               "margin) still apply on top.")
    st.button("Reset all to 1.0", key="riskmult_reset", on_click=_reset_market_risk)
    st.checkbox("Rescale so the average multiplier is 1.0", key="riskmult_normalize",
                help="On: the multipliers of the markets in a run are divided by their average, so the mix "
                     "only moves risk BETWEEN markets and the total stays what 'Risk per trade' says. Off: "
                     "a multiplier of 2 on everything is simply twice the leverage.")
    search_on = st.checkbox(
        "Let the Optimizer and Walk-Forward search each market's multiplier", key="riskmult_search_on",
        help="Each ticked market gets its own multiplier, chosen from the list below; the unticked markets "
             "keep the number typed here. Every searched market is one more parameter fitted on the same "
             "history.")
    if search_on:
        st.text_input("Multipliers to choose from", value=DEFAULT_MARKET_RISK_CHOICES, key="riskmult_choices")
        _, error = market_risk_candidates()
        if error:
            st.caption(f":red[{error}]")
        if not ss.get("riskmult_normalize", False):
            st.caption(":orange[Not rescaled: a search that ranks by CAGR will tend to pick the biggest "
                       "multiplier everywhere, which is just more leverage, not a better mix. Tick the "
                       "rescale box above to search for the mix alone.]")
        st.caption("Tested before adding this: Walk-Forward on 15 markets, ranked by CAGR, 3 seeds, with every "
                   "market searched over 0.5 / 1 / 1.5 / 2. Out-of-sample CAGR came out lower than with no "
                   "multipliers (22% rescaled, 19% not rescaled, against 25%), while the in-sample CAGR rose. "
                   "The picks did not repeat from one window or seed to the next, which is what fitting noise "
                   "looks like. One parameter per market is a lot to fit on a few years of history; setting "
                   "the multipliers by hand, from a view you hold, avoids that.")
    head = st.columns([3, 2.4, 1.6] if search_on else [3, 4])
    head[0].markdown("**Market**")
    head[1].markdown("**Multiplier**")
    if search_on:
        head[2].markdown("**Search**")
    for asset in assets:
        row = st.columns([3, 2.4, 1.6] if search_on else [3, 4])
        row[0].markdown(asset if asset in selected else f":gray[{asset}]")
        ss.setdefault(market_risk_key(asset), 1.0)
        row[1].number_input(asset, 0.0, 10.0, step=0.25, key=market_risk_key(asset), label_visibility="collapsed")
        if search_on:
            ss.setdefault(market_search_key(asset), True)
            row[2].checkbox("search", key=market_search_key(asset), label_visibility="collapsed",
                            disabled=asset not in selected)
    chosen = [a for a in assets if a in selected]
    if ss.get("riskmult_normalize", False) and chosen:
        eff = td.asset_risk_multipliers(chosen, {a: manual_market_risk(a) for a in chosen}, True)
        if any(abs(v - 1.0) > 1e-9 for v in eff.values()):
            st.caption("After rescaling, for the markets selected now: "
                       + ", ".join(f"{a} {v:.2f}" for a, v in eff.items()))
    st.caption("Greyed markets are not selected in the sidebar. With a search on, the Optimizer and "
               "Walk-Forward results list the multiplier each trial chose.")


def _on_broker_change():
    """Switching broker resets the sizing choice to one that exists for it (runs before the rerun,
    which is the only time another widget's state may be changed)."""
    ss = st.session_state
    paths = broker_profiles()
    name = ss.get("broker_profile")
    ss.sizing_mode = brokers.sizing_options(brokers.load(paths[name]))[0] if name in paths \
        else brokers.SIZING_FRACTIONAL


def render_fees_panel():
    """Account, broker, position sizing, margin and cost settings -- the Backtest, Optimizer and
    Walk-Forward all use exactly these (see apply_universal_costs near the top of the file)."""
    ss = st.session_state
    assets = list(ss.get("all_tickers", td.TICKERS))
    st.caption("One account, broker and set of costs for the Backtest, Optimizer and Walk-Forward.")
    st.number_input("Account size ($)", 1000.0, 1_000_000_000.0, DEFAULT_ACCOUNT, 10_000.0, key="acct_capital",
                    help="The starting equity of the one shared account. With whole contracts or lots it "
                         "decides which trades are possible at all: a small account cannot size most signals.")
    st.selectbox("Broker profile", list(broker_profiles()) + [CUSTOM_PROFILE], key="broker_profile",
                 on_change=_on_broker_change,
                 help="Every broker-specific number (fees, spreads, margin, lot sizes) comes from a profile "
                      "file in the brokers folder. Edit the numbers below to match your account, and save "
                      "them as a new profile to reuse. A new broker is just a new profile.")
    name, base, sizing = selected_broker()
    if base is not None:
        with st.expander("About this profile"):
            st.caption(base.get("notes", ""))
        options = brokers.sizing_options(base)
        sizing = st.selectbox("Position sizing", options, index=options.index(sizing), key="sizing_mode",
                              format_func=SIZING_LABELS.get,
                              help="Whole contracts / lots: sizes are rounded DOWN to what the broker "
                                   "allows, so a small account may be unable to take a signal. "
                                   "Fractional: any size, the original engine behavior.")
        if sizing in (brokers.SIZING_MICRO, brokers.SIZING_MINI):
            st.radio("When the risk-based size is less than one contract", [SKIP_LABEL, TAKE_ONE_LABEL],
                     key="sizing_take_one", horizontal=True,
                     help="Skipping is the honest answer for a small account: the trade the strategy wanted "
                          "does not exist. Taking one contract anyway risks more than the setting says "
                          "(still limited by the exposure cap and by margin).")
    st.selectbox("Trades are filled at", [NEXT_OPEN_LABEL, SAME_CLOSE_LABEL], key="exec_timing",
                 help="The strategy decides on a bar's close. Realistic: the order fills at the next "
                      "bar's open (entries, pyramid adds and channel exits; stop exits always fill at "
                      "the stop or the open). Optimistic: fills at that same close, which no one can do.")
    st.checkbox("Count open positions in account equity (mark to market)", value=True, key="acct_mtm",
                help="On: the equity curve, drawdowns, position sizing and the exposure cap include the "
                     "unrealized profit or loss of open positions, as a real account would. Off: equity "
                     "only changes when a trade is closed, so a position deep under water is invisible "
                     "until it is sold. Always on while margin calls are on.")
    margin_on = st.checkbox(
        "Margin calls", value=False, key="margin_calls_on",
        help="A new position is only opened (or shrunk) to the size the account can pay the initial "
             "margin for. At each daily close, if account equity is below the maintenance margin of "
             "everything open, the broker closes whole positions, the biggest margin user first, until "
             "it is covered. Checked on closing prices only, not intraday lows.")
    if margin_on:
        m1, m2 = st.columns(2)
        with m1:
            st.number_input("Margin stress (x)", 0.5, 10.0, 1.0, 0.25, key="margin_stress",
                            help="Multiplies every initial margin. Exchanges and brokers raise margins sharply "
                                 "in crises, which is when margin calls happen; try 2-3.")
        if base is None or base["kind"] == "cfd":
            with m2:
                st.number_input("Stop-out (% of margin used)", 5.0, 100.0,
                                float(base["defaults"].get("stop_out_pct", 50.0)) if base else 50.0, 5.0,
                                key="margin_stopout",
                                help="The broker closes positions when equity falls to this % of the margin in "
                                     "use (50% is typical, and the ESMA rule for retail clients).")
        if base is None:
            st.number_input("Leverage (x)", 1.0, 500.0, 20.0, 1.0, key="margin_custom_leverage")
    st.number_input("Extra slippage on stop fills (%)", 0.0, 5.0, 0.0, 0.05, key="stop_slip_pct", format="%.2f",
                    help="A stop order fills at its level or beyond it; in a fast market, beyond. This is extra slippage, "
                         "as a % of the stop level, on every stop-type exit that fills at its level (a gap through the stop "
                         "already fills at the open). 0 fills exactly at the level. The strategy research used 0.25.")
    carry_on = st.checkbox(
        "Interest on cash and carry on positions", value=False, key="carry_on",
        help="Off: the account reports price returns only (idle cash earns nothing, dividends and the cost of "
             "leverage are ignored). On, futures-style: each calendar day the account earns the US T-bill rate on "
             "its whole equity, and each open position pays (T-bill rate minus the market's dividend yield) on its "
             "market value. A position worth 100% of equity then earns its dividend, one worth 200% pays the "
             "T-bill rate on the extra 100%, and cash earns the T-bill rate. This matters most for rules that sit "
             "in cash part of the time and for leveraged setups. CFD swap costs are separate (the holding cost "
             "in the broker table).")
    if carry_on:
        k1, k2 = st.columns(2)
        with k1:
            st.number_input("Dividend yields (x defaults)", 0.0, 3.0, 1.0, 0.1, key="carry_div_scale",
                            help="Scales the built-in yields (S&P 500 2.7%, Nasdaq 0.9%, Dow 2.5%, Russell 1.3%, "
                                 "Ger40 2.5%, Nikkei 1.7%; commodities and crypto pay none).")
        with k2:
            st.number_input("Interest rate without history (%)", 0.0, 20.0, 4.0, 0.25, key="carry_rate_flat",
                            help="Used before the saved T-bill history starts (1960), or when it cannot be loaded.")
        if cached_tbill_series() is None:
            st.caption(":orange[No T-bill history could be loaded: the flat rate above is used for every date.]")
    if base is None:
        slip_default, spread_default = _ENGINE_COSTS["SLIPPAGE_PCT"], _ENGINE_COSTS["SPREAD_PCT"]
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Spread (% of price)", 0.0, 5.0, float(spread_default), 0.001, format="%.4f",
                            key="fees_spread",
                            help="Full bid-ask spread as a % of price; half is paid on each side. "
                                 "Points of spread / index price x 100.")
            st.number_input("Commission (% per side)", 0.0, 5.0, float(_ENGINE_COSTS["COMMISSION_PCT"]),
                            0.0005, format="%.4f", key="fees_commission",
                            help="Commission per side as a % of trade value.")
        with c2:
            st.number_input("Slippage (% per side)", 0.0, 5.0, float(slip_default), 0.001, format="%.4f",
                            key="fees_slippage")
            st.number_input("Holding cost (% / year)", 0.0, 50.0,
                            float(_ENGINE_COSTS["HOLDING_COST_PCT_PER_YEAR"]), 0.1, format="%.2f",
                            key="fees_holding",
                            help="Overnight swap, or a swap-free account's admin fee, as a yearly % "
                                 "of open position value.")
        st.caption("Flat percentages for every asset, whole-size rounding off. Pick a broker profile for "
                   "per-asset costs, contract sizes and margin.")
        return
    rows = brokers.table_rows(base, assets, sizing)
    st.markdown("**Costs and margin per asset**")
    st.caption("Edit any number to match your account; an edited row counts as unverified. Spread and "
               "slippage are % of price per side (half the spread on each side).")
    disabled = ["Asset", "Contract" if base["kind"] == "futures" else "Symbol", "Verified"]
    st.data_editor(pd.DataFrame(rows), key=f"broker_table_{name}_{sizing}", hide_index=True,
                   width="stretch", disabled=disabled, height=300)
    for note in ss.get("broker_notes", []):
        st.warning(note)
    st.caption("Not modelled: roll costs, data fees, currency conversion, intraday margin checks.")
    save_name = st.text_input("Save these numbers as a new profile", key="broker_save_name",
                              placeholder="e.g. My IBKR account")
    if st.button("Save profile", key="broker_save") and save_name.strip():
        edited, _ = current_profile()
        brokers.save(edited, save_name.strip())
        st.success(f"Saved as '{save_name.strip()}' -- pick it in the profile list.")


def render_search_setup_panel():
    with st.container(border=True, key="card_setup"):
        st.subheader("Setup")
        st.caption("Shared by the Backtest, Optimizer and Walk-forward tabs.")
        mh_tab, params_tab, ranges_tab, risk_tab, fees_tab = st.tabs(
            [":material/trending_up: Monthly high", ":material/tune: Parameters", ":material/linear_scale: Ranges", ":material/balance: Risk", ":material/payments: Fees"])
        with mh_tab:
            render_monthly_high_panel()
        with params_tab:
            if mh_mode_active():
                st.info("Not used while the monthly-high strategy is selected -- its numbers are on the Monthly high tab.", icon=":material/info:")
            st.caption("Which parameters get searched, and what the others are pinned to.")
            render_opt_param_panel("dual", DUAL_OPT_CATALOG, fixed_defaults=(None if mh_mode_active() else dual_fixed_values()), n_cols=1,
                                   show_recommended_reset=True)
        with ranges_tab:
            if mh_mode_active():
                st.info("Not used while the monthly-high strategy is selected -- its candidate values are on the Monthly high tab.", icon=":material/info:")
            render_search_range_editor()
        with risk_tab:
            render_market_risk_panel()
        with fees_tab:
            render_fees_panel()
        _, space_errors = current_search_space()
        for message in space_errors:
            st.warning(message)


if st.session_state.pop("_sp500_regime_part2", False):
    # second half of the S&P 500 regime setup, once the search catalog exists and before the panel's widgets do:
    # search only the exposure (the average's length and band stay at the values loaded into S2), over 100-200%
    for _cat in DUAL_OPT_CATALOG.values():
        for _key, _ in _cat:
            st.session_state[_opt_inc_key("dual", _key)] = (_key == "MAX_TOTAL_EXPOSURE_PCT")
    st.session_state["space_MAX_TOTAL_EXPOSURE_PCT"] = "100, 150, 200"

with setup_slot:
    render_search_setup_panel()


def trial_market_risk(params: dict) -> dict:
    """{market: multiplier} a trial's parameters carry (empty when the multipliers were not searched)."""
    prefix = td.ASSET_RISK_PARAM_PREFIX
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def show_trial_market_risk(params: dict):
    mults = trial_market_risk(params)
    if not mults:
        return
    effective = td.asset_risk_multipliers(list(mults), mults)   # the rescale, if it is on, as the engine applies it
    rows = [{"Market": a, "Multiplier chosen": mults[a], "Used in the run": round(effective[a], 3)}
            for a in sorted(mults, key=lambda a: -mults[a])]
    with st.expander("Risk multiplier chosen for each market"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def send_trial_to_backtest(params: dict, rank: int):
    """Loads a trial's S1/S2 settings (and its exposure cap) into the Backtest tab. The Backtest
    widgets were already created earlier in this pass, so their keys can't be set directly: the
    S1/S2 editors are rebuilt under a new generation key, and the exposure cap goes through a
    pending value applied at the top of the next run. Reruns immediately so the Backtest tab is
    rebuilt from the new values."""
    if mh_mode_active():
        # the monthly-high strategy has no S1 / S2: the trial's numbers go into the Monthly high tab (applied when it is drawn on the next pass)
        st.session_state.pending_mh_values = {k: params[k] for k in list(MH_DEFAULTS) + ["risk_per_trade_pct"] if k in params}
    else:
        for system in opt.params_to_systems(params):
            flat = {k: v for k, v in system.items() if k != "name"}
            if system["name"] == "S1":
                st.session_state.s1_params = flat
            elif system["name"] == "S2":
                st.session_state.s2_params = flat
        clear_widget_keys_with_prefix("bt_s1")
        clear_widget_keys_with_prefix("bt_s2")
        st.session_state.bt_widget_gen += 1
    if params.get("MAX_TOTAL_EXPOSURE_PCT") is not None:
        st.session_state.pending_bt_exposure = int(params["MAX_TOTAL_EXPOSURE_PCT"])
    market_mults = trial_market_risk(params)
    if market_mults:
        st.session_state.pending_market_risk = market_mults   # into the Risk per market tab
    st.session_state.opt_sent_note = (f"Trial rank {rank} sent to the Backtest tab -- switch tabs to see it "
                                      f"loaded there. Not saved to disk until you click Save on that tab.")
    st.rerun()


def _prepare_trial_environment():
    """The same engine setup the optimizer's own run uses (which of S1/S2 is active, portfolio
    scoring with the Backtest tab's capital and risk cap). Call inside a snapshot/restore."""
    opt.ENABLE_S1 = st.session_state.get("bt_en_s1", True)
    opt.ENABLE_S2 = st.session_state.get("bt_en_s2", True)
    apply_mh_mode_to_engine()
    opt.SCORE_MODE = "portfolio"
    td.PORTFOLIO_STARTING_CAPITAL = float(st.session_state.get("acct_capital", DEFAULT_ACCOUNT))
    td.USE_PORTFOLIO_RISK_CAP = st.session_state.get("bt_port_riskcap_on", True)
    td.MAX_PORTFOLIO_RISK_PCT = st.session_state.get("bt_port_riskcap", td.MAX_PORTFOLIO_RISK_PCT)


def resimulate_trial(params: dict, tickers: dict) -> dict:
    """Re-runs one optimizer trial's parameters as a full shared-portfolio backtest and returns
    the portfolio result (trades, equity curve), so tests that need the trades can use it."""
    snapshot = snapshot_engine_globals()
    try:
        _prepare_trial_environment()
        key = tuple(sorted(tickers.items()))
        raw = cached_download_data(key, td.START_DATE, td.END_DATE)
        ath = cached_fetch_all_time_highs(key)
        opt.apply_params(params)
        features = {}
        for name, df in raw.items():
            features[name] = td.build_features(df, full_history_ath=ath.get(name))
            features[name].attrs["asset"] = name
        return td.simulate_portfolio(features)
    finally:
        restore_engine_globals(snapshot)


def run_param_robustness(params: dict, tickers: dict, rank_metric: str, n_variants: int, n_workers: int):
    """Scores parameter sets one step away from `params` (only the parameters the search actually
    tuned) with the optimizer's own scoring. Returns (number of variants, list of their scores)."""
    snapshot = snapshot_engine_globals()
    try:
        _prepare_trial_environment()
        opt.RANK_METRIC = rank_metric
        opt.set_search_space(current_search_space()[0])
        variants = monte_carlo.neighbor_params(params, opt.get_search_space(), dual_only_optimize_set(),
                                               n_variants, is_valid=opt.is_valid_params)
        if not variants:
            return 0, []
        key = tuple(sorted(tickers.items()))
        raw = cached_download_data(key, td.START_DATE, td.END_DATE)
        ath = cached_fetch_all_time_highs(key)
        results, _ = opt._run_trial_batch(variants, raw, ath, int(n_workers), "robustness", time.time())
        return len(variants), [r[rank_metric] for r in results]
    finally:
        restore_engine_globals(snapshot)


def show_trial_robustness_checks(best: dict, tickers: dict, rank_metric: str, picked_score: float, n_workers: int,
                                 all_sharpes=None):
    """Expander under an optimizer result with the two checks that need more than the stored
    scores: a trade-level Monte Carlo on the re-run trial, and the parameter-nudge test."""
    signature = repr(sorted(best["params"].items(), key=lambda kv: kv[0]))
    with st.expander("Monte Carlo and robustness checks for the selected trial", icon=":material/casino:"):
        st.markdown("**1. Trade-level Monte Carlo**")
        st.caption("Re-runs this trial's parameters as a full backtest, then replays its trades "
                   "reshuffled, resampled or with some skipped.")
        if st.button("Re-run this trial and run the Monte Carlo test", key="opt_mc_run"):
            with st.spinner("Re-running the selected trial..."):
                port = resimulate_trial(best["params"], tickers)
            st.session_state.opt_mc = {"sig": signature, "curve": port["equity_curve"]}
        mc_state = st.session_state.get("opt_mc")
        if mc_state and mc_state["sig"] == signature:
            render_monte_carlo("opt_trial", equity_curve=mc_state["curve"],
                               years=_curve_years(mc_state["curve"]), series_mode=True)

        st.markdown("**2. Selection bias -- is this just the best of many tries?**")
        st.caption("Picking the top trial out of many always flatters it: even strategies with no edge "
                   "produce a good-looking best. This compares the chosen trial's Sharpe with what the "
                   "best of this many luck-only tries would show (the deflated Sharpe ratio).")
        n_obs, skew, kurt = int(252 * 30), 0.0, 3.0
        mc_curve = (st.session_state.get("opt_mc") or {}).get("curve")
        if mc_curve is not None:
            rets = monte_carlo.equity_returns(mc_curve)
            if len(rets) > 100:
                from scipy import stats as _stats
                n_obs, skew, kurt = len(rets), float(_stats.skew(rets)), float(_stats.kurtosis(rets, fisher=False))
        deflated = (monte_carlo.deflated_sharpe(best["sharpe"], all_sharpes, n_obs, skew, kurt)
                    if all_sharpes is not None else None)
        if deflated is None:
            st.caption("Needs the Sharpe of every trial tried.")
        else:
            prob, luck_sharpe = deflated
            q1, q2, q3 = st.columns(3)
            q1.metric("Chosen trial's Sharpe", f"{best['sharpe']:.2f}")
            q2.metric(f"Best-of-{len(all_sharpes)} luck-only Sharpe", f"{luck_sharpe:.2f}",
                      help="The Sharpe the top of this many tries would be expected to show by luck alone, "
                           "given how much the tried trials' Sharpes vary.")
            q3.metric("Chance it beats pure luck", f"{prob*100:.0f}%",
                      help="Deflated Sharpe ratio. Near 100%: clearly better than luck. Around 50% or "
                           "below: indistinguishable from the best of many random tries. Assumes the "
                           "tries were independent; a guided search like Optuna concentrates its tries, "
                           "so this is optimistic. Uses the simulated returns' own skew and tail "
                           "weight once you have run the trade-level test above, otherwise assumes "
                           "normal returns over about 30 years.")

        st.markdown("**3. Parameter robustness -- does a small nudge wreck it?**")
        st.caption("Scores parameter sets one step away from the selected one (each searched numeric "
                   "parameter moves to a neighbouring value on its list, e.g. a 20-day lookback to 15 "
                   "or 30). A result that stays good across these neighbours sits on a plateau; one "
                   "that collapses sits on a spike that probably won't repeat.")
        n_variants = st.number_input("Nearby parameter sets to try", 10, 200, 40, 10, key="opt_rob_n")
        if st.button("Run the parameter robustness test", key="opt_rob_run"):
            with st.spinner("Scoring nearby parameter sets..."):
                n_tried, scores = run_param_robustness(best["params"], tickers, rank_metric,
                                                       int(n_variants), int(n_workers))
            st.session_state.opt_rob = {"sig": signature, "metric": rank_metric, "picked": picked_score,
                                        "n_tried": n_tried, "scores": scores}
        rob = st.session_state.get("opt_rob")
        if rob and rob["sig"] == signature:
            render_robustness_summary(rob["picked"], rob["scores"], rob["metric"], rob["n_tried"])


def render_robustness_summary(picked: float, scores, metric_label: str, n_tried: int):
    """Result display shared by the Optimizer's and the Backtest's parameter robustness tests:
    the picked configuration's score against the scores of its nudged neighbours."""
    scores = np.array([x for x in scores if x is not None and np.isfinite(x)], dtype=float)
    if len(scores) == 0:
        st.caption("No valid nearby parameter sets (no searched numeric parameters to nudge, or "
                   "every neighbour failed).")
        return
    near = float((scores >= picked - 0.25 * abs(picked)).mean())
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Selected setup's {metric_label}", f"{picked:.3f}")
    r2.metric("Nearby sets: median / worst", f"{np.median(scores):.3f} / {scores.min():.3f}")
    r3.metric("Nearby sets within 25% of it", f"{near*100:.0f}%",
              help=f"{len(scores)} of {n_tried} nearby sets produced a valid result. Share of them "
                   f"scoring no more than 25% worse than the selected setup.")
    st.caption(f"{(scores > 0).mean()*100:.0f}% of nearby sets scored above zero.")
    counts, edges = np.histogram(scores, bins=min(20, max(5, len(scores) // 3)))
    st.bar_chart(pd.Series(counts, index=[f"{edges[i]:.2f}" for i in range(len(counts))]))


def show_backtest_robustness():
    """Parameter-nudge robustness test for the backtest that was just run (Backtest tab)."""
    with st.expander("Parameter robustness test (does a small nudge wreck it?)", icon=":material/casino:"):
        cfg = st.session_state.get("bt_run_config")
        if cfg is None:
            st.caption("Run a backtest first.")
            return
        st.caption("Re-runs this backtest with its parameters nudged: each parameter ticked in the "
                   "Setup panel moves one step along its range list (a 20-day lookback to 15 or 30; "
                   "S1 and S2 nudged independently). A result that holds up across the neighbours "
                   "sits on a plateau; one that collapses sits on a spike that probably won't repeat. "
                   "Only parameters whose current value is on its range list get nudged.")
        c1, c2 = st.columns(2)
        with c1:
            n_variants = st.number_input("Nearby parameter sets to try", 10, 200, 40, 10, key="bt_rob_n")
        with c2:
            metric = st.selectbox("Score by", list(robustness.METRICS), key="bt_rob_metric")
        gen = st.session_state.bt_result_gen
        if st.button("Run the parameter robustness test", key="bt_rob_run"):
            variants, nudged = robustness.neighbor_variants(
                cfg["systems"], cfg["td"], current_search_space()[0], dual_only_optimize_set(), int(n_variants))
            if not variants:
                st.session_state.bt_rob = {"gen": gen, "nudged": nudged, "baseline": None, "scores": []}
            else:
                base = {"systems": cfg["systems"],
                        "td": {"MAX_TOTAL_EXPOSURE_PCT": cfg["td"]["MAX_TOTAL_EXPOSURE_PCT"]}}
                with st.spinner(f"Scoring {len(variants)} nearby parameter sets..."):
                    scored = robustness.evaluate(
                        [base] + variants, st.session_state.bt_portfolio_raw, cfg["ath"], cfg["td"],
                        int(st.session_state.get("opt_workers", min(4, os.cpu_count() or 1))))
                st.session_state.bt_rob = {"gen": gen, "nudged": nudged, "baseline": scored[0],
                                           "scores": scored[1:], "n_tried": len(variants)}
        rob = st.session_state.get("bt_rob")
        if rob and rob["gen"] == gen:
            if rob["baseline"] is None:
                st.caption(f"Nothing to nudge: none of the ticked parameters ({', '.join(rob['nudged']) or 'none'}) "
                           f"is currently set to a value on its range list.")
            else:
                st.caption("Nudged: " + ", ".join(rob["nudged"]))
                render_robustness_summary(rob["baseline"][metric],
                                          [r[metric] if r else None for r in rob["scores"]],
                                          metric, rob["n_tried"])


if bt_robustness_slot is not None:
    with bt_robustness_slot:
        show_backtest_robustness()


# ----------------------------------------------------------------------------
# OPTIMIZER TAB
# ----------------------------------------------------------------------------
with tab_optimize:
    st.header("Parameter optimizer")
    st.caption("Searches the Turtle Dual System's parameters. Every trial runs ONE shared-capital "
               "simulation across all selected assets and is scored on that portfolio curve, using "
               "the portfolio capital and risk cap from the Backtest tab and the trading costs from "
               "the Fees tab in the sidebar.")

    sub_opt_run, sub_opt_res, sub_opt_rob = st.tabs(
        [":material/tune: Settings & run", ":material/leaderboard: Results", ":material/casino: Robustness"])

    with sub_opt_run:
        opt_tickers = st.session_state.selected_tickers
        c1, c2 = st.columns(2)
        with c1:
            n_trials = st.number_input("Number of trials", 5, 5000, 100, 5, key="opt_ntrials")
        with c2:
            rank_metric = st.selectbox("Rank by", ["sharpe", "calmar", "cagr", "total_return"], key="opt_rank")
        default_workers = min(4, os.cpu_count() or 1)
        n_workers = st.number_input(
            "Worker processes", 1, 16, default_workers, key="opt_workers",
            help=f"Each trial is a fully independent backtest, so this parallelizes cleanly -- "
                 f"{os.cpu_count() or '?'} cores detected. If it ever hangs, set this back to 1.")

        st.markdown("**Which trial to pick as the winner**")
        pick_middle_of_top_n = st.checkbox(
            "Pick the middle of the top N instead of the single best", value=False,
            key="opt_pick_middle",
            help="Off (default): the single highest-ranked trial wins -- simple, but that trial can "
                 "just be a lucky outlier the search happened to stumble on. On: rank every trial, "
                 "take the top N, and pick the one sitting in the MIDDLE of that group instead of the "
                 "top -- a parameter set that's merely very good, surrounded on the leaderboard by "
                 "other very-good ones, is more likely to be robust than the single best one, which "
                 "may just be the top of the noise.")
        top_n_for_pick = 10
        if pick_middle_of_top_n:
            top_n_for_pick = st.number_input(
                "N (how many top trials to consider)", 3, 200, 10, 1, key="opt_top_n",
                help="The winner becomes the trial ranked in the middle of these N -- e.g. N=10 picks "
                     "the 5th-best trial, not the 1st.")

        opt_search_method = st.selectbox(
            "Search method", ["optuna", "random"], key="opt_search_method",
            help="optuna: Bayesian (TPE) search -- each round learns from the scores so far and "
                 "concentrates trials where they score well. On the same trial budget it found "
                 "clearly better parameter sets than random in testing (average of the top 10 "
                 "trials' Sharpe about 0.6 vs 0.35). random: blind random sampling. The experimental "
                 "two-stage option below overrides this.")

        use_two_stage = st.checkbox(
            "Experimental: two-stage coarse-to-fine search", value=False, key="opt_twostage")
        if use_two_stage:
            st.caption(":orange[Tested rigorously across 6 seeds before adding this: broad search, "
                       "then a refined round narrowed around the top trials, plus a hedge that keeps "
                       "part of the refined round fully blind. Result: 0 wins, 4 ties, 2 losses "
                       "against plain blind search of the same total budget (average Sharpe 0.713 vs "
                       "0.760). This search space has many weakly-correlated, largely independent "
                       "parameters, so narrowing around what the top trials happened to share is "
                       "often just noise, not a genuinely better region. Left available to "
                       "experiment with, but the evidence doesn't support it as the better default.]")

        if mh_mode_active():
            st.success("Monthly-high strategy: the search draws only the ticked numbers on the Monthly high tab (entry, pyramiding, SL, TP); everything else is off.",
                       icon=":material/trending_up:")
        st.markdown("**Which parameters should the optimizer actually search?**")
        st.info("Choose the parameters, the values to pin the others to, and each parameter's search "
                "range in the **Setup** panel in the sidebar -- the same settings the "
                "Walk-Forward tab uses.")

        if st.button("Run optimization", type="primary", icon=":material/play_arrow:", key="opt_run"):
            if not opt_tickers:
                st.error("Pick at least one asset.")
            else:
                with st.spinner("Downloading data (cached -- instant on repeat runs with the same "
                                 "assets/date range)..."):
                    raw_data = cached_download_data(tuple(sorted(opt_tickers.items())), td.START_DATE, td.END_DATE)
                    # require_all_time_high is part of the search space -- any trial might sample it
                    # True, so fetch this once here too (cached), same reasoning as
                    # optimize_turtle_dual_system.py's own main().
                    opt_ath_refs = cached_fetch_all_time_highs(tuple(sorted(opt_tickers.items())))

                dual_only = dual_only_optimize_set()
                dual_fixed = {**dual_fixed_values(), **dual_fixed_overrides("dual")}

                # Every trial's apply_params() mutates td.SYSTEMS/USE_DRAWDOWN_SCALING/etc. directly on
                # the LIVE module -- fine for workers in a separate Pool process, but a sequential
                # fallback runs every trial in THIS process, so whichever trial happens to run LAST
                # would leave ITS sampled config stuck as the live module state afterward.
                # Snapshotting here and restoring in `finally` (even on an error mid-run) is what
                # makes "hold this fixed" in the Search setup panel mean fixed at what the Backtest
                # tab has NOW, not at whatever a PRIOR run's last trial happened to leave.
                engine_snapshot = snapshot_engine_globals()
                # Sync which of S1/S2 is active from the Backtest tab, right before any trial runs
                # (restored in `finally` like every other engine global).
                opt.ENABLE_S1 = st.session_state.get("bt_en_s1", True)
                opt.ENABLE_S2 = st.session_state.get("bt_en_s2", True)
                apply_mh_mode_to_engine()                    # the monthly-high strategy: one system, only its mechanics
                # the ranges from the Search setup panel; restore_engine_globals() puts the file's
                # defaults back afterwards
                opt.set_search_space(current_search_space()[0])
                try:
                    opt.RANK_METRIC = rank_metric
                    opt.SCORE_MODE = "portfolio"
                    td.PORTFOLIO_STARTING_CAPITAL = float(st.session_state.get("acct_capital", DEFAULT_ACCOUNT))
                    td.USE_PORTFOLIO_RISK_CAP = st.session_state.get("bt_port_riskcap_on", True)
                    td.MAX_PORTFOLIO_RISK_PCT = st.session_state.get("bt_port_riskcap", td.MAX_PORTFOLIO_RISK_PCT)

                    start = time.time()
                    stage1_best_score = None
                    if use_two_stage:
                        results, n_failed, stage1_best_score = run_two_stage_search(
                            raw_data, opt_ath_refs, int(n_trials), int(n_workers), opt, rank_metric,
                            only_optimize=dual_only, fixed_values=dual_fixed)
                        ran_parallel = int(n_workers) > 1
                    elif opt_search_method == "optuna":
                        progress_bar, status = run_progress_widgets("Optimizer")

                        def _optuna_progress(done, total_trials, n_valid, n_bad):
                            progress_bar.progress(min(done / total_trials, 1.0))
                            status.write(f"{done}/{total_trials} trials -- {n_valid} valid, {n_bad} skipped -- "
                                         f"{time.time()-start:.0f}s elapsed (Optuna, {int(n_workers)} workers)")

                        results, n_failed = opt.search_optuna(
                            raw_data, opt_ath_refs, int(n_trials), int(n_workers),
                            only_optimize=dual_only, fixed_values=dual_fixed,
                            seed=opt.RANDOM_SEED, progress_cb=_optuna_progress)
                        ran_parallel = int(n_workers) > 1
                    else:
                        rng = random.Random(opt.RANDOM_SEED)
                        param_list = opt.sample_unique_param_sets(
                            rng, int(n_trials), only_optimize=dual_only, fixed_values=dual_fixed)
                        progress_bar, status = run_progress_widgets("Optimizer")
                        results, n_failed, ran_parallel = run_trials_with_progress(
                            raw_data, param_list, int(n_workers), progress_bar, status,
                            max(1, len(param_list) // 40), start, worker_module=opt, ath_references=opt_ath_refs)
                    if not ran_parallel and n_workers > 1:
                        st.caption(":material/info: Fell back to sequential execution -- see the message above for why.")
                    st.session_state.opt_stage1_best_score = stage1_best_score
                finally:
                    restore_engine_globals(engine_snapshot)

                elapsed = time.time() - start
                note_last_run("Optimizer", f"{len(results)} valid trials in {elapsed:.0f}s")
                # stored so results/apply survive any LATER, unrelated rerun (switching tabs,
                # touching any other widget anywhere in the app) -- Streamlit runs every tab's code
                # on every rerun, not just the one you're looking at, so anything living only inside
                # this button's if-block would silently vanish the moment anything else happens.
                # needs_apply=True fires the apply-and-clear-widget-keys logic exactly ONCE, on the
                # very next render, then flips itself off -- so it won't keep re-overwriting the
                # Backtest tab if you've since edited it by hand.
                st.session_state.opt_results = results
                st.session_state.opt_n_failed = n_failed
                st.session_state.opt_elapsed = elapsed
                st.session_state.pop("opt_table", None)   # a new result set: forget the old row choice
        if st.session_state.get("opt_results"):
            st.caption(f"Last run: {len(st.session_state.opt_results)} valid trials in "
                       f"{st.session_state.get('opt_elapsed', 0):.0f}s -- ranked in the Results tab.")

    if "opt_results" in st.session_state and st.session_state.opt_results:
        results = st.session_state.opt_results
        key_fn = {"sharpe": lambda r: r["sharpe"], "calmar": lambda r: r["calmar"],
                  "cagr": lambda r: r["cagr"], "total_return": lambda r: r["total_return"]}[rank_metric]
        results = sorted(results, key=key_fn, reverse=True)

        # "Pick middle of top N" (opt_pick_middle, set above): instead of always taking the
        # single #1 trial -- which can just be a lucky outlier the search happened to stumble on
        # -- take the top N by rank_metric and pick the one sitting in the MIDDLE of that group.
        # (n_eff-1)//2 is the standard "lower median index": N=10 -> index 4 (5th of 10), N=11 ->
        # index 5 (6th of 11, the exact middle), N=1 -> index 0 (the only option, same as off).
        if pick_middle_of_top_n and len(results) > 1:
            n_eff = min(top_n_for_pick, len(results))
            auto_rank0 = (n_eff - 1) // 2
        else:
            n_eff = 1
            auto_rank0 = 0

        with sub_opt_res:
            st.success(f"Done -- {len(results)} valid trials in {st.session_state.opt_elapsed:.0f}s.")
            stage1_best = st.session_state.get("opt_stage1_best_score")
            if stage1_best is not None:
                metric_key = {"sharpe": "sharpe", "calmar": "calmar", "cagr": "cagr",
                              "total_return": "total_return"}[rank_metric]
                final_best = results[auto_rank0][metric_key]
                st.caption(f"Two-stage search: stage 1 alone found best {rank_metric}={stage1_best:.3f}; "
                           f"after stage 2, best {rank_metric}={final_best:.3f} "
                           f"({'+' if final_best >= stage1_best else ''}{final_best-stage1_best:.3f})")
            if st.session_state.get("opt_sent_note"):
                st.success(st.session_state.pop("opt_sent_note"))

            # The table is where the trial is CHOSEN: click a row to use that trial for everything
            # below (its settings, the robustness checks) and for sending to the Backtest tab. With no
            # row clicked, the automatic pick is used. Shows enough rows that the automatic pick
            # (possibly ranked below 25 if N is large) is visible.
            n_rows = min(max(25, n_eff), 100)
            rows = []
            for rank, r in enumerate(results[:n_rows], start=1):
                rows.append({"rank": rank, "auto pick": "<-- auto" if rank == auto_rank0 + 1 else "",
                             "sharpe": round(r["sharpe"], 2), "calmar": round(r["calmar"], 2),
                             "CAGR": f"{r['cagr']*100:.2f}%", "MaxDD": f"{r['max_dd']*100:.2f}%",
                             "trades": r["total_trades"]})
            st.markdown("**Choose a trial** -- click a row. With no row chosen, the automatic pick is used.")
            table_event = st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                                       on_select="rerun", selection_mode="single-row", key="opt_table")
            clicked = table_event.selection.rows if table_event is not None else []
            chosen_rank0 = clicked[0] if clicked and clicked[0] < len(results) else auto_rank0
            best = results[chosen_rank0]
            top_trial = results[0]

            if clicked:
                st.subheader(f"Chosen trial: rank {chosen_rank0+1} of {len(results)} (picked from the table)")
            elif pick_middle_of_top_n and auto_rank0 > 0:
                st.subheader(f"Automatic pick: middle of the top {n_eff} (rank {auto_rank0+1} of {len(results)})")
                st.caption(f"Top trial alone: {rank_metric}={key_fn(top_trial):.3f} -- "
                           f"the automatic pick: {rank_metric}={key_fn(best):.3f}.")
            else:
                st.subheader("Automatic pick: the best trial")
            st.json({k: v for k, v in best["params"].items()})
            show_trial_market_risk(best["params"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Sharpe", f"{best['sharpe']:.2f}", border=True)
            m2.metric("CAGR", f"{best['cagr']*100:.2f}%", border=True)
            m3.metric("Portfolio MaxDD", f"{best['max_dd']*100:.2f}%", border=True)

            if st.button("Send this trial's settings to the Backtest tab", type="primary", icon=":material/send:", key="opt_send"):
                send_trial_to_backtest(best["params"], chosen_rank0 + 1)
            if mh_mode_active():
                st.caption("Loads its numbers into the Monthly high tab of the Setup panel (the Backtest tab runs them). "
                           "Nothing is saved to disk.")
            else:
                st.caption("Loads its S1/S2 parameters (and the exposure cap, and the per-market risk multipliers if "
                           "they were searched) into the Backtest tab. Nothing is saved to disk until you click "
                           "Save on that tab.")

        with sub_opt_rob:
            show_param_sensitivity_section(results, rank_metric, rank_metric, key_prefix="opt_single")
            show_trial_robustness_checks(best, opt_tickers, rank_metric, key_fn(best), int(n_workers),
                                         all_sharpes=[r["sharpe"] for r in results])
    else:
        with sub_opt_res:
            if "opt_results" in st.session_state:
                st.warning("No valid trials -- every combination either had no trades or errored.")
            else:
                st.caption("Run an optimization on the first tab to see the ranked trials here.")
        with sub_opt_rob:
            st.caption("Run an optimization and pick a trial in the Results tab first.")


def build_param_stability_table(window_reports: list) -> pd.DataFrame:
    """For every parameter searched across a walk-forward run's windows, summarizes how much its
    WINNING value actually varied window to window -- a narrow numeric range (or a dominant
    boolean/categorical value) means the search converged on something stable; full-range scatter
    means there isn't enough signal in that window's data (or too many free dimensions relative
    to the trial budget) for this parameter to converge on anything at all. Directly answers "is
    it even worth searching this parameter" -- an unconverged one is basically free-floating
    noise, and narrowing the opt-in panel to just the converged ones is a concrete next step."""
    ok = [r for r in window_reports if r.get("status") == "ok" and "params" in r]
    if len(ok) < 2:
        return pd.DataFrame()
    all_names = set()
    for r in ok:
        all_names.update(r["params"].keys())
    rows = []
    for pname in sorted(all_names):
        values = [r["params"][pname] for r in ok if pname in r["params"]]
        if not values:
            continue
        if all(isinstance(v, bool) for v in values):
            true_frac = sum(1 for v in values if v) / len(values)
            rows.append({"parameter": pname, "type": "bool/on-off", "n_windows": len(values),
                          "summary": f"{true_frac*100:.0f}% of windows True",
                          "converged": true_frac <= 0.15 or true_frac >= 0.85})
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            arr = np.array(values, dtype=float)
            spread = arr.std() / (abs(arr.mean()) + 1e-9)
            rows.append({"parameter": pname, "type": "numeric", "n_windows": len(values),
                          "summary": f"{arr.min():.3g} to {arr.max():.3g}  (mean {arr.mean():.3g}, "
                                      f"std {arr.std():.3g})",
                          "converged": spread < 0.3})
        else:
            # str() first: value_counts() silently drops None, so a parameter pinned to None in
            # every window (the flat risk schedule) used to produce an empty result and crash
            counts = pd.Series([str(v) for v in values]).value_counts()
            top_val, top_count = counts.index[0], counts.iloc[0]
            top_frac = top_count / len(values)
            rows.append({"parameter": pname, "type": "categorical", "n_windows": len(values),
                          "summary": f"'{top_val}' in {top_frac*100:.0f}% of windows",
                          "converged": top_frac >= 0.6})
    df = pd.DataFrame(rows)
    return df.sort_values("converged") if not df.empty else df


def show_param_stability_section(window_reports: list, engine_key: str = "dual"):
    """Renders the walk-forward parameter stability panel as an expander -- the same manual
    analysis (read the per-window params, check std/range per column) this project has used
    directly on walk_forward_windows.csv in the past, now built into the UI instead."""
    with st.expander("Parameter stability across windows (did the search converge on anything?)", icon=":material/explore:"):
        stability = build_param_stability_table(window_reports)
        if stability.empty:
            st.caption("Need at least 2 successful windows to compare parameter stability.")
            return
        n_unconverged = int((~stability["converged"]).sum())
        st.caption(f"Each searched parameter's WINNING value across every successful window. "
                   f"{':orange[:material/warning:] ' + str(n_unconverged) + ' parameter(s) never converged on a stable value' if n_unconverged else ':green[:material/check_circle:] All parameters converged reasonably'} "
                   f"-- an unconverged one is essentially floating noise given this trial budget; "
                   f"consider holding it fixed (uncheck it in the opt-in panel above) instead of "
                   f"re-searching it every window.")
        if n_unconverged:
            unconverged_params = stability.loc[~stability["converged"], "parameter"].tolist()
            if st.button(f"Uncheck the {n_unconverged} non-converged parameter(s) above",
                         key="wf_uncheck_unconverged",
                         help="Unchecks these parameters in the opt-in panel above so the next "
                              "walk-forward run holds them fixed at the Backtest tab's current "
                              "values instead of re-searching noise."):
                st.session_state.wf_pending_uncheck_params = {
                    "engine_key": engine_key, "params": unconverged_params}
                st.rerun()
        display = stability.copy()
        display["converged"] = display["converged"].map(lambda c: "yes" if c else "no")
        st.dataframe(
            display.rename(columns={"parameter": "Parameter", "type": "Type", "n_windows": "# Windows",
                                      "summary": "Winning values across windows", "converged": "Converged?"}),
            width="stretch", hide_index=True)


def show_wf_monte_carlo(window_reports: list, oos_curve):
    """Monte Carlo checks on a walk-forward's out-of-sample results: a bootstrap over the windows,
    and a block test on the stitched out-of-sample portfolio equity curve."""
    ok = [r for r in window_reports if r.get("status") == "ok"]
    with st.expander("Monte Carlo on the out-of-sample results", icon=":material/casino:"):
        if len(ok) < 3:
            st.caption("Need at least 3 windows with an out-of-sample result.")
            return
        st.markdown("**Across windows -- is the average out-of-sample result real?**")
        st.caption("Resamples the windows' own out-of-sample results 5,000 times, in runs of 3 "
                   "neighbouring windows (good and bad market stretches span several windows, so "
                   "single windows are not independent), and looks at the average each time. If a "
                   "good share of resamples average zero or less, the positive average could easily "
                   "be luck of which windows happened to be in the run. With only about 26 windows "
                   "the interval is still rough.")
        cols = st.columns(2)
        for col, (label, key, as_pct) in zip(cols, (("Sharpe", "oos_sharpe", False), ("CAGR", "oos_cagr", True))):
            values = [r[key] for r in ok]
            if as_pct:
                # windows are single years: the figure that matters is what they COMPOUND to, which is the
                # geometric mean (average the log growth, then convert back), not the plain average
                values = [float(np.log1p(max(v, -0.999999))) for v in values]
            bs = monte_carlo.bootstrap_mean(values, block=3)
            if bs is None:
                continue
            if as_pct:
                bs = dict(bs, mean=float(np.expm1(bs["mean"])), p5=float(np.expm1(bs["p5"])),
                          p95=float(np.expm1(bs["p95"])))
                label = "CAGR (compounded)"
            fmt = (lambda v: f"{v*100:.2f}%") if as_pct else (lambda v: f"{v:.2f}")
            with col:
                st.metric(f"Mean out-of-sample {label}", fmt(bs["mean"]),
                          help=f"Over {bs['n']} windows. 90% interval from resampling: "
                               f"{fmt(bs['p5'])} to {fmt(bs['p95'])}.")
                st.metric(f"Chance the true mean {label} is zero or less", f"{bs['p_le_zero']*100:.1f}%")
                st.caption(f"90% interval: {fmt(bs['p5'])} to {fmt(bs['p95'])}")

        st.markdown("**Across days -- the stitched out-of-sample portfolio curve**")
        if oos_curve is None or len(oos_curve) < 60:
            st.caption("No stitched out-of-sample curve was kept for this run.")
            return
        render_monte_carlo("wf_oos", equity_curve=oos_curve, years=_curve_years(oos_curve), series_mode=True)


# ----------------------------------------------------------------------------
# WALK-FORWARD TAB
# ----------------------------------------------------------------------------
with tab_walkforward:
    st.header("Walk-forward validation")
    st.caption("Reuses walk_forward_turtle.py's own logic as-is; output below is its normal "
               "printed report, captured and shown here rather than refactored. Rolling windows: "
               "each window's parameters are searched using ONLY the train period, then evaluated "
               "on the test period immediately after (data the search never saw).")
    wf_tickers = st.session_state.selected_tickers

    sub_wf_run, sub_wf_report, sub_wf_stab, sub_wf_pos = st.tabs(
        [":material/tune: Settings & run", ":material/article: Report", ":material/query_stats: Stability & Monte Carlo",
         ":material/candlestick_chart: Positions"])

    with sub_wf_run:
        if mh_mode_active():
            st.success("Monthly-high strategy: every window searches only the ticked numbers on the Monthly high tab (entry, pyramiding, SL, TP); S1 / S2 do not apply.",
                       icon=":material/trending_up:")
        wfs1, wfs2 = st.columns(2)
        with wfs1:
            wf_enable_s1 = st.checkbox("Test S1", value=True, key="wf_en_s1",
                                         help="Its OWN toggle, independent of the Backtest tab's Enable "
                                              "S1/S2 -- lets you walk-forward S1 alone, S2 alone, or both "
                                              "together without changing what you're currently backtesting.")
        with wfs2:
            wf_enable_s2 = st.checkbox("Test S2", value=True, key="wf_en_s2")

        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        with wc1:
            wf_train_years = st.number_input("Train years", 1, 20, 4, 1, key="wf_train_years")
        with wc2:
            wf_test_years = st.number_input("Test years", 1, 10, 1, 1, key="wf_test_years")
        with wc3:
            wf_anchored = st.checkbox("Expanding train window", value=False, key="wf_anchored",
                                        help="Off = rolling fixed-length train window (default). On = "
                                             "train window always starts at the earliest available date.")
        with wc4:
            n_trials_per_window = st.number_input("Trials per window", 5, 2000, 150, 5, key="wf_ntrials")
        with wc5:
            wf_rank_metric = st.selectbox("Rank by", ["sharpe", "calmar", "cagr", "total_return"],
                                            key="wf_rank_metric",
                                            help="Which metric picks the winning param set out of each "
                                                 "window's in-sample search (walk_forward_turtle.py's own "
                                                 "RANK_METRIC -- previously always Sharpe, hardcoded, not "
                                                 "exposed here).")

        st.caption("Each window runs ONE shared-capital simulation across every selected asset and "
                   "scores that single portfolio curve, in-sample and out-of-sample, using the "
                   "portfolio capital and risk cap from the Backtest tab and the trading costs from "
                   "the Fees tab in the sidebar.")

        wfm1, wfm2 = st.columns(2)
        with wfm1:
            wf_search_method = st.selectbox(
                "Search method", ["optuna", "random"], key="wf_search_method",
                help="optuna: Bayesian (TPE) search that concentrates trials where scores are good, and "
                     "stops clearly-bad trials early. random: the original blind random search. Either "
                     "way, windows run in parallel across CPU cores.")
        with wfm2:
            wf_warm_start = st.checkbox(
                "Fast simulation start", value=True, key="wf_warm_start",
                help="On: each trial simulates from 3 years before the window's train start (several "
                     "times faster, no look-ahead). Off: simulates every asset from its first bar, as "
                     "the original did; only path-dependent carried-in state (open positions, "
                     "cooldowns, drawdown scaling) can differ between the two.")

        wf_pick_middle = st.checkbox(
            "Pick the middle of the top N instead of the single best", value=True, key="wf_pick_middle",
            help="Off: each window uses its single highest-scoring in-sample trial -- which can just be "
                 "a lucky outlier. On: each window ranks its trials, takes the top N, and uses the one "
                 "in the MIDDLE of that group (N=10 picks the 5th best), a parameter set that's merely "
                 "very good alongside other very-good ones and so more likely to hold up out-of-sample.")
        wf_top_n = 10
        if wf_pick_middle:
            wf_top_n = st.number_input(
                "N (how many top trials to consider)", 3, 200, 10, 1, key="wf_top_n",
                help="Capped at the trials-per-window count. The winner becomes the trial ranked in the "
                     "middle of these N.")

        st.markdown("**Which parameters should walk-forward actually search?**")
        st.info("Set in the **Setup** panel in the sidebar -- the same parameter selection, "
                "pinned values and search ranges the Optimizer tab uses. With ~15 free dimensions and "
                "only a few hundred trials per window, an unconstrained search mostly can't converge "
                "on anything stable: narrow it down to the few parameters you actually want tested.")

        if st.button("Run walk-forward validation", type="primary", icon=":material/play_arrow:", key="wf_run"):
            _wf_lock = ui_locks.get_lock("walk_forward")
            if not wf_tickers:
                st.error("Pick at least one asset.")
            elif not wf_enable_s1 and not wf_enable_s2:
                st.error("Pick at least one of S1 / S2 to test.")
            elif _wf_lock.locked():
                # Checked BEFORE touching td.TICKERS/snapshotting anything below -- an earlier click's
                # background thread may still be running and reading those same module globals; briefly
                # mutating-then-restoring them here while that's happening would race with it. See
                # run_captured_live()'s own comment for why an orphaned thread like this can exist at
                # all (Streamlit can abort a script run, but not the background thread it started).
                st.warning("A previous walk-forward run is still finishing in the background (likely "
                           "an earlier click). Wait for it to complete before starting another.")
            else:
                # wf.main() reads td.TICKERS internally rather than taking it as a parameter, so it
                # has to be set temporarily -- but ALWAYS restored afterward (even on error), so the
                # full available ticker list never permanently shrinks for other tabs
                original_tickers = dict(td.TICKERS)
                # Same snapshot/restore reasoning as the Optimizer tab's "Run optimization" handler --
                # opt.apply_params() (called by every window's trials here too) mutates td.SYSTEMS/
                # USE_DRAWDOWN_SCALING/etc. directly on the live module, and any sequential fallback
                # (n_workers=1, or a failed Pool) runs trials in THIS process, so without restoring,
                # whichever trial ran last would leave its own sampled config stuck as the live module
                # state -- the exact "opt not respecting what I turned off" bug, just never fixed here.
                engine_snapshot = snapshot_engine_globals()

                def _restore_wf_globals():
                    td.TICKERS = original_tickers
                    restore_engine_globals(engine_snapshot)

                # Restoring is routed through run_captured_live()'s on_complete (fires from INSIDE its
                # background thread) rather than a plain try/finally here -- if a later click aborts
                # THIS script execution while wf.main() is still running in the background, a
                # try/finally here would restore these globals out from under that still-running
                # thread for every window it hasn't reached yet. The except/captured-is-None branches
                # below are just a safety net for the rare case no thread ever actually started (e.g.
                # an exception before run_captured_live() is even reached).
                try:
                    td.TICKERS = wf_tickers
                    wf.N_TRIALS_PER_WINDOW = int(n_trials_per_window)
                    wf.TRAIN_YEARS = int(wf_train_years)
                    wf.TEST_YEARS = int(wf_test_years)
                    wf.ANCHORED_TRAIN = bool(wf_anchored)
                    wf.RANK_METRIC = wf_rank_metric
                    wf.PICK_MIDDLE_OF_TOP_N = int(wf_top_n) if wf_pick_middle else 1
                    # trading costs come from the universal Fees tab, already applied to the engine
                    # (and shipped to the worker processes inside the portfolio config)
                    wf.COST_OVERRIDES = None
                    wf.USE_IBKR_FEES = False
                    wf.SEARCH_METHOD = wf_search_method
                    wf.SIM_WARMUP_YEARS = 3 if wf_warm_start else None
                    wf.SCORE_MODE = "portfolio"
                    # Own S1/S2 toggle, independent of the Backtest tab's -- opt.ENABLE_S1/ENABLE_S2
                    # are what opt.is_valid_params()/opt.params_to_systems() (reused here for every
                    # trial) actually check, and snapshot_engine_globals() above already covers
                    # restoring them afterward (same _OPT_GLOBAL_NAMES fix as the Optimizer tab).
                    opt.ENABLE_S1 = wf_enable_s1
                    opt.ENABLE_S2 = wf_enable_s2
                    apply_mh_mode_to_engine()                # the monthly-high strategy: one system, only its mechanics
                    wf.ONLY_OPTIMIZE = dual_only_optimize_set("dual")
                    wf.FIXED_VALUES = {**dual_fixed_values(enable_s1=wf_enable_s1, enable_s2=wf_enable_s2),
                                       **dual_fixed_overrides("dual")}
                    # the ranges from the Search setup panel; wf.main() copies them to its worker
                    # processes, and _restore_wf_globals() (via restore_engine_globals) resets them
                    opt.set_search_space(current_search_space()[0])
                    td.PORTFOLIO_STARTING_CAPITAL = float(st.session_state.get("acct_capital", DEFAULT_ACCOUNT))
                    td.USE_PORTFOLIO_RISK_CAP = st.session_state.get("bt_port_riskcap_on", True)
                    td.MAX_PORTFOLIO_RISK_PCT = st.session_state.get(
                        "bt_port_riskcap", td.MAX_PORTFOLIO_RISK_PCT)
                    st.markdown("**Progress** (also in the Run status box in the left sidebar)")
                    wf_progress, wf_status = run_progress_widgets("Walk-forward")
                    wf_log = st.empty()
                    wf_started = time.time()
                    wf_result, captured = run_captured_live(wf.main, wf_log, wf_progress, run_key="walk_forward",
                                                              on_complete=_restore_wf_globals,
                                                              status_placeholder=wf_status)
                    if captured is not None:
                        note_last_run("Walk-forward", f"{len((wf_result or {}).get('window_reports', []))} windows "
                                                      f"in {time.time() - wf_started:.0f}s, {cost_model_label()} costs")
                    if captured is None:
                        # Lock was somehow already held at the moment run_captured_live() actually
                        # acquired it (the earlier `elif _wf_lock.locked()` check above and that
                        # acquire aren't atomic together) -- no thread started, so on_complete never
                        # ran; restore here instead.
                        _restore_wf_globals()
                except Exception:
                    _restore_wf_globals()
                    raise
                if captured is not None:
                    st.success("Done. The printed report is in the Report tab; the window stability, Monte Carlo checks and position chart are in the tabs after it.")
                    st.session_state.wf_report = captured
                    # Stashed in session_state (not a local var) so the position-chart section below
                    # still has something to render on a LATER, unrelated rerun (switching tabs, etc.)
                    # -- same reasoning as bt_results/bt_raw for the Backtest/Trade Chart tabs.
                    chart_data = (wf_result or {}).get("window_oos_chart_data")
                    st.session_state.wf_chart_data = chart_data or None
                    st.session_state.wf_chart_tickers = dict(wf_tickers) if chart_data else None
                    st.session_state.wf_window_reports = (wf_result or {}).get("window_reports")
                    st.session_state.wf_oos_curve = ((wf_result or {}).get("stitched_oos_curves") or {}).get("PORTFOLIO")

    with sub_wf_report:
        wf_report_text = st.session_state.get("wf_report")
        if wf_report_text:
            st.code(wf_report_text, language=None)
        else:
            st.caption("Run a walk-forward on the first tab to see its printed report here.")

    wf_window_reports = st.session_state.get("wf_window_reports")
    with sub_wf_stab:
        if wf_window_reports:
            show_param_stability_section(wf_window_reports)
            show_wf_monte_carlo(wf_window_reports, st.session_state.get("wf_oos_curve"))
        else:
            st.caption("Run a walk-forward on the first tab first.")

    # ---- Out-of-sample position chart, per tested asset ----
    # Read from session_state (not a local var from the button handler above) so this section
    # still renders on a LATER, unrelated rerun -- switching tabs and back, touching any other
    # widget -- same reasoning as bt_results/bt_raw for the Backtest/Trade Chart tabs.
    with sub_wf_pos:
        wf_chart_data = st.session_state.get("wf_chart_data")
        wf_chart_tickers = st.session_state.get("wf_chart_tickers")
        if wf_chart_data:
            st.subheader("Out-of-sample positions")
            st.caption("Every window's own out-of-sample trades, stitched together in chronological "
                       "order onto that asset's full price history -- each window used a DIFFERENT "
                       "winning parameter set (see the per-window params in the report above), so a "
                       "visible change in trading style partway through is expected, not a bug.")
            if not LIGHTWEIGHT_CHARTS_AVAILABLE:
                st.error("streamlit-lightweight-charts isn't installed. Run: "
                         "pip install streamlit-lightweight-charts --break-system-packages")
            else:
                wf_chart_assets = sorted({name for w in wf_chart_data for name in w["assets"]})
                if not wf_chart_assets:
                    st.info("No out-of-sample trades were captured for any asset in this run.")
                else:
                    wf_chart_asset = st.selectbox("Asset", wf_chart_assets, key="wf_chart_asset_pick")
                    ticker_symbol = (wf_chart_tickers or {}).get(wf_chart_asset)
                    full_prices = (cached_download_data(((wf_chart_asset, ticker_symbol),),
                                                           td.START_DATE, td.END_DATE)
                                    if ticker_symbol else {})
                    full_df = full_prices.get(wf_chart_asset)
                    if full_df is None or full_df.empty:
                        st.warning("Couldn't re-download price history for this asset to chart against.")
                    else:
                        # Stitch every window's already-filtered (own-test-period-only) trades/exit
                        # events together in window order -- windows are non-overlapping in time
                        # (build_windows), so this produces one clean chronological trade history, not
                        # duplicated or out-of-order activity.
                        stitched_trades = {}
                        stitched_exit_dates = {}
                        for w in wf_chart_data:
                            asset_data = w["assets"].get(wf_chart_asset)
                            if not asset_data:
                                continue
                            for sysname, trades in asset_data["trades_by_system"].items():
                                stitched_trades.setdefault(sysname, []).extend(trades)
                            for sysname, markers in asset_data["exit_markers_by_system"].items():
                                stitched_exit_dates.setdefault(sysname, []).extend(markers)

                        # Exit markers were captured as (date, price, reason) -- valid against ANY
                        # dataframe sharing this asset's date index, unlike the raw iloc each window's
                        # own internal (shorter) features dataframe originally used -- resolved back to
                        # an iloc here against full_df, which is what build_lightweight_chart_data
                        # (shared with the Trade Chart tab) actually expects.
                        stitched_exit_markers = {}
                        for sysname, markers in stitched_exit_dates.items():
                            kept = []
                            for date, price, reason in markers:
                                try:
                                    iloc = full_df.index.get_loc(date)
                                except KeyError:
                                    continue
                                kept.append((iloc, price, reason))
                            stitched_exit_markers[sysname] = kept

                        wf_combined_result = {"trades_by_system": stitched_trades,
                                                "exit_markers_by_system": stitched_exit_markers}
                        wf_candles, wf_markers, wf_position_lines = build_lightweight_chart_data(
                            full_df, wf_combined_result)
                        wf_markers, wf_position_lines = filter_chart_labels(
                            wf_markers, wf_position_lines, chart_label_controls("wf"))

                        wf_show_position_lines = st.checkbox(
                            "Show entry/SL/TP position boxes", value=True,
                            key="wf_chart_show_position_lines")

                        wf_chart_options = {
                            "height": 650,
                            "layout": chart_layout(),
                            "timeScale": {"borderColor": "rgba(197, 203, 206, 0.8)", "timeVisible": False},
                            "rightPriceScale": {"borderColor": "rgba(197, 203, 206, 0.8)"},
                            "grid": {"vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
                                      "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}},
                        }
                        wf_candlestick_series = [{
                            "type": "Candlestick",
                            "data": wf_candles,
                            "options": {
                                "upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False,
                                "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
                            },
                            "markers": wf_markers,
                        }]
                        wf_all_series = wf_candlestick_series + (
                            wf_position_lines if wf_show_position_lines else [])
                        renderLightweightCharts([{"chart": wf_chart_options, "series": wf_all_series}],
                                                  key=f"wf_chart_{wf_chart_asset}")
        else:
            st.caption("Run a walk-forward on the first tab to chart its out-of-sample trades here.")

# ----------------------------------------------------------------------------
# REGIME DIAGNOSTIC TAB
# ----------------------------------------------------------------------------
with tab_regime:
    st.header("Regime diagnostic")
    st.caption("Reuses regime_diagnostic.py's own logic as-is; output below is its normal "
               "printed report, captured and shown here rather than refactored.")
    rd_tickers = st.session_state.selected_tickers

    if st.button("Run regime diagnostic", type="primary", icon=":material/play_arrow:", key="rd_run"):
        if not rd_tickers:
            st.error("Pick at least one asset.")
        else:
            # same reasoning as the walk-forward tab -- rd.main() reads td.TICKERS internally,
            # set temporarily and always restored afterward, even on error
            original_tickers = dict(td.TICKERS)
            try:
                td.TICKERS = rd_tickers
                with st.spinner("Running regime diagnostic..."):
                    _, captured = run_captured(rd.main)
            finally:
                td.TICKERS = original_tickers
            st.success("Done.")
            st.code(captured, language=None)