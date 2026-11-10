"""
Dual-System Turtle Breakout Strategy: SP500, Gold, Silver, Nasdaq (Daily)
============================================================================
Two independent breakout systems run simultaneously on each asset, in the
spirit of the classic Turtle Trading rules.
"""

import os
import calendar
import random
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ============================================================================
# ASSETS
# ============================================================================
TICKERS = {
    "SP500": "^GSPC",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Nasdaq": "^NDX",
    "Bitcoin": "BTC-USD",
    "Ger40": "^GDAXI",
    "Platinum": "PL=F",
    "Palladium": "PA=F",    # one of the most extreme sustained commodity trends in recent history                     # (2016-2022), but thinner liquidity/history than the others -- verify carefully
    "Copper": "HG=F",
    "Nikkei225": "^N225",
    "Russell2000": "^RUT",
    "Dow": "^DJI",
    "CrudeOil": "CL=F",
    "Corn": "ZC=F",
    "Cocoa": "CC=F",# industrial-demand-driven, a genuinely different character than gold/silver
}
START_DATE = (pd.Timestamp.now() - pd.DateOffset(years=30)).strftime("%Y-%m-%d")   # last 30 years
END_DATE = None   # None = up to today

STARTING_CAPITAL_PER_ASSET = 10000.0

DEFAULT_SYSTEM_PARAMS = {
    "entry_buffer_n_mult": 0.25,
    "n_atr_period": 30,
    "risk_per_trade_pct": 3.0,
    "risk_per_trade_pct_schedule": None,
    "pyramid_spacing_n_mult": 0.25,
    "pyramid_method": "spacing",   # "spacing" (current default), "swing_high", "monthly_high" (adds on every NEW monthly-high breakout, see mh_* below) or "breakout" (new: adds on a
                                     # fresh N-day-high, using pyramid_breakout_lookback below)
    "pyramid_breakout_lookback": 5,  # only used when pyramid_method == "breakout" -- serves TWO
                                       # roles: (1) how many days define a "fresh high" for an
                                       # add, and (2) the MINIMUM days that must pass since the
                                       # last add, regardless of price. (2) is what actually
                                       # prevents rapid-fire adds during one uninterrupted thrust
                                       # -- a longer lookback alone can't, since every day of a
                                       # sustained climb is trivially a fresh N-day high no matter
                                       # how large N is (there's no older peak left to compare
                                       # against once the rally itself dominates the window)
    "swing_lookback": 15,   # only used when pyramid_method == "swing_high" -- how many bars on
                             # EACH SIDE define a confirmed swing pivot. Bigger = requires a
                             # bigger, more genuine pullback before the next add can even become
                             # possible -- this is what actually produces months-long-trend-style
                             # pyramiding (sparse, structurally spaced adds), not a longer
                             # lookback on a rolling-high check (that approach cannot do this --
                             # proven directly: it still fires every day of an uninterrupted move)
    "max_units_per_trend": 3,
    "stop_n_mult": 1.5,
    "use_fixed_stop": True,   # the core ratcheting stop (exit_reason="SL") -- previously had no
                               # toggle at all. Turning this OFF removes your only exit that
                               # ratchets tightly to price-vs-N in real time; you would then be
                               # relying entirely on whichever of trailing_stop/chandelier/exit_
                               # channel are still enabled for downside protection.
    "use_average_entry_stop": False,
    "ratchet_stop_on_pyramid": True,   # if False, pyramid adds do NOT tighten the stop at all --
                                        # it stays wherever the FIRST entry set it, for the life
                                        # of the position (use_breakeven_stop is unaffected, still
                                        # its own separate, independently controlled ratchet)
    "use_time_stop": False, "time_stop_days": 10, "time_stop_min_r": 0.25,
    "use_scale_out": True, "scale_out_levels": [(2.0, 1 / 3), (4.0, 1 / 3)],
    "tp_atr_period": 20, "tp_n_mult": 3.0,
    "use_n_day_low_trailing_stop": True, "trailing_stop_lookback_days": 20,
    "use_breakeven_stop": True, "breakeven_trigger_r": 1.5,
    "use_chandelier_exit": True, "chandelier_atr_mult": 5.0,
    # NEW: per-UNIT trailing profit give-back -- every other stop here is shared across the WHOLE
    # episode (one level, applied identically to every pyramided unit), which is why a late add
    # bought near the top gets force-closed at the same price as an early add sitting on a big
    # cushion, never getting a chance to lock in ITS OWN gains first. This one is unit-aware: once
    # a unit's own profit reaches giveback_min_profit_r, its stop trails at
    # entry + giveback_fraction * (that unit's OWN peak price since ITS OWN entry - entry) -- an
    # early, deeply profitable unit can give back a lot of room and still hold; a late, barely-
    # profitable unit gets protected almost immediately. Off by default (opt-in, changes nothing
    # for anyone who doesn't turn it on).
    "use_giveback_stop": False, "giveback_fraction": 0.4, "giveback_min_profit_r": 1.0,
    "adx_period": 30,
    "use_regime_scale_out": False, "regime_scale_out_adx_threshold": 35.0, "regime_scale_out_fraction": 0.5,
    # OPTIONAL: blocks a pyramid ADD (not the initial entry) if ADX is already above this at
    # that moment -- directly built from the regime diagnostic finding that high-ADX-confirmed
    # moves underperform going forward. Reuses the same ADX series as regime_scale_out.
    "use_pyramid_adx_filter": False, "pyramid_max_adx": 35.0,
    "use_ema_filter": False, "ema_period": 200,   # require close > EMA(period) to enter
    # OPTIONAL: a moving-average REGIME instead of the breakout/exit channels. The system enters when the
    # close is above its ma_regime_period-day simple average (plus ma_regime_band_pct %) and leaves when it
    # closes below it (minus the band); the breakout and exit lookbacks and the entry buffer are ignored.
    # Built for a long-only account in a drifting market, where a channel exit sells every pullback and buys
    # back higher. Everything else (sizing, caps, stops, next-open fills, costs) works as usual.
    "use_ma_regime": False, "ma_regime_period": 200, "ma_regime_band_pct": 0.0,
    # OPTIONAL: enter on "the close breaks above an exponential moving average" instead of the N-day-high breakout (the entry buffer is not used).
    # ema_entry_mode "cross" = only on the day the close moves from at-or-below the EMA to above it (after an exit the system waits for the next cross);
    # "above" = whenever the system is flat and the close is above the EMA. Stops, sizing, pyramiding and caps are unchanged.
    # OPTIONAL, independent of the entry: leave when the close falls below the ema_exit_period-day EMA, INSTEAD of the exit_lookback-day-low channel
    # (the exit is still reported as "exit_channel"). It works with either entry (breakout or EMA). An EMA entry / exit overrides the moving-average
    # regime's levels when both are on. Research so far: no robust edge (see research/risk/README.md).
    "use_ema_entry": False, "ema_entry_period": 50, "ema_entry_mode": "cross",
    "use_ema_exit": False, "ema_exit_period": 20,
    # The N-day-low channel exit (exit_lookback, reported as "exit_channel"). ON by default = unchanged. OFF = the system has no channel exit at all (the clean monthly-high mode below).
    "use_channel_exit": True,
    # OPTIONAL entry: MONTHLY-HIGH BREAKOUT. Every completed calendar month has a monthly high (its highest HIGH). Each monthly high breaks out ONCE: the first bar whose close is above
    # (that high + the entry buffer) after the month ended is its breakout, and the high is used up for good. The breakout is an ENTRY signal only if the bar that set the high is then
    # between mh_min_age and mh_max_age bars old (at least a month, at most about a year) and the close is above its mh_sma_period-day EMA (mh_ma_type "ema", the default) or simple
    # average ("sma"); 0 = no trend filter. A high that is broken while too young (or while the filter fails) is used up without a trade. Days are daily bars.
    # While this entry is on, the position ADDS a unit on EVERY such signal (every time a new entry would have happened, no spacing rule); pyramid_method "monthly_high" gives the same
    # adds to any other entry type. The buffer is part of the level here, and this entry overrides the EMA entry and the moving-average regime when several are on.
    # See monthly_high_months / monthly_high_events / monthly_high_entry_level.
    "use_monthly_high_entry": False, "mh_min_age": 20, "mh_max_age": 365, "mh_sma_period": 200, "mh_ma_type": "ema",
    # OPTIONAL: a resting peak-trail stop. After every close the stop for the NEXT bar is set at (highest high since entry) - peak_trail_atr_mult x
    # (current N), and it only ever rises; the next bar's low touching it fills there (or at the open if the market gapped below it). Unlike the
    # Chandelier exit below, which measures the same bar's own high and range, this is an order that could really have been resting in the market.
    "use_peak_trail": False, "peak_trail_atr_mult": 8.0,
    # OPTIONAL: shared, universal entry filter -- requires the market to genuinely be trending
    # (Choppiness Index below this threshold) before allowing an entry. Built directly from the
    # regime diagnostic finding that every asset performed best in its own "Trending" bucket by
    # this exact measure. LOW Choppiness Index = trending, HIGH = choppy/directionless.
    "use_choppiness_filter": False, "choppiness_period": 14, "choppiness_max": 38.2,
    "require_all_time_high": False,   # optional: entry must ALSO be a genuine all-time high
                                        # (not just an N-day one) -- no overhead resistance from
                                        # anyone above water waiting to sell
    # NOTE: not in the config you pasted -- added back with the same defaults as before, since
    # the code requires this key to exist (no fallback) and it directly addresses the "S2
    # re-entering right after a loss" issue from a couple messages ago. Remove/edit freely.
    "use_cooldown": True, "cooldown_days": 10,
}

s2_SYSTEM_PARAMS = {
    "entry_buffer_n_mult": 0.25,
    "n_atr_period": 30,
    "risk_per_trade_pct": 3.0,
    "risk_per_trade_pct_schedule": None,
    "pyramid_spacing_n_mult": 0.25,
    "pyramid_method": "spacing",
    "pyramid_breakout_lookback": 5,
    "swing_lookback": 15,
    "max_units_per_trend": 3,
    "stop_n_mult": 1.5,
    "use_fixed_stop": True,
    "use_average_entry_stop": False,
    "ratchet_stop_on_pyramid": True,
    "use_time_stop": False, "time_stop_days": 10, "time_stop_min_r": 0.25,
    "use_scale_out": True, "scale_out_levels": [(2.0, 1 / 3), (4.0, 1 / 3)],
    "tp_atr_period": 20, "tp_n_mult": 3.0,
    "use_n_day_low_trailing_stop": True, "trailing_stop_lookback_days": 20,
    "use_breakeven_stop": True, "breakeven_trigger_r": 1.5,
    "use_chandelier_exit": True, "chandelier_atr_mult": 5.0,
    # NEW: per-UNIT trailing profit give-back -- every other stop here is shared across the WHOLE
    # episode (one level, applied identically to every pyramided unit), which is why a late add
    # bought near the top gets force-closed at the same price as an early add sitting on a big
    # cushion, never getting a chance to lock in ITS OWN gains first. This one is unit-aware: once
    # a unit's own profit reaches giveback_min_profit_r, its stop trails at
    # entry + giveback_fraction * (that unit's OWN peak price since ITS OWN entry - entry) -- an
    # early, deeply profitable unit can give back a lot of room and still hold; a late, barely-
    # profitable unit gets protected almost immediately. Off by default (opt-in, changes nothing
    # for anyone who doesn't turn it on).
    "use_giveback_stop": False, "giveback_fraction": 0.4, "giveback_min_profit_r": 1.0,
    "adx_period": 30,
    "use_regime_scale_out": False, "regime_scale_out_adx_threshold": 35.0, "regime_scale_out_fraction": 0.5,
    "use_pyramid_adx_filter": False, "pyramid_max_adx": 35.0,
    "use_ema_filter": True, "ema_period": 300,   # require close > EMA(period) to enter
    "use_ma_regime": False, "ma_regime_period": 200, "ma_regime_band_pct": 0.0,   # see DEFAULT_SYSTEM_PARAMS
    "use_ema_entry": False, "ema_entry_period": 200, "ema_entry_mode": "cross",   # see DEFAULT_SYSTEM_PARAMS
    "use_ema_exit": False, "ema_exit_period": 60,
    "use_channel_exit": True,                       # see DEFAULT_SYSTEM_PARAMS
    "use_monthly_high_entry": False, "mh_min_age": 20, "mh_max_age": 365, "mh_sma_period": 200, "mh_ma_type": "ema",   # see DEFAULT_SYSTEM_PARAMS
    "use_peak_trail": False, "peak_trail_atr_mult": 8.0,
    "use_choppiness_filter": False, "choppiness_period": 14, "choppiness_max": 38.2,
    "require_all_time_high": False,
    # NOTE: same addition as S1 above -- not in what you pasted, added back so the file doesn't
    # crash. S2 keeps the longer 15-day cooldown from before.
    "use_cooldown": True, "cooldown_days": 15,
}

# CLEAN MONTHLY-HIGH MODE ("only these mechanics"): one system that does exactly this and nothing else --
#   INITIAL ENTRY  close above its EMA and the first close above a monthly high (plus entry_buffer_n_mult x ATR, default 0) that is mh_min_age..mh_max_age bars old
#   PYRAMIDING     one more unit on every further such signal (same criteria; no unit cap, no spacing, no other filter)
#   SL             stop_n_mult x ATR(n_atr_period) under the newest unit's entry; every add re-sets the shared stop to the new unit's stop (the ratchet)
#   TP             every unit closes completely at its entry + tp_n_mult x ATR(tp_atr_period)
# Everything else (channel exit, trailing / chandelier / break-even / give-back / peak-trail / time stops, regime exits, ADX / EMA / choppiness / ATH filters, cooldown,
# risk schedule) is switched off. The entry buffer is part of the entry, so it is a setting (entry_buffer_n_mult, 0 when absent), not a constant here. The optimizer / walk-forward applies this dict on top of DEFAULT_SYSTEM_PARAMS when MH_CLEAN_MODE is True (see params_to_systems); the UI
# builds the same system for a plain backtest.
MH_CLEAN_MODE = False
MH_CLEAN_SYSTEM_PARAMS = {
    "use_monthly_high_entry": True, "mh_ma_type": "ema", "pyramid_method": "monthly_high",
    "risk_per_trade_pct_schedule": None,
    "use_channel_exit": False,
    "use_fixed_stop": True, "ratchet_stop_on_pyramid": True, "use_average_entry_stop": False,
    "use_scale_out": True, "scale_out_levels": [(1.0, 1.0)],
    "use_n_day_low_trailing_stop": False, "use_breakeven_stop": False, "use_chandelier_exit": False,
    "use_giveback_stop": False, "use_peak_trail": False, "use_time_stop": False,
    "use_regime_scale_out": False, "use_pyramid_adx_filter": False,
    "use_ema_filter": False, "use_ema_entry": False, "use_ema_exit": False, "use_ma_regime": False,
    "use_choppiness_filter": False, "require_all_time_high": False, "use_cooldown": False,
}

# Set either of these to False to run with just ONE system -- everything downstream (feature
# computation, simulation, plotting, stats) already iterates generically over whatever's in
# SYSTEMS, so turning a system off here is enough; no other code needs to change.
ENABLE_S1 = True
ENABLE_S2 = True

# breakout_lookback/exit_lookback/name live HERE, not inside DEFAULT_SYSTEM_PARAMS/s2_SYSTEM_PARAMS
# -- if a shared-params dict ever contains its own "breakout_lookback" (or "exit_lookback" or
# "name") key, its value SILENTLY WINS over what's set here, since the dict spread (**...) is
# unpacked AFTER these explicit keys in the literal below. Keep lookbacks/name OUT of the shared
# dicts entirely to avoid that trap.
SYSTEMS = [
    {"name": "S1", "breakout_lookback": 20, "exit_lookback": 15, **DEFAULT_SYSTEM_PARAMS},
    {"name": "S2", "breakout_lookback": 300, "exit_lookback": 20, **s2_SYSTEM_PARAMS},
]
SYSTEMS = [s for s in SYSTEMS if (s["name"] != "S1" or ENABLE_S1) and (s["name"] != "S2" or ENABLE_S2)]
assert len(SYSTEMS) > 0, "At least one system must be enabled -- set ENABLE_S1 or ENABLE_S2 to True"

# Per-asset cap on S1+S2 combined notional, as % of account equity. 300% (3x) is the default: in
# testing, raising it to 5000% (the old default) changed nothing except on tiny-risk settings, while
# lowering it to 100% cut only about 2 CAGR points. The risk-per-trade and open-risk settings, not
# this cap, are what set the drawdown. Change it here or in the Backtest tab.
MAX_TOTAL_EXPOSURE_PCT = 300

# The Turtles' account-adjustment rule. They did not size positions from their real equity but from a
# NOTIONAL equity that fell twice as fast: for every 10% the account lost they traded as if it were 20%
# smaller (80% of the reference at -10%, 60% at -20%, nothing at -50%), and the cut is undone as the account
# recovers. Only NEW units are sized from it; open units stay as they are. It responds to how the ACCOUNT is
# doing, which N-based sizing (the asset's own volatility) does not. ON by default.
USE_DRAWDOWN_SCALING = True
DRAWDOWN_SCALING_THRESHOLD = 0.10   # each full 10% the account is below the reference is one step
DRAWDOWN_SCALING_FACTOR = 0.20      # each step takes this fraction of the reference off the notional equity
DRAWDOWN_SCALING_REFERENCE = "year_start"   # what the loss is measured from:
#   "year_start"  the account's value at the end of the previous year (the Turtles' rule: it resets every January)
#   "peak"        the account's running high (stricter: the cut stays until the account makes a new high)

# ============================================================================
# PORTFOLIO MODE (simulate_portfolio(), opt-in -- simulate_asset() above is untouched and still
# treats every asset as its own fully independent $STARTING_CAPITAL_PER_ASSET account)
# ============================================================================
# NOTE: the DRAWDOWN_SCALING_* settings above are reused as-is in portfolio mode, not duplicated --
# they operate on whichever equity they're handed (see _EquityState), which is one shared
# portfolio-wide value in simulate_portfolio() instead of a private per-asset one, so the exact same
# toggle becomes genuine portfolio-level account adjustment automatically, with no separate constant needed.
PORTFOLIO_STARTING_CAPITAL = None   # None = STARTING_CAPITAL_PER_ASSET * however many assets are
                                      # actually passed to simulate_portfolio() that run
USE_PORTFOLIO_RISK_CAP = True
MAX_PORTFOLIO_RISK_PCT = 20.0   # sum of every OPEN position's risk_amount -- what you'd actually
                                  # lose if every stop got hit at once -- across EVERY asset and
                                  # both systems, capped at this % of portfolio equity. This is
                                  # the actual answer to "how bad could a correlated crash get,"
                                  # which no per-asset backtest can show (each one thinks it has
                                  # the WHOLE account's risk budget to itself). The existing
                                  # MAX_TOTAL_EXPOSURE_PCT above still applies too, unchanged --
                                  # it caps S1+S2's combined NOTIONAL on one asset; this caps total
                                  # RISK across the whole portfolio. Different constraints, both
                                  # active at once in portfolio mode.


SLIPPAGE_PCT = 0.02
SPREAD_PCT = 0.02
COMMISSION_PCT = 0.01
HOLDING_COST_PCT_PER_YEAR = 2.0

# Per-contract futures commissions (see ibkr_fees.py, which fills these in). When on, an asset that
# has an entry in FEE_SCHEDULE is charged shares / multiplier * fee_per_contract per fill INSTEAD of
# COMMISSION_PCT; assets without an entry keep the percentage commission. multiplier = dollars of
# P&L per contract per 1.0 move in the price series, so shares / multiplier = number of contracts.
USE_PER_CONTRACT_FEES = False
FEE_SCHEDULE = {}   # {asset name: {"multiplier": float, "fee_per_contract": float, "code": str}}

# Account realism, portfolio mode only (simulate_portfolio(); see margin_sizing.py, which fills
# these in from the Fees tab). All off/empty by default, which keeps the original behavior.
#   MARK_TO_MARKET: the equity curve, position sizing, the exposure cap and drawdown scaling use
#       account equity = realized cash + the unrealized P&L of open positions, instead of realized
#       cash only. Without it a position that is 30% under water does not show up anywhere until it
#       is closed. Forced on by USE_MARGIN_CALLS.
#   SIZE_STEP: {asset: smallest tradable size, in engine shares}. A futures contract is `multiplier`
#       shares, a CFD lot step is a fraction of a unit. Sizes are rounded DOWN to a whole number of
#       steps; an asset missing from the dict is sized fractionally as before.
#   TAKE_MIN_ONE_UNIT: when the risk-based size rounds down to zero steps, open one step anyway
#       (as long as the exposure cap and margin allow it) instead of skipping the trade.
#   USE_MARGIN_CALLS / MARGIN_SPEC: {asset: (initial rate, maintenance rate)} as fractions of the
#       position's current notional. A new unit is only opened (or shrunk to fit) if the account can
#       pay its initial margin. At each day's close, if account equity < total maintenance margin
#       the broker liquidates whole units, largest margin user first, until equity covers it again.
#   ASSET_COSTS: {asset: {"slippage_pct", "spread_pct", "commission_pct", "holding_pct"}} -- per-asset
#       replacements for the global cost settings above (any key left out falls back to the global).
#   EXECUTION_TIMING: "same_close" fills a signal at the very close that produced it (the original,
#       optimistic: you cannot trade a close you have only just seen). "next_open" fills entries,
#       pyramid adds and channel exits at the NEXT bar's open, which is what a trader acting on the
#       signal actually gets. Stop exits are unaffected: they already fill at the stop or the open.
ASSET_COSTS = {}
#   STOP_SLIPPAGE_PCT: extra slippage, in % of the stop level, on a stop-type exit that fills at its level (not on a gap, which already fills at
#       the open). A stop order fills at or beyond its level; in fast markets beyond. 0 keeps the original fill at the level itself.
STOP_SLIPPAGE_PCT = 0.0
EXECUTION_TIMING = "same_close"
#   RECORD_HISTORY: simulate_portfolio() also returns a day-by-day record of the account (cash,
#       unrealized profit, margin, open positions, costs) for the Backtest tab's replay. Off by
#       default and for every search, since it costs time and memory.
#   STOP_PRIORITY: when SL, trailing and chandelier stops are all under the day's low, "highest_level"
#       fills at the highest of them (the one price reaches first on the way down); "legacy" always
#       prefers SL, then trailing, then chandelier, which can fill lower than any real order would.
STOP_PRIORITY = "highest_level"
#   ASSET_PRIORITY: several assets can signal on the same day while the risk or margin budget only has
#       room for some. "listed" always serves them in the order they were loaded (SP500 always first),
#       which hands the same assets the budget every time; "rotate" shifts the starting asset by one
#       each date, so no asset is permanently first.
ASSET_PRIORITY = "rotate"
#   NEVER_LOOSEN_STOPS: a pyramid add normally SETS the shared stop to (new entry - stop distance), which
#       can be LOWER than the stop already in force (e.g. after the breakeven ratchet, or when N has
#       widened). True keeps the stop where it is if the new level would be lower, as stops are
#       normally managed. False is the original behavior.
NEVER_LOOSEN_STOPS = False
# Portfolio-level risk overlays (all off by default, portfolio mode only). None of them tries to
# recognize a market regime (choppiness/trend-strength gauges look back many bars, so they only
# notice a sideways market after the failed breakouts have happened); each limits exposure directly.
#   MAX_GROSS_EXPOSURE_PCT   cap on the TOTAL notional of every open position, % of account equity
#                            (MAX_TOTAL_EXPOSURE_PCT only caps ONE asset's notional)
#   VOL_TARGET_PCT           when set, new risk is scaled by target / the account's own realized volatility
#                            over the last VOL_TARGET_WINDOW bars, kept between VOL_TARGET_FLOOR and
#                            VOL_TARGET_CAP (cap 1.0 = only ever cuts risk). Uses yesterday's reading.
#   HITRATE_WINDOW           when set, new risk is multiplied by HITRATE_SCALE while the average R of the
#                            last HITRATE_WINDOW closed episodes is below HITRATE_MIN_AVG_R
# Risk multipliers, applied to a new unit's risk_per_trade_pct (all 1.0 / empty = off, portfolio or per-asset mode):
#   ATH_RISK_MULT / BELOW_ATH_RISK_MULT   multiply risk when the signal bar's close makes a NEW all-time high
#                            (above yesterday's all-time high in the features) / when it does not;
#                            ATH_FIRST_UNIT_ONLY limits that to the first unit of a trend
#   ASSET_RISK_MULT          {asset name: multiplier}, e.g. to risk more on markets you believe trend more
#   ASSET_RISK_NORMALIZE     True rescales the multipliers of the markets in a portfolio run so their average
#                            is 1.0: the mix moves risk BETWEEN markets without raising the total (without
#                            it, a multiplier of 2 on everything is just twice the leverage)
#   ASSET_RISK_PARAM_PREFIX  an optimizer parameter named PREFIX + market name carries that market's
#                            multiplier (see optimize_turtle_dual_system.apply_params)
#   CHOP_RISK_MULT          multiplier when the market's Choppiness Index at the signal bar is at or below
#                            CHOP_TREND_MAX (low = trending). A lagging gauge: see the notes in the testing.
#   ATH_MAX_DISTANCE_PCT     a softer version of require_all_time_high: entries and pyramid adds are only
#                            allowed while the close is within this % of the all-time high (None = no limit)
ATH_MAX_DISTANCE_PCT = None
# ALLOW_EXIT_ABOVE_ENTRY   False: the optimizer only tries systems whose exit channel is shorter than their entry
#                          channel (the Turtle's own pairing, e.g. in on a 55-day high, out on a 20-day low).
#                          True: any pair is allowed, e.g. in on a 20-day high and out only on a 100-day low,
#                          which keeps a long-only account invested through ordinary dips. The simulation itself
#                          never cared; only the optimizer's validity check does.
ALLOW_EXIT_ABOVE_ENTRY = False
ATH_RISK_MULT = 1.0
BELOW_ATH_RISK_MULT = 1.0
ATH_FIRST_UNIT_ONLY = False
ASSET_RISK_MULT = {}
ASSET_RISK_NORMALIZE = False
ASSET_RISK_PARAM_PREFIX = "RISKMULT::"
CHOP_RISK_MULT = 1.0
CHOP_TREND_MAX = 40.0
MAX_GROSS_EXPOSURE_PCT = None
VOL_TARGET_PCT = None
VOL_TARGET_WINDOW = 126
VOL_TARGET_FLOOR = 0.25
VOL_TARGET_CAP = 1.0
HITRATE_WINDOW = None
HITRATE_MIN_AVG_R = -0.1
HITRATE_SCALE = 0.5
# Tail-aware risk. Sizing from N (everyday volatility) equalises what each market risks on an ordinary day, not what a
# bad day costs: a stop does nothing against a gap, and the account's worst days are mostly ONE market jumping while it
# holds a big position. A market's "tail" here is its own worst one-day close-to-close drop over the last TAIL_WINDOW
# bars (causal: only days up to the signal bar; TAIL_FALLBACK until TAIL_MIN_BARS days exist). All off by default.
#   GAP_SIZING_K            > 0: a new unit is sized as if its stop sat at least K x tail away, so a repeat of the worst
#                           day costs about its risk budget (strategy level; also works per asset)
#   STRESS_TOTAL_CAP_PCT    caps the sum over every open market of (notional x tail), % of account equity: the loss if
#                           every market repeated its worst day at once (portfolio mode)
#   STRESS_SINGLE_CAP_PCT   the same cap for each market on its own (the account's worst days are single-market jumps)
#   STRESS_CLUSTER_CAP_PCT  the same cap per group of markets in MARKET_CLUSTERS (sum within the group)
#   BOOK_VOL_CAP_PCT        a new unit is cut so the open book's annualised volatility, from the last BOOK_VOL_WINDOW
#                           days of every market's returns AND their correlations, stays under this (portfolio mode)
#   EQUITY_MA_WINDOW        new risk is multiplied by EQUITY_MA_SCALE while the account is below its own average of the
#                           last EQUITY_MA_WINDOW closes
#   SHOCK_PAUSE_LOSS_PCT    no new units for SHOCK_PAUSE_DAYS calendar days after a day the account lost this much
#   EXTENSION_MA            risk is multiplied by EXTENSION_RISK_MULT (0 = no unit at all) when the close is more than
#                           EXTENSION_N_MULT x N above its EXTENSION_MA-day average: stops adding to a parabolic run,
#                           where the biggest positions and the biggest reversals meet
TAIL_WINDOW = 1250
TAIL_MIN_BARS = 250
TAIL_FALLBACK = 0.15
GAP_SIZING_K = 0.0
STRESS_TOTAL_CAP_PCT = None
STRESS_SINGLE_CAP_PCT = None
STRESS_CLUSTER_CAP_PCT = None
MARKET_CLUSTERS = {
    "SP500": "equity", "Nasdaq": "equity", "Dow": "equity", "Russell2000": "equity", "Ger40": "equity", "Nikkei225": "equity",
    "Gold": "precious", "Silver": "precious", "Platinum": "precious", "Palladium": "precious",
    "Copper": "industrial", "CrudeOil": "industrial", "Corn": "agri", "Cocoa": "agri", "Bitcoin": "crypto",
}
BOOK_VOL_CAP_PCT = None
BOOK_VOL_WINDOW = 60
EQUITY_MA_WINDOW = None
EQUITY_MA_SCALE = 0.5
SHOCK_PAUSE_LOSS_PCT = None
SHOCK_PAUSE_DAYS = 10
EXTENSION_MA = None
EXTENSION_N_MULT = 6.0
EXTENSION_RISK_MULT = 0.0
# Intraday daily-loss exit: an open position is sold when the market trades more than DAILY_LOSS_STOP_N_MULT x N (yesterday's N)
# or DAILY_LOSS_STOP_PCT % below YESTERDAY'S CLOSE (the tighter of the two when both are set), filled at that level, or at the
# open when the market opened below it. The usual stop never moves during a long trend and the channel exit only looks at
# closes, so a crash that unfolds inside one day (most of the account's worst days) otherwise runs to the close.
DAILY_LOSS_STOP_N_MULT = None
DAILY_LOSS_STOP_PCT = None
# Market selection (portfolio mode): every day all markets with data are ranked by trend strength = the log return over the last
# ENTRY_STRENGTH_LOOKBACK bars divided by its volatility (daily log-return std x sqrt(lookback)), so a smooth persistent trend beats a violent
# one with the same gain. With ENTRY_TOP_K set, a NEW trend (the first unit; with ENTRY_RANK_FIRST_UNIT_ONLY False, pyramid adds too) may only be
# started in a market ranked 1..K that day with strength above ENTRY_MIN_STRENGTH. Open positions are never closed by a rank change. Uses only
# data up to the signal bar.
ENTRY_TOP_K = None
ENTRY_STRENGTH_LOOKBACK = 126
ENTRY_MIN_STRENGTH = 0.0
ENTRY_RANK_FIRST_UNIT_ONLY = True
# Volatility-scaled risk (per market, on top of the N-based sizing): a new unit's risk_per_trade_pct is multiplied by
#   clip(ratio ** -VOL_RISK_GAMMA, VOL_RISK_MIN, VOL_RISK_MAX),  ratio = ATR(VOL_RISK_ATR) / its average over the previous VOL_RISK_REF bars,
# using bars up to the signal bar only. ratio 1 = the market is as volatile as usual for it. GAMMA 1 = risk inversely proportional to how loud the
# market is against its own history (less when it is wild, more when it is quiet); 0.5 is a milder version; 0 = off. Until VOL_RISK_REF bars exist
# the multiplier is 1.
VOL_RISK_GAMMA = 0.0
VOL_RISK_ATR = 20
VOL_RISK_REF = 120
VOL_RISK_MIN = 0.5
VOL_RISK_MAX = 1.5
# EMA entry: replace the N-day-high breakout by "the close breaks above an exponential moving average". ENTRY_EMA_PERIOD is None (off), one number used by
# both systems, or a dict such as {"S1": 50, "S2": 200}. ENTRY_EMA_MODE "cross" enters only on the day the close moves from at-or-below the EMA to above
# it (an event, so after an exit the system waits for the next cross); "above" enters whenever the close is above the EMA and the system is flat (a
# state, so it buys straight back after any exit). The entry buffer is not used. The signal is read at the close and acted on at the next open, like every
# other signal. ENTRY_EMA_EXIT True also exits when the close falls below the EMA (instead of the N-day-low channel). Stops, sizing, pyramiding and caps
# are unchanged.
ENTRY_EMA_PERIOD = None
ENTRY_EMA_MODE = "cross"
ENTRY_EMA_EXIT = False
# Carry (portfolio mode only). Off by default: the engine then reports price returns, with idle cash earning nothing.
#   CARRY_MODE "futures": futures-style accounting. Every calendar day, the account earns the short-term interest rate on
#                         its whole equity, and each open position pays (interest rate - dividend yield) on its market
#                         value. A position worth 100% of equity therefore earns just its dividend yield, one worth 200%
#                         pays the interest rate on the extra 100%, and an account sitting in cash earns the interest rate.
#   CARRY_RATE_SERIES     annual short-term rate as a fraction, indexed by date (forward-filled; see carry_data.py)
#   CARRY_RATE_FLAT       the rate used before the series starts, or when there is none
#   CARRY_DIVIDEND_YIELD  {asset: annual dividend yield as a fraction}; an asset not listed pays none
CARRY_MODE = "off"
CARRY_RATE_SERIES = None
CARRY_RATE_FLAT = 0.0
CARRY_DIVIDEND_YIELD = {}
RECORD_HISTORY = False
MARK_TO_MARKET = False
SIZE_STEP = {}
TAKE_MIN_ONE_UNIT = False
USE_MARGIN_CALLS = False
MARGIN_SPEC = {}
DEFAULT_MARGIN_RATES = (1.0, 0.5)   # for an asset with no MARGIN_SPEC entry: fully cash funded


# REPAIR_BAD_DATA: repair known faults in a downloaded price history before it is used (see repair_ohlc). Set False to use the data exactly as
# downloaded, apart from widening each bar's range to contain its open and close.
REPAIR_BAD_DATA = True


def repair_ohlc(df: pd.DataFrame, window: int = 120):
    """Fixes the faults found in Yahoo's futures histories, after download. Returns (clean OHLC frame, report dict of what was done). In order:
      1. rows with a zero or negative price are dropped (crude oil printed -37.63 on 2020-04-20);
      2. a persistent jump by about a power of ten is a change of quoting units (the micro gold contract goes from 164 to 1,665 in one day):
         the earlier stretch is rescaled to the later scale;
      3. a one-day move of 25% or more that is almost fully reversed the next day is a bad tick and the bar is dropped;
      4. every bar's high and low are widened to contain its open and close (Yahoo mixes contracts at a roll);
      5. a bar with no range at all (open = high = low = close, many illiquid or old futures days: settlement only) is given an OPEN at the previous
         close and an ESTIMATED range, the median range of the previous `window` real bars (past bars only, so nothing looks ahead).
    Step 5 gives the right average ATR, not the true high and low of that day: a stop or trail can fire or be missed on an estimated bar.
    Clean data comes back unchanged."""
    report = {"rows_dropped": 0, "unit_rescales": 0, "spikes_removed": 0, "widened": 0, "flat_estimated": 0}
    d = df[["Open", "High", "Low", "Close"]].dropna().astype(float).copy()
    n0 = len(d)
    d = d[(d > 0).all(axis=1)]
    report["rows_dropped"] = n0 - len(d)
    ratio = d["Close"] / d["Close"].shift()
    for dt in ratio.index[(ratio > 5) | (ratio < 0.2)]:
        k = d.index.get_loc(dt)
        r = float(ratio.loc[dt])
        factor = 10.0 ** round(float(np.log10(r)))
        if factor != 1.0 and abs(r / factor - 1) < 0.25:
            d.iloc[:k] = d.iloc[:k] * factor
            report["unit_rescales"] += 1
    ret = d["Close"].pct_change()
    spike = ((ret.abs() >= 0.25) & (ret.shift(-1) * ret < 0)
             & (((1 + ret) * (1 + ret.shift(-1)) - 1).abs() < 0.10)).reindex(d.index).fillna(False)
    report["spikes_removed"] = int(spike.sum())
    d = d[~spike]
    hi = d[["Open", "High", "Low", "Close"]].max(axis=1)
    lo = d[["Open", "High", "Low", "Close"]].min(axis=1)
    report["widened"] = int(((hi > d["High"]) | (lo < d["Low"])).sum())
    d["High"], d["Low"] = hi, lo
    flat = (d["Open"] == d["High"]) & (d["High"] == d["Low"]) & (d["Low"] == d["Close"])
    if flat.any():
        rng = ((d["High"] - d["Low"]) / d["Close"]).where(~flat)
        scale = rng.rolling(window, min_periods=20).median().shift(1).ffill()      # previous real bars only
        prev_close = d["Close"].shift()
        new_open = d["Open"].where(~flat, prev_close).fillna(d["Open"])
        body = (d["Close"] - new_open).abs()
        extra = np.maximum((scale * d["Close"]).fillna(0.0), body) - body
        d["High"] = d["High"].where(~flat, np.maximum(new_open, d["Close"]) + extra / 2)
        d["Low"] = d["Low"].where(~flat, np.minimum(new_open, d["Close"]) - extra / 2)
        d["Open"] = new_open
        report["flat_estimated"] = int(flat.sum())
    return d, report


def download_data(tickers: dict, start: str, end) -> dict:
    import yfinance as yf
    data = {}
    for name, tk in tickers.items():
        df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"WARNING: no data for {name} ({tk})")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if REPAIR_BAD_DATA:
            cleaned, report = repair_ohlc(df)
        else:
            # Yahoo's continuous futures sometimes carry an Open or Close from a different contract than
            # the High/Low, leaving a bar whose high is below its own close (or low above its open). Stop
            # and gap logic needs every bar's range to contain its open and close, so widen it to do so.
            cleaned = df[["Open", "High", "Low", "Close"]].dropna().copy()
            cleaned["High"] = cleaned[["Open", "High", "Low", "Close"]].max(axis=1)
            cleaned["Low"] = cleaned[["Open", "High", "Low", "Close"]].min(axis=1)
            report = {}
        data[name] = cleaned
        data[name].attrs["asset"] = name   # lets the engine look up this asset's fee schedule
        data[name].attrs["repair"] = report
        fixed = {k: v for k, v in report.items() if v}
        print(f"{name}: {len(data[name])} rows, {data[name].index.min().date()} -> {data[name].index.max().date()}"
              + (f" (repaired: {fixed})" if fixed else ""))
    return data


def data_quality(raw: dict) -> list:
    """One row per asset describing how trustworthy its price history is for THIS strategy, which leans
    on daily highs and lows (stops, ATR) and on opens (gap fills). Returns a list of dicts."""
    rows = []
    for name, df in raw.items():
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        n = len(df)
        no_range = float((h == l).mean())
        recent = df.iloc[-min(n, 1260):]
        no_range_recent = float((recent["High"] == recent["Low"]).mean())
        flat_open = float(((o / c.shift() - 1).abs() < 1e-9).mean())
        years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
        returns = c.pct_change()
        issues = []
        rep = df.attrs.get("repair") or {}
        est_share = rep.get("flat_estimated", 0) / max(n + rep.get("rows_dropped", 0) + rep.get("spikes_removed", 0), 1)
        if est_share > 0.02:
            issues.append(f"{est_share*100:.0f}% of bars had no high-low range at all (settlement only): their range was ESTIMATED from the "
                          f"previous real bars, so ATR is right on average but a stop can fire or be missed on those days")
        fixes = [f"{rep[k]} {label}" for k, label in (("rows_dropped", "zero/negative-price rows dropped"), ("unit_rescales", "unit change(s) rescaled"),
                                                       ("spikes_removed", "bad tick(s) removed")) if rep.get(k)]
        if fixes:
            issues.append("repaired: " + ", ".join(fixes))
        if no_range_recent > 0.15:
            issues.append(f"{no_range_recent*100:.0f}% of the last 5 years' bars have no high-low range, so "
                          f"stops fire on closes only and N is understated")
        elif no_range > 0.15:
            issues.append(f"{no_range*100:.0f}% of bars (mostly early history) have no high-low range")
        if flat_open > 0.25:
            issues.append(f"the open equals the previous close on {flat_open*100:.0f}% of days, so gap risk is understated")
        if (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=None):
            issues.append("contains zero or negative prices")
        if returns.abs().max() > 0.25:
            issues.append(f"a one-day move of {returns.abs().max()*100:.0f}% (real, or a bad tick / contract roll)")
        if int((df.index.dayofweek >= 5).sum()) > 0:
            issues.append("trades on weekends (24/7 market)")
        rows.append({"Asset": name, "First bar": df.index[0].date(), "Years": round(years, 1),
                     "Bars with no range (%)": round(no_range * 100, 1),
                     "Open = prev close (%)": round(flat_open * 100, 1),
                     "Worst 1-day move (%)": round(float(returns.min()) * 100, 1),
                     "Verdict": "OK" if not issues else "; ".join(issues)})
    return rows


def fetch_all_time_highs(tickers: dict) -> dict:
    """Fetches the MAXIMUM available history for each ticker (period='max' -- as far back as
    Yahoo Finance actually has data), completely decoupled from the backtest's own START_DATE
    window, and returns each asset's running all-time-high (by intraday High, cumulative max)
    as a full-history Series.

    WHY this exists: without it, "all-time high" only ever means "highest point within
    whatever window the backtest happens to be running over" -- if the true historical peak
    happened before START_DATE, the require_all_time_high filter would never know about it,
    understating the real level. This fetches the genuinely longest available record per
    ticker specifically to fix that, independent of how far back the actual backtest runs.

    Returns {name: pd.Series} -- each series is the running max High, at full available
    history's daily resolution. Callers should reindex/ffill this onto their own backtest
    date range (see build_features' full_history_ath parameter). If a fetch fails for any
    ticker, that ticker is simply omitted -- callers should fall back to the in-window
    cummax behavior for it rather than crash.
    """
    import yfinance as yf
    result = {}
    for name, tk in tickers.items():
        try:
            df = yf.download(tk, period="max", progress=False, auto_adjust=True)
            if df.empty:
                print(f"  WARNING: no full-history data for {name} ({tk}) -- falling back to "
                      f"in-window all-time-high for this asset")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            high = df["High"].dropna()
            result[name] = high.cummax()
            print(f"  {name}: full history back to {high.index.min().date()} "
                  f"(true all-time high so far: {high.cummax().iloc[-1]:.2f})")
        except Exception as e:
            print(f"  WARNING: couldn't fetch full history for {name} ({tk}): {e} -- falling "
                  f"back to in-window all-time-high for this asset")
    return result


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0)


def compute_choppiness_index(df: pd.DataFrame, period: int) -> pd.Series:
    """Same formula validated in regime_diagnostic.py: LOW values = trending (price moving
    directly to its destination), HIGH values = choppy (zigzagging). Range is roughly 0-100."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    tr_sum = tr.rolling(period).sum()
    range_high = high.rolling(period).max()
    range_low = low.rolling(period).min()
    span = (range_high - range_low).replace(0, np.nan)
    ci = 100 * np.log10(tr_sum / span) / np.log10(period)
    return ci


def compute_confirmed_swing_high_level(df: pd.DataFrame, k: int) -> pd.Series:
    """The level of the most recently CONFIRMED swing high, as genuinely known on each day --
    used by the "swing_high" pyramid method. A swing high at day j requires k bars on BOTH
    sides higher than nothing exceeding it, which structurally means it can only be confirmed k
    days AFTER it happens (you can't know nothing exceeded it until those k days have passed).
    That confirmation lag is the entire point, not a flaw -- it's what forces a genuine pullback
    to happen before the "last swing high" reference can move at all, unlike a rolling-N-day-high
    check (which updates continuously and can fire every single day of an uninterrupted thrust,
    confirmed directly in this project: it still added on all 8 days of an 8-day rally even with
    a 55-day lookback).

    Mechanics:
      1. A centered (2k+1)-bar rolling max identifies whether day j's High is the highest in its
         own neighborhood [j-k, j+k] -- this doesn't leak into trading decisions on its own,
         since step 2 only treats it as "known" starting k days later.
      2. Shifting that by k days moves day j's peak value to position j+k -- exactly when it
         becomes confirmed (nothing in [j+1, j+k] exceeded it).
      3. Forward-filling keeps that as "the current reference level" until an even newer swing
         high gets confirmed.
      4. One more day of shift for the same lookahead-avoidance convention used everywhere else
         in this file: a level computed through yesterday's close gates today's decision, never
         today's own price.
    """
    high = df["High"]
    window = 2 * k + 1
    is_peak = high == high.rolling(window, center=True).max()
    peak_price = high.where(is_peak)
    confirmed_peak = peak_price.shift(k)
    swing_high_level = confirmed_peak.ffill()
    return swing_high_level.shift(1)


def ma_regime_levels(close: pd.Series, period: int, band_pct: float):
    """(entry level, exit level) of a moving-average regime system: the simple average of the last `period`
    closes, today's included (the signal is read at the close and acted on at the next open), raised and
    lowered by the band. The entry fires on a close above the first, the exit on a close below the second."""
    ma = close.rolling(int(period)).mean()
    band = float(band_pct) / 100.0
    return ma * (1.0 + band), ma * (1.0 - band)


def ema_entry_settings(system_name: str, sysdef: dict = None):
    """(period, mode) of the EMA entry for this system, or None when it is off. The system's own `use_ema_entry` setting wins; otherwise the
    module-level ENTRY_EMA_* switch (used by the research scripts) applies."""
    if sysdef is not None and sysdef.get("use_ema_entry"):
        return int(sysdef["ema_entry_period"]), sysdef.get("ema_entry_mode", "cross")
    period = ENTRY_EMA_PERIOD.get(system_name) if isinstance(ENTRY_EMA_PERIOD, dict) else ENTRY_EMA_PERIOD
    if not period:
        return None
    return int(period), ENTRY_EMA_MODE


def ema_exit_period(system_name: str, sysdef: dict = None):
    """Length of the EMA the close must fall below to exit (replacing the exit channel), or None when the channel exit stays. The system's own
    `use_ema_exit` setting wins; otherwise the research switch ENTRY_EMA_EXIT exits at the entry EMA of a system that has one."""
    if sysdef is not None and sysdef.get("use_ema_exit"):
        return int(sysdef["ema_exit_period"])
    if ENTRY_EMA_EXIT:
        settings = ema_entry_settings(system_name, sysdef)
        if settings is not None:
            return settings[0]
    return None


def entry_ema_levels(close: pd.Series, system_name: str, sysdef: dict = None):
    """(entry level or None, exit level or None) of the EMA entry / EMA exit for this system, or None when both are off.
    The EMAs include today's close (signal read at the close, acted on at the next open). An entry level of 'cross' mode is +infinity on every day
    except the first one after a close at-or-below the EMA, so `close > level` is true only on the crossing day. The first `period` bars of each
    level are NaN (EMA warm-up)."""
    settings = ema_entry_settings(system_name, sysdef)
    exit_len = ema_exit_period(system_name, sysdef)
    if settings is None and exit_len is None:
        return None
    entry = exit_level = None
    if settings is not None:
        period, mode = settings
        ema = close.ewm(span=period, adjust=False).mean()
        if mode == "cross":
            was_at_or_below = (close.shift(1) <= ema.shift(1)).fillna(False)
            entry = ema.where(was_at_or_below, np.inf)
        elif mode == "above":
            entry = ema.copy()
        else:
            raise ValueError(f"EMA entry mode must be 'cross' or 'above', got {mode!r}")
        entry.iloc[:period] = np.nan
    if exit_len is not None:
        exit_level = close.ewm(span=exit_len, adjust=False).mean()
        exit_level.iloc[:exit_len] = np.nan
    return entry, exit_level


def monthly_high_settings(system_name: str, sysdef: dict = None):
    """(min_age, max_age, ma_period, ma_type) of the monthly-high breakout entry for this system, or None when it is off. ma_type is "ema" (default) or "sma"."""
    if sysdef is not None and sysdef.get("use_monthly_high_entry"):
        return (int(sysdef["mh_min_age"]), int(sysdef["mh_max_age"]), int(sysdef.get("mh_sma_period", 0) or 0), str(sysdef.get("mh_ma_type", "ema")).lower())
    return None


def monthly_high_add_settings(sysdef: dict = None):
    """Same tuple for the monthly-high ADD trigger, or None. A system with the monthly-high ENTRY adds on every new signal of that entry (every time a new entry would have
    happened); pyramid_method == "monthly_high" gives any other entry type the same adds (using the mh_* parameters)."""
    if sysdef is not None and (sysdef.get("use_monthly_high_entry") or sysdef.get("pyramid_method") == "monthly_high"):
        return (int(sysdef.get("mh_min_age", 20)), int(sysdef.get("mh_max_age", 365)), int(sysdef.get("mh_sma_period", 0) or 0), str(sysdef.get("mh_ma_type", "ema")).lower())
    return None


def monthly_high_months(index: pd.DatetimeIndex, high: pd.Series):
    """For every calendar month of the data: its high (the highest HIGH), the bar number of the bar that set it (the most recent bar wins a tie) and the bar number of the first bar AFTER the
    month (from there the month counts as completed). Three arrays of one value per month."""
    h = high.to_numpy(dtype=float)
    n = len(h)
    codes = pd.factorize(index.to_period("M"), sort=True)[0]
    n_groups = int(codes.max()) + 1 if n else 0
    top = np.full(n_groups, np.nan)
    pos = np.full(n_groups, np.nan)
    nxt = np.full(n_groups, n, dtype=int)
    for g in range(n_groups):
        where = np.flatnonzero(codes == g)
        vals = h[where]
        nxt[g] = int(where[-1]) + 1
        if np.isnan(vals).all():
            continue
        best = np.nanmax(vals)
        top[g] = best
        pos[g] = where[np.flatnonzero(vals == best)[-1]]
    return top, pos, nxt


def monthly_high_events(close: pd.Series, months, n_series: pd.Series, buffer_n_mult: float, min_age: int, max_age: int) -> np.ndarray:
    """True on every bar that is the FIRST breakout of at least one monthly high that is usable on that bar. A monthly high breaks out the first time, after its month ended, that the close is
    above (the high + buffer_n_mult x N of that bar); it is usable when the bar that set it is `min_age`..`max_age` bars old. Either way the high is used up by its first breakout, so a
    later re-cross of the same high is not a signal, and a high broken while still too young never is."""
    top, pos, nxt = months
    c = close.to_numpy(dtype=float)
    n = len(c)
    buf = float(buffer_n_mult) * n_series.to_numpy(dtype=float)
    event = np.zeros(n, dtype=bool)
    for g in range(len(top)):
        t0 = int(nxt[g])
        if t0 >= n or np.isnan(top[g]):
            continue
        broken = c[t0:] > top[g] + buf[t0:]                         # NaN N -> False: nothing breaks out before N exists
        if not broken.any():
            continue
        t = t0 + int(np.argmax(broken))
        if int(min_age) <= t - pos[g] <= int(max_age):
            event[t] = True
    return event


def monthly_high_entry_level(close: pd.Series, months, n_series: pd.Series, buffer_n_mult: float, min_age: int, max_age: int, ma_period: int, ma_type: str = "ema") -> pd.Series:
    """The entry level of the monthly-high breakout, built so that the engine's plain test `close > level` fires exactly on a breakout day of monthly_high_events: 0 on those days,
    +infinity on every other day, NaN (no entry, no add) until `max_age` bars of history exist, while N is unknown, or while the close is not above its `ma_period`-day moving average
    (an EMA by default, or a simple average with ma_type "sma"; today's close included; 0 = no filter)."""
    n = len(close)
    event = monthly_high_events(close, months, n_series, buffer_n_mult, min_age, max_age)
    level = np.where(event, 0.0, np.inf)
    level = np.where(n_series.isna().to_numpy() | (np.arange(n) < int(max_age)), np.nan, level)
    entry = pd.Series(level, index=close.index)
    if ma_period and int(ma_period) > 0:
        if str(ma_type).lower() == "sma":
            average = close.rolling(int(ma_period)).mean()
        else:
            average = close.ewm(span=int(ma_period), adjust=False).mean()
            average.iloc[:int(ma_period)] = np.nan                     # EMA warm-up, like the EMA entry
        entry = entry.where(close > average)
    return entry


def build_features(df: pd.DataFrame, full_history_ath: pd.Series = None) -> pd.DataFrame:
    out = df.copy()
    for sysdef in SYSTEMS:
        name = sysdef["name"]
        out[f"breakout_level_{name}"] = out["Close"].rolling(sysdef["breakout_lookback"]).max().shift(1)
        out[f"exit_low_level_{name}"] = out["Low"].rolling(sysdef["exit_lookback"]).min().shift(1)
        if not sysdef.get("use_channel_exit", True):
            out[f"exit_low_level_{name}"] = np.nan                               # no channel exit: a NaN level never triggers (see channel_exit in simulate)
        if sysdef.get("use_ma_regime"):
            out[f"breakout_level_{name}"], out[f"exit_low_level_{name}"] = ma_regime_levels(
                out["Close"], sysdef["ma_regime_period"], sysdef["ma_regime_band_pct"])
        _ema_levels = entry_ema_levels(out["Close"], name, sysdef)
        if _ema_levels is not None:
            if _ema_levels[0] is not None:
                out[f"breakout_level_{name}"] = _ema_levels[0]
            if _ema_levels[1] is not None:
                out[f"exit_low_level_{name}"] = _ema_levels[1]
        # pyramid "breakout" method (optional alternative to price-distance spacing): a fresh
        # N-day high, using ITS OWN lookback -- completely separate from breakout_level above,
        # so pyramid-add sensitivity can be tuned independently of the entry signal
        out[f"pyramid_breakout_level_{name}"] = out["Close"].rolling(
            sysdef["pyramid_breakout_lookback"]).max().shift(1)
        # pyramid "swing_high" method: requires a genuinely CONFIRMED swing pivot (real
        # structure, not just a rolling window) -- see compute_confirmed_swing_high_level's own
        # docstring for the full reasoning
        out[f"swing_high_level_{name}"] = compute_confirmed_swing_high_level(out, sysdef["swing_lookback"])
        out[f"trailing_stop_low_{name}"] = out["Low"].rolling(sysdef["trailing_stop_lookback_days"]).min().shift(1)
        out[f"n_{name}"] = compute_atr(out, sysdef["n_atr_period"])
        _mh = monthly_high_settings(name, sysdef)      # the monthly-high breakout entry (needs N for its buffer); overrides the EMA entry / regime levels above
        if _mh is not None:
            out[f"breakout_level_{name}"] = monthly_high_entry_level(
                out["Close"], monthly_high_months(out.index, out["High"]), out[f"n_{name}"], sysdef["entry_buffer_n_mult"], _mh[0], _mh[1], _mh[2], _mh[3])
        _mha = monthly_high_add_settings(sysdef)       # pyramid_method "monthly_high": the same breakout event, used to ADD while in a position
        if _mha is not None:
            out[f"mh_add_level_{name}"] = monthly_high_entry_level(
                out["Close"], monthly_high_months(out.index, out["High"]), out[f"n_{name}"], sysdef["entry_buffer_n_mult"], _mha[0], _mha[1], _mha[2], _mha[3])
        # TP's OWN N, computed with a potentially different period than the stop's N -- fully
        # decoupled, so TP distance doesn't have to move in lockstep with stop distance
        out[f"tp_n_{name}"] = compute_atr(out, sysdef["tp_atr_period"])
        # ADX is kept for the optional regime_scale_out take-profit mechanic and the optional
        # pyramid-add ADX filter -- not used as an entry filter anywhere.
        out[f"adx_{name}"] = compute_adx(out, sysdef["adx_period"])
        # EMA trend filter (optional, entry-side): require close > EMA(period) to enter
        out[f"ema_trend_{name}"] = out["Close"].ewm(span=sysdef["ema_period"], adjust=False).mean()
        # Choppiness Index filter (optional, entry-side): require CI below threshold to enter
        out[f"choppiness_{name}"] = compute_choppiness_index(out, sysdef["choppiness_period"])
        # all-time-high (optional, entry-side): built on the running max HIGH as of YESTERDAY
        # (shifted, same lookahead-avoidance convention as breakout_level above) -- NOT the max
        # close. The logic behind this filter is "no one above water waiting to sell" -- someone
        # who bought at an intraday spike is genuinely a real potential seller at that level even
        # if price never actually CLOSED there, so the reference level itself uses High. The
        # actual entry TRIGGER still compares against today's CLOSE (see ath_ok below), consistent
        # with how every other entry condition in this file works -- this only changes what
        # level has to be cleared, not the mechanism that checks it.
        if full_history_ath is not None:
            # a GENUINE all-time-high, sourced from the full available history (see
            # fetch_all_time_highs), not just whatever's inside this particular backtest window.
            # Reindexed onto this df's own dates (ffill, since the source series may have gaps
            # or a different calendar), then shifted by 1 for the same lookahead-avoidance
            # convention as everything else. Also unioned with this window's own cummax, in case
            # (rare, but possible) this window's own data extends slightly past what the
            # full-history fetch returned (e.g. a data-provider lag on the most recent days).
            reindexed = full_history_ath.reindex(out.index.union(full_history_ath.index)).ffill()
            reindexed = reindexed.reindex(out.index)
            combined = pd.concat([reindexed, out["High"].cummax()], axis=1).max(axis=1)
            out[f"all_time_high_{name}"] = combined.shift(1)
        else:
            # fallback: "all-time" means "within the downloaded dataset so far" -- the honest,
            # available-data definition when no external full-history reference was supplied;
            # there's no way to know a true all-time high beyond that in this case.
            out[f"all_time_high_{name}"] = out["High"].cummax().shift(1)
    return out


def _asset_cost(asset, key: str, default: float) -> float:
    """A cost setting for one asset: its ASSET_COSTS entry if it has one for `key`, else the global."""
    if ASSET_COSTS:
        overrides = ASSET_COSTS.get(asset)
        if overrides is not None and key in overrides:
            return overrides[key]
    return default


def entry_fill_price(signal_price: float, asset=None) -> float:
    slippage = _asset_cost(asset, "slippage_pct", SLIPPAGE_PCT)
    spread = _asset_cost(asset, "spread_pct", SPREAD_PCT)
    return signal_price * (1 + (slippage + spread / 2) / 100.0)


def exit_fill_price(signal_price: float, asset=None) -> float:
    slippage = _asset_cost(asset, "slippage_pct", SLIPPAGE_PCT)
    spread = _asset_cost(asset, "spread_pct", SPREAD_PCT)
    return signal_price * (1 - (slippage + spread / 2) / 100.0)


def commission_cost(notional: float, asset=None) -> float:
    return notional * (_asset_cost(asset, "commission_pct", COMMISSION_PCT) / 100.0)


def commission_for(asset, shares: float, notional: float) -> float:
    """Commission for one fill: per-contract if USE_PER_CONTRACT_FEES and this asset has a
    FEE_SCHEDULE entry, otherwise the percentage commission (unchanged default behavior)."""
    if USE_PER_CONTRACT_FEES:
        spec = FEE_SCHEDULE.get(asset)
        if spec is not None:
            return shares / spec["multiplier"] * spec["fee_per_contract"]
    return commission_cost(notional, asset)


def daily_holding_cost(notional: float, asset=None, calendar_days: float = 365.0 / 252.0) -> float:
    """Holding cost for `calendar_days` days of an open position: the yearly rate x days / 365, the way
    swap and financing are charged (a Monday bar pays for the weekend, a 7-day crypto bar for one day).
    The default is the average gap between two trading-day bars."""
    return notional * (_asset_cost(asset, "holding_pct", HOLDING_COST_PCT_PER_YEAR) / 100.0) * calendar_days / 365.0


def _round_partial(asset, shares: float, available: float, up: bool = False) -> float:
    """Whole-contract mode: a partial exit is a whole number of steps too (nearest, or rounded up
    when `up` so a trim is never short of what the cap needs), never more than what is open."""
    step = SIZE_STEP.get(asset) if SIZE_STEP else None
    if not step:
        return shares
    n = np.ceil(shares / step - 1e-9) if up else np.floor(shares / step + 0.5)
    return float(min(n * step, available))


def size_unit(equity: float, entry_price: float, n_value: float, stop_n_mult: float, current_notional: float,
              risk_pct: float = 1.0, portfolio_risk_room_left: float = None,
              step: float = None, take_min_one: bool = False, margin_room_notional: float = None,
              counters=None, skip_key=None, gross_room_notional: float = None,
              min_risk_distance: float = None, stress_room_notional: float = None):
    """
    Authentic Turtle sizing: contracts/shares = account_risk / (stop_n_mult * N in
    price terms). Verified directly against the book's own worked examples --
    e.g. a $50,000 account risking 2% ($1,000) on live cattle with N=0.80
    ($320/point) and a 2N stop gives $1,000 / (2 * 320) = 1.5625, which
    matches the book's "1.56, rounded down to 1.0" exactly.

    This means risk_pct is a genuine, direct "if the stop is hit, you lose
    approximately risk_pct% of equity" figure -- NOT "risk-per-N" the way an
    earlier version of this code mistakenly treated it (dividing by bare N
    instead of the full stop distance sized positions stop_n_mult times too
    large, and made a stated "2% risk" actually mean a 2*stop_n_mult% loss
    at the stop).

    portfolio_risk_room_left (optional): used ONLY by simulate_portfolio()'s shared-capital mode --
    the remaining portfolio-wide risk budget (MAX_PORTFOLIO_RISK_PCT% of portfolio equity, minus
    every OTHER currently-open position's own risk_amount, across every asset/system). When given,
    the position is ALSO shrunk (or rejected, same as the notional cap below already does) so its
    own risk_amount fits within this, on top of the existing per-asset notional cap -- whichever
    constraint binds tighter wins. None (the default, and what simulate_asset() always passes)
    means there's no portfolio-wide budget to check, leaving single-asset behavior unchanged.
    """
    if np.isnan(n_value) or n_value <= 0 or stop_n_mult <= 0:
        return None
    risk_amount = equity * (risk_pct / 100.0)
    stop_distance = stop_n_mult * n_value
    # GAP_SIZING_K: the unit is sized as if the stop were at least this far away (the market's own worst day); the risk
    # recorded below (and checked against the portfolio risk cap) stays the nominal stop distance
    target_shares = risk_amount / (stop_distance if not min_risk_distance else max(stop_distance, min_risk_distance))
    target_notional = target_shares * entry_price

    max_total_notional = (MAX_TOTAL_EXPOSURE_PCT / 100.0) * equity
    room_left = max_total_notional - current_notional
    if room_left <= 0:
        return None

    final_notional = min(target_notional, room_left)

    if portfolio_risk_room_left is not None:
        if portfolio_risk_room_left <= 0:
            return None
        # risk scales linearly with notional at this fixed stop distance (risk = notional *
        # stop_distance / entry_price), so capping risk is equivalent to capping notional at
        # room_left rescaled by (entry_price / stop_distance) -- same clamp-or-reject shape as
        # the per-asset notional cap right above, just against a different budget
        max_notional_by_risk = portfolio_risk_room_left * (entry_price / stop_distance)
        final_notional = min(final_notional, max_notional_by_risk)

    if gross_room_notional is not None:
        if gross_room_notional <= 0:
            if counters is not None:
                counters.n_gross_blocked += 1
            return None
        final_notional = min(final_notional, gross_room_notional)

    if stress_room_notional is not None:
        if stress_room_notional <= 0:
            if counters is not None:
                counters.n_stress_blocked += 1
            return None
        final_notional = min(final_notional, stress_room_notional)

    if margin_room_notional is not None:
        # initial-margin budget: notional the account can still fund at this asset's margin rate
        if margin_room_notional <= 0:
            if counters is not None:
                counters.note_skip("n_margin_blocked", skip_key)
            return None
        if final_notional > margin_room_notional:
            if counters is not None:
                counters.note_skip("n_margin_blocked", skip_key)
            final_notional = margin_room_notional

    final_shares = final_notional / entry_price
    if step:
        whole_steps = int(np.floor(final_shares / step + 1e-9))
        if whole_steps < 1:
            one_notional = step * entry_price
            if (not take_min_one or one_notional > room_left
                    or (margin_room_notional is not None and one_notional > margin_room_notional)):
                if counters is not None:
                    counters.note_skip("n_rounding_skips", skip_key)
                return None
            whole_steps = 1
        final_shares = whole_steps * step
        final_notional = final_shares * entry_price
    if final_shares <= 1e-9:
        # a dust-sized unit (the exposure/risk room left was a sliver) is rejected like any other
        # "no room" case: the stepping code drops units this small without ever recording them as
        # closed, which used to crash closing out the episode at low exposure caps
        return None
    actual_risk_amount = final_shares * stop_distance
    return {
        "shares": final_shares, "risk_amount": actual_risk_amount,
        "capital_deployed": final_notional, "n_value": n_value,
    }


class _EquityState:
    """Mutable equity/risk accounting, threaded through every position-sizing and P&L decision in
    _make_unit_functions()/_process_systems_and_exposure() below. simulate_asset() creates a
    private instance per call (today's per-asset-isolated behavior, unchanged). simulate_portfolio()
    creates ONE instance and shares it across every asset's context -- that's the entire mechanism
    turning "N independent per-asset backtests" into one real shared-capital simulation: the
    stepping functions don't know or care which mode they're in, they just read/write whichever
    instance they're given. USE_DRAWDOWN_SCALING (module-level) consults THIS instance's own
    equity, peak and year-start value, so it becomes genuine portfolio-wide account adjustment
    automatically in portfolio mode, with no separate constant needed."""
    __slots__ = ("year", "year_start", "last_equity", "equity", "peak", "open_risk_total", "portfolio_risk_cap_pct",
                  "total_slippage_spread_cost", "total_commission_cost", "total_holding_cost", "total_carry",
                  "contexts", "n_rounding_skips", "n_margin_blocked", "margin_events", "last_skip",
                  "max_margin_use", "wiped_out", "vol_scale", "recent_r", "n_gross_blocked",
                  "n_stress_blocked", "eq_hist", "eq_sum", "pause_until", "cur_row", "ret_matrix", "asset_col", "cov_cache",
                  "n_rank_blocked")

    def __init__(self, starting_capital: float, portfolio_risk_cap_pct: float = None):
        self.contexts = None          # {asset: ctx}; set by simulate_portfolio() only
        self.last_skip = {}           # (counter, asset, system) -> bar of its latest skip
        self.n_rounding_skips = 0     # entries skipped because the size rounded to zero contracts
        self.n_margin_blocked = 0     # entries refused or shrunk because the account could not fund the margin
        self.margin_events = []       # one dict per margin call
        self.max_margin_use = 0.0     # highest initial margin / account equity seen at a daily close
        self.wiped_out = False
        self.vol_scale = 1.0          # VOL_TARGET_PCT: today's multiplier on new risk
        self.recent_r = []            # HITRATE_WINDOW: R multiples of the latest closed episodes
        self.n_gross_blocked = 0
        self.n_stress_blocked = 0     # entries refused because a STRESS_* or BOOK_VOL cap left no room
        self.n_rank_blocked = 0       # signal-days refused because the market was not among the ENTRY_TOP_K strongest
        self.eq_hist = []             # EQUITY_MA_WINDOW: the account's latest daily closes
        self.eq_sum = 0.0
        self.pause_until = None       # SHOCK_PAUSE_LOSS_PCT: no new units up to and including this date
        self.cur_row = None           # BOOK_VOL_CAP_PCT: today's row in ret_matrix (daily returns on a common calendar)
        self.ret_matrix = None
        self.asset_col = {}
        self.cov_cache = (None, None)
        self.equity = starting_capital
        self.peak = starting_capital
        self.year = None              # calendar year of the latest bar seen; year_start is the account as that year began
        self.year_start = starting_capital
        self.last_equity = starting_capital   # the account as it closed the latest processed bar
        self.open_risk_total = 0.0   # sum of risk_amount across every unit CURRENTLY open under
                                       # this state -- only consulted when portfolio_risk_cap_pct
                                       # is set, but kept up to date unconditionally (cheap) so
                                       # turning the cap on/off never needs a backfill pass
        self.portfolio_risk_cap_pct = portfolio_risk_cap_pct   # None = no portfolio-wide risk cap
        self.total_slippage_spread_cost = 0.0
        self.total_commission_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_carry = 0.0       # interest earned on equity minus the carry paid on open positions (CARRY_MODE)

    def accrue_carry(self, days: float, rate: float):
        """CARRY_MODE "futures", for `days` calendar days at the annual `rate`, on the account as it stood at the last close."""
        years = days / 365.0
        amount = self.sizing_equity() * rate * years
        for name, _, px, u in self._open_units():
            amount += u["shares"] * px * (CARRY_DIVIDEND_YIELD.get(name, 0.0) - rate) * years
        self.equity += amount
        self.total_carry += amount

    def mark_close(self, equity: float, date):
        """The account's value at a daily close: moves the running peak and the year-start reference along."""
        self._roll_year(date)
        if SHOCK_PAUSE_LOSS_PCT and self.last_equity > 0 and equity / self.last_equity - 1.0 <= -SHOCK_PAUSE_LOSS_PCT / 100.0:
            self.pause_until = date + pd.Timedelta(days=SHOCK_PAUSE_DAYS)
        if EQUITY_MA_WINDOW:
            self.eq_hist.append(equity)
            self.eq_sum += equity
            if len(self.eq_hist) > EQUITY_MA_WINDOW:
                self.eq_sum -= self.eq_hist.pop(0)
        self.peak = max(self.peak, equity)
        self.last_equity = equity

    def _roll_year(self, date):
        if date.year != self.year:
            self.year = date.year
            self.year_start = self.last_equity   # the account as it closed the previous year

    def drawdown_scale(self, equity: float, date) -> float:
        """Multiplier on new risk from the Turtles' account-adjustment rule: sizing as if the account were
        DRAWDOWN_SCALING_FACTOR smaller (of the reference) for every full DRAWDOWN_SCALING_THRESHOLD it is
        below the reference. 1.0 when the rule is off, or the account is at or above the reference."""
        if not USE_DRAWDOWN_SCALING:
            return 1.0
        self._roll_year(date)
        reference = self.peak if DRAWDOWN_SCALING_REFERENCE == "peak" else self.year_start
        if reference <= 0 or equity <= 0 or equity >= reference:
            return 1.0
        steps = int((reference - equity) / reference / DRAWDOWN_SCALING_THRESHOLD + 1e-9)
        # the notional equity never exceeds the real one: before the first full step there is no cut at all
        return min(1.0, max(0.0, 1.0 - steps * DRAWDOWN_SCALING_FACTOR) * reference / equity)

    def risk_overlay_scale(self) -> float:
        """Multiplier on new risk from the vol-target and hit-rate overlays (1.0 when both are off)."""
        scale = self.vol_scale if VOL_TARGET_PCT else 1.0
        if HITRATE_WINDOW and len(self.recent_r) >= HITRATE_WINDOW:
            if sum(self.recent_r[-HITRATE_WINDOW:]) / HITRATE_WINDOW < HITRATE_MIN_AVG_R:
                scale *= HITRATE_SCALE
        if EQUITY_MA_WINDOW and len(self.eq_hist) >= EQUITY_MA_WINDOW and self.last_equity < self.eq_sum / EQUITY_MA_WINDOW:
            scale *= EQUITY_MA_SCALE
        return scale

    def stress_room(self, asset, tail: float, equity: float):
        """Largest extra notional `asset` can take before a STRESS_* cap binds (None when no such cap is set; <= 0 when
        full). Each open unit's stress loss is its market value times its market's tail."""
        if STRESS_TOTAL_CAP_PCT is None and STRESS_SINGLE_CAP_PCT is None and STRESS_CLUSTER_CAP_PCT is None:
            return None
        total = own = cluster = 0.0
        group = MARKET_CLUSTERS.get(asset, asset)
        for name, ctx, px, u in self._open_units():
            loss = u["shares"] * px * _tail_at(ctx, ctx["cur_i"])
            total += loss
            if name == asset:
                own += loss
            if MARKET_CLUSTERS.get(name, name) == group:
                cluster += loss
        room = float("inf")
        if STRESS_TOTAL_CAP_PCT is not None:
            room = min(room, STRESS_TOTAL_CAP_PCT / 100.0 * equity - total)
        if STRESS_SINGLE_CAP_PCT is not None:
            room = min(room, STRESS_SINGLE_CAP_PCT / 100.0 * equity - own)
        if STRESS_CLUSTER_CAP_PCT is not None:
            room = min(room, STRESS_CLUSTER_CAP_PCT / 100.0 * equity - cluster)
        return room / tail

    def book_vol_room(self, asset, equity: float):
        """Largest extra notional `asset` can take before the open book's volatility (BOOK_VOL_WINDOW-day covariance of
        every market, so correlations count) would pass BOOK_VOL_CAP_PCT a year; None when the cap is off or there is
        not enough history yet. Solves  vol(book + f x asset)^2 = cap^2  for f."""
        if BOOK_VOL_CAP_PCT is None or self.ret_matrix is None or self.cur_row is None:
            return None
        row, col = self.cur_row, self.asset_col.get(asset)
        if col is None or row < BOOK_VOL_WINDOW:
            return None
        if self.cov_cache[0] != row:
            self.cov_cache = (row, np.cov(self.ret_matrix[row - BOOK_VOL_WINDOW + 1:row + 1], rowvar=False))
        cov = self.cov_cache[1]
        weights = np.zeros(cov.shape[0])
        for name, _, px, u in self._open_units():
            c = self.asset_col.get(name)
            if c is not None:
                weights[c] += u["shares"] * px / equity
        a, b = float(cov[col, col]), float(cov[col] @ weights)
        c = float(weights @ cov @ weights) - (BOOK_VOL_CAP_PCT / 100.0) ** 2 / 252.0
        if a <= 1e-12:
            return None
        if c >= 0:
            return 0.0
        return max((-b + np.sqrt(b * b - a * c)) / a, 0.0) * equity

    def gross_notional(self) -> float:
        if self.contexts is None:
            return 0.0
        return sum(u["shares"] * px for _, _, px, u in self._open_units())

    def note_skip(self, counter: str, key):
        """Counts a skipped or limited entry. The engine retries on every bar while a signal stays
        valid, so consecutive bars for the same asset and system count as ONE skip, not one per day."""
        if key is None:
            setattr(self, counter, getattr(self, counter) + 1)
            return
        asset, sys_name, iloc = key
        last = self.last_skip.get((counter, asset, sys_name))
        if last is None or iloc - last > 1:
            setattr(self, counter, getattr(self, counter) + 1)
        self.last_skip[(counter, asset, sys_name)] = iloc

    def portfolio_risk_room_left(self):
        if self.portfolio_risk_cap_pct is None:
            return None
        return (self.portfolio_risk_cap_pct / 100.0) * self.sizing_equity() - self.open_risk_total

    def _open_units(self):
        """(asset, ctx, price now, unit) for every open unit, marked at each asset's latest close."""
        for name, ctx in self.contexts.items():
            i = ctx.get("cur_i")
            if i is None:
                continue
            px = ctx["close"][i]
            for s in ctx["systems"].values():
                for u in s["units"]:
                    yield name, ctx, px, u

    def unrealized(self):
        if self.contexts is None:
            return 0.0
        total = 0.0
        next_open = EXECUTION_TIMING == "next_open"
        for ctx in self.contexts.values():   # inlined (no generator): this runs once per date
            i = ctx["cur_i"]
            if i is None:
                continue
            for s in ctx["systems"].values():
                units = s["units"]
                if units:
                    px = ctx["close"][i]
                    for u in units:
                        if next_open and u["entry_iloc"] == i:
                            continue   # filled at the next open: no profit or loss before it exists
                        total += u["shares"] * (px - u["entry_price"])
        return total

    def sizing_equity(self):
        """Account equity used for sizing, the exposure cap and drawdown scaling."""
        if self.contexts is None or not (MARK_TO_MARKET or USE_MARGIN_CALLS):
            return self.equity
        return self.equity + self.unrealized()

    def margin_required(self):
        """(initial, maintenance) margin of everything currently open."""
        init = maint = 0.0
        for name, _, px, u in self._open_units():
            rates = MARGIN_SPEC.get(name, DEFAULT_MARGIN_RATES)
            notional = u["shares"] * px
            init += notional * rates[0]
            maint += notional * rates[1]
        return init, maint


def _tail_needed() -> bool:
    return bool(GAP_SIZING_K) or STRESS_TOTAL_CAP_PCT is not None or STRESS_SINGLE_CAP_PCT is not None \
        or STRESS_CLUSTER_CAP_PCT is not None


def _tail_at(ctx: dict, i: int) -> float:
    """The market's tail (worst one-day drop, as a fraction) as of bar i: TAIL_FALLBACK while the history is too short,
    never below 2% so a very quiet stretch cannot make the caps unlimited."""
    v = ctx["tail"][i]
    return TAIL_FALLBACK if not np.isfinite(v) else max(float(v), 0.02)


def _build_asset_context(df: pd.DataFrame) -> dict:
    """Everything about ONE asset's simulation that does NOT involve equity -- price/feature
    arrays, per-system position state, and a date -> integer-position lookup (used by
    simulate_portfolio()'s date-driven outer loop; simulate_asset() just uses range(n_bars) and
    never touches it, but building it unconditionally here is cheap and keeps this one function
    the single source of truth for "what does one asset's simulation context look like").

    "dates" is a plain list of Timestamps, boxed once here: the per-bar loop reads ctx["dates"][i]
    every bar, and indexing a DatetimeIndex re-boxes a Timestamp on every access (about a quarter
    of a whole simulation's runtime, measured)."""
    dates = list(df.index)
    ctx = {
        "asset": df.attrs.get("asset"),
        "dates": dates,
        "open": df["Open"].values, "high": df["High"].values, "low": df["Low"].values,
        "close": df["Close"].values,
        "breakout_level": {s["name"]: df[f"breakout_level_{s['name']}"].values for s in SYSTEMS},
        "exit_low_level": {s["name"]: df[f"exit_low_level_{s['name']}"].values for s in SYSTEMS},
        "trailing_stop_low": {s["name"]: df[f"trailing_stop_low_{s['name']}"].values for s in SYSTEMS},
        "sys_n": {s["name"]: df[f"n_{s['name']}"].values for s in SYSTEMS},
        "sys_tp_n": {s["name"]: df[f"tp_n_{s['name']}"].values for s in SYSTEMS},
        "sys_adx": {s["name"]: df[f"adx_{s['name']}"].values for s in SYSTEMS},
        "sys_ema_trend": {s["name"]: df[f"ema_trend_{s['name']}"].values for s in SYSTEMS},
        "sys_choppiness": {s["name"]: df[f"choppiness_{s['name']}"].values for s in SYSTEMS},
        "sys_all_time_high": {s["name"]: df[f"all_time_high_{s['name']}"].values for s in SYSTEMS},
        "sys_pyramid_breakout": {s["name"]: df[f"pyramid_breakout_level_{s['name']}"].values for s in SYSTEMS},
        "sys_mh_add": {s["name"]: (df[f"mh_add_level_{s['name']}"].values if f"mh_add_level_{s['name']}" in df
                                   else np.full(len(df), np.nan)) for s in SYSTEMS},
        "sys_swing_high": {s["name"]: df[f"swing_high_level_{s['name']}"].values for s in SYSTEMS},
        "sys_params": {s["name"]: s for s in SYSTEMS},
        "date_to_iloc": {date: i for i, date in enumerate(dates)},
        "n_bars": len(df),
        "systems": {},
    }
    if _tail_needed():
        # the market's worst one-day drop over the last TAIL_WINDOW bars, up to and including each bar (NaN until
        # TAIL_MIN_BARS days exist; see _tail_at)
        daily = pd.Series(ctx["close"]).pct_change()
        ctx["tail"] = (-daily.rolling(int(TAIL_WINDOW), min_periods=int(min(TAIL_MIN_BARS, TAIL_WINDOW))).min()
                       ).clip(lower=0.0).to_numpy()
    if EXTENSION_MA:
        ctx["ext_ma"] = pd.Series(ctx["close"]).rolling(int(EXTENSION_MA), min_periods=int(EXTENSION_MA)).mean().to_numpy()
    if VOL_RISK_GAMMA:
        hi, lo, cl = pd.Series(ctx["high"]), pd.Series(ctx["low"]), pd.Series(ctx["close"])
        true_range = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(int(VOL_RISK_ATR), min_periods=int(VOL_RISK_ATR)).mean()
        ratio = atr / atr.shift(1).rolling(int(VOL_RISK_REF), min_periods=int(VOL_RISK_REF)).mean()
        ctx["vol_risk_mult"] = (ratio ** (-float(VOL_RISK_GAMMA))).clip(lower=VOL_RISK_MIN, upper=VOL_RISK_MAX).fillna(1.0).to_numpy()
    for sysdef in SYSTEMS:
        ctx["systems"][sysdef["name"]] = {
            "name": sysdef["name"], "units": [], "in_position": False,
            "current_stop": None, "last_add_price": None, "last_add_iloc": None,
            "last_swing_level_used": None,
            "last_loss_exit_iloc": None,
            "trailing_stop_level": None,
            "highest_high_since_entry": None,   # for the OPTIONAL Chandelier exit
            "peak_trail_peak": None, "peak_trail_level": None,   # for the OPTIONAL resting peak-trail stop
            "breakeven_triggered": False,        # has the OPTIONAL breakeven stop already fired this episode
            "first_entry_price": None, "first_entry_n_value": None,   # frozen at the episode's FIRST entry,
                                                                        # used as the breakeven stop's reference
            "episode_closed_units": [], "trades": [],
            "entry_markers": [], "exit_markers": [],
        }
    return ctx


def _other_systems_notional(systems: dict, exclude_name: str) -> float:
    return sum(u["capital_deployed"] for name, s in systems.items() if name != exclude_name for u in s["units"])


def asset_risk_multipliers(names, raw: dict = None, normalize: bool = None) -> dict:
    """{market: risk multiplier} for the markets in a portfolio run. raw/normalize default to the
    module's ASSET_RISK_MULT / ASSET_RISK_NORMALIZE; a market with no entry counts as 1.0. With
    normalize the values are divided by their average over `names`, so the average multiplier is
    exactly 1.0 and the mix only moves risk between markets."""
    raw = ASSET_RISK_MULT if raw is None else raw
    normalize = ASSET_RISK_NORMALIZE if normalize is None else normalize
    names = list(names)
    mults = {name: max(0.0, float(raw.get(name, 1.0))) for name in names}
    if normalize and mults:
        mean = sum(mults.values()) / len(mults)
        if mean > 0:
            mults = {name: value / mean for name, value in mults.items()}
    return mults


def _make_unit_functions(ctx: dict, equity_state: "_EquityState"):
    """Builds close_unit/open_unit/finalize_episode as closures bound to ONE asset's context and
    ONE equity_state -- private in simulate_asset() (one call, one asset), or called once PER
    ASSET but against the SAME shared equity_state in simulate_portfolio(). Every dollar amount
    (equity, slippage/commission totals, open risk) reads and writes through equity_state, never a
    local/nonlocal float, which is what makes sharing it across assets safe."""
    sys_params = ctx["sys_params"]
    sys_tp_n = ctx["sys_tp_n"]
    sys_adx = ctx["sys_adx"]
    asset = ctx.get("asset")

    def close_unit(u, raw_exit_price, exit_reason, date, shares_override=None):
        shares_to_close = u["shares"] if shares_override is None else shares_override
        fill = exit_fill_price(raw_exit_price, asset)
        slip_cost = shares_to_close * abs(raw_exit_price - fill)
        pnl = shares_to_close * (fill - u["entry_price"])
        comm = commission_for(asset, shares_to_close, shares_to_close * fill)
        # the entry commission was already taken from equity when the unit opened; the trade record
        # carries its share too, so a trade's P&L is what it really made after BOTH commissions
        entry_comm_share = (u.get("entry_commission", 0.0) * shares_to_close / u["initial_shares"]
                            if u["initial_shares"] > 0 else 0.0)
        equity_state.equity += pnl - comm
        equity_state.total_slippage_spread_cost += slip_cost
        equity_state.total_commission_cost += comm
        fraction = shares_to_close / u["initial_shares"] if u["initial_shares"] > 0 else 0.0
        closed_risk = u["risk_amount"] * fraction
        equity_state.open_risk_total -= closed_risk
        u_closed = dict(u)
        u_closed.update({
            "shares": shares_to_close, "exit_date": date, "exit_price": fill, "exit_reason": exit_reason,
            "pnl": pnl - comm - entry_comm_share, "exit_commission": comm,
            "risk_amount": closed_risk,
            "capital_deployed": shares_to_close * u["entry_price"],
        })
        return u_closed

    def open_unit(iloc, date, raw_signal_price, n_value, current_notional, sys_name, unit_number):
        signal_close = raw_signal_price
        if equity_state.pause_until is not None and date <= equity_state.pause_until:
            return None   # SHOCK_PAUSE_LOSS_PCT: the account just took a big one-day loss
        if ENTRY_TOP_K and "entry_rank" in ctx and (unit_number == 1 or not ENTRY_RANK_FIRST_UNIT_ONLY):
            rank_here, strength_here = ctx["entry_rank"][iloc], ctx["entry_strength"][iloc]
            if not (np.isfinite(rank_here) and rank_here <= ENTRY_TOP_K and strength_here > ENTRY_MIN_STRENGTH):
                equity_state.n_rank_blocked += 1
                return None   # ENTRY_TOP_K: not one of the strongest trends today
        if EXECUTION_TIMING == "next_open":
            # the signal was seen at this bar's close; the order fills at the next bar's open
            next_iloc = iloc + 1
            if next_iloc >= ctx["n_bars"] or not np.isfinite(ctx["open"][next_iloc]):
                return None
            raw_signal_price = ctx["open"][next_iloc]
        fill = entry_fill_price(raw_signal_price, asset)
        schedule = sys_params[sys_name]["risk_per_trade_pct_schedule"]
        if schedule:
            # unit_number is 1-indexed; the LAST value in the schedule covers every unit beyond
            # the list's length (e.g. [0.5, 1.0] -> unit 1 = 0.5%, unit 2+ = 1.0%)
            risk_pct_here = schedule[min(unit_number - 1, len(schedule) - 1)]
        else:
            risk_pct_here = sys_params[sys_name]["risk_per_trade_pct"]
        # a portfolio run puts the (optionally rescaled) per-market multiplier in the context; a
        # single-asset run has no other markets to rescale against and uses the raw value
        multiplier = ctx["risk_mult"] if "risk_mult" in ctx else ASSET_RISK_MULT.get(asset, 1.0)
        if (ATH_RISK_MULT != 1.0 or BELOW_ATH_RISK_MULT != 1.0) and (not ATH_FIRST_UNIT_ONLY or unit_number == 1):
            ath_here = ctx["sys_all_time_high"][sys_name][iloc]
            if np.isfinite(ath_here):
                multiplier *= ATH_RISK_MULT if signal_close > ath_here else BELOW_ATH_RISK_MULT
        if CHOP_RISK_MULT != 1.0:
            chop_here = ctx["sys_choppiness"][sys_name][iloc]
            if np.isfinite(chop_here) and chop_here <= CHOP_TREND_MAX:
                multiplier *= CHOP_RISK_MULT
        if EXTENSION_MA:
            ext_ma = ctx["ext_ma"][iloc]
            if np.isfinite(ext_ma) and signal_close - ext_ma > EXTENSION_N_MULT * n_value:
                multiplier *= EXTENSION_RISK_MULT   # a parabolic run: little or nothing more on it
        if VOL_RISK_GAMMA and "vol_risk_mult" in ctx:
            multiplier *= ctx["vol_risk_mult"][iloc]   # VOL_RISK_GAMMA: less risk while the market is louder than usual for it
        risk_pct_here *= multiplier
        account_equity = equity_state.sizing_equity()
        risk_pct_here *= equity_state.drawdown_scale(account_equity, date)
        if equity_state.contexts is not None:
            risk_pct_here *= equity_state.risk_overlay_scale()
        gross_room = None
        if MAX_GROSS_EXPOSURE_PCT and equity_state.contexts is not None:
            gross_room = MAX_GROSS_EXPOSURE_PCT / 100.0 * account_equity - equity_state.gross_notional()
        margin_room = None
        if USE_MARGIN_CALLS and equity_state.contexts is not None:
            init_rate = MARGIN_SPEC.get(asset, DEFAULT_MARGIN_RATES)[0]
            margin_room = (account_equity - equity_state.margin_required()[0]) / init_rate
        tail = _tail_at(ctx, iloc) if "tail" in ctx else None
        stress_room = None
        if equity_state.contexts is not None:
            if tail is not None:
                stress_room = equity_state.stress_room(asset, tail, account_equity)
            vol_room = equity_state.book_vol_room(asset, account_equity)
            if vol_room is not None:
                stress_room = vol_room if stress_room is None else min(stress_room, vol_room)
        sized = size_unit(account_equity, fill, n_value, sys_params[sys_name]["stop_n_mult"],
                            current_notional, risk_pct_here,
                            portfolio_risk_room_left=equity_state.portfolio_risk_room_left(),
                            step=SIZE_STEP.get(asset) if equity_state.contexts is not None else None,
                            take_min_one=TAKE_MIN_ONE_UNIT, margin_room_notional=margin_room,
                            counters=equity_state, skip_key=(asset, sys_name, iloc),
                            gross_room_notional=gross_room,
                            min_risk_distance=GAP_SIZING_K * tail * fill if GAP_SIZING_K and tail is not None else None,
                            stress_room_notional=stress_room)
        if sized is None:
            return None
        comm = commission_for(asset, sized["shares"], sized["capital_deployed"])
        slip_cost = sized["shares"] * abs(fill - raw_signal_price)
        equity_state.equity -= comm
        equity_state.total_slippage_spread_cost += slip_cost
        equity_state.total_commission_cost += comm
        equity_state.open_risk_total += sized["risk_amount"]
        return {"entry_iloc": iloc, "entry_date": date, "entry_price": fill,
                "shares": sized["shares"], "initial_shares": sized["shares"], "triggered_scale_levels": set(),
                "risk_amount": sized["risk_amount"], "equity_at_entry": equity_state.sizing_equity(),
                "capital_deployed": sized["capital_deployed"], "n_value": n_value, "entry_commission": comm,
                "stop_n_mult": sys_params[sys_name]["stop_n_mult"],
                # TP's OWN N (frozen at entry, using tp_atr_period) and its OWN multiplier --
                # fully independent of the stop's N/multiplier above
                "tp_n_value": sys_tp_n[sys_name][iloc], "tp_n_mult": sys_params[sys_name]["tp_n_mult"],
                "entry_adx": sys_adx[sys_name][iloc],
                "entry_day_of_week": date.day_name(), "entry_month": date.month, "entry_year": date.year,
                # tracked per-unit for the giveback stop -- this unit's OWN highest price since
                # ITS OWN entry, not the whole episode's (that's highest_high_since_entry, used by
                # Chandelier instead)
                "peak_price_since_entry": fill}

    def finalize_episode(sys_state, equity_after):
        all_units = sys_state["episode_closed_units"] + []
        # DEDUPLICATE by entry_iloc for anything meant to reflect "how many distinct times did
        # we add to this position." A single unit that gets partially scaled out (use_scale_out,
        # at multiple R levels) produces MULTIPLE entries in all_units -- one per partial-exit
        # event -- each an exact copy of the original unit's entry fields (close_unit does
        # dict(u), preserving entry_date/entry_price/entry_iloc unchanged; only the exit-side
        # fields differ). Naively using len(all_units) for n_units, or averaging entry_prices
        # over all_units, overcounts/miscounts whenever scale-outs happened -- confirmed directly
        # against a real stress test, where 252 of 310 "adds" reported this way were actually
        # scale-out fragments of an add that had already happened, not new adds at all.
        # unit_details keeps the FULL granular list (legitimately useful: every partial-exit
        # event with its own date/price/reason). pnl/risk_amount/capital_deployed sums stay
        # correctly additive over every partial-exit event -- only entry_dates/entry_prices/
        # n_units needed the fix, since those should count distinct adds, not exit events.
        seen_ilocs = {}
        for u in all_units:
            seen_ilocs.setdefault(u["entry_iloc"], u)
        distinct_units = sorted(seen_ilocs.values(), key=lambda u: u["entry_iloc"])

        total_pnl = sum(u["pnl"] for u in all_units)
        total_risk = sum(u["risk_amount"] for u in all_units)
        if HITRATE_WINDOW and total_risk:
            equity_state.recent_r.append(total_pnl / total_risk)
            del equity_state.recent_r[:-HITRATE_WINDOW]
        sys_state["trades"].append({
            "system": sys_state["name"],
            "entry_dates": [u["entry_date"] for u in distinct_units],
            "entry_prices": [u["entry_price"] for u in distinct_units],
            "unit_details": all_units, "n_units": len(distinct_units),
            "exit_date": all_units[-1]["exit_date"], "exit_price": all_units[-1]["exit_price"],
            "exit_reason": all_units[-1]["exit_reason"],
            "pnl": total_pnl, "equity_after": equity_after,
            "total_risk_amount": total_risk,
            # PnL expressed in units of risk taken -- matches turtle_soup.py's own r_multiple field
            # name/meaning, so episode_stats() can report "Avg R" consistently for both engines.
            "r_multiple": total_pnl / total_risk if total_risk else float("nan"),
            "total_capital_deployed": sum(u["capital_deployed"] for u in all_units),
        })

    return open_unit, close_unit, finalize_episode


def carry_rates_for(dates: pd.DatetimeIndex) -> np.ndarray:
    """The annual short-term rate in force on each date: CARRY_RATE_SERIES carried forward, CARRY_RATE_FLAT before it starts."""
    if CARRY_RATE_SERIES is None or len(CARRY_RATE_SERIES) == 0:
        return np.full(len(dates), float(CARRY_RATE_FLAT))
    s = CARRY_RATE_SERIES[~CARRY_RATE_SERIES.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates).fillna(float(CARRY_RATE_FLAT)).to_numpy(dtype=float)


def _apply_holding_cost(ctx: dict, equity_state: "_EquityState", date=None):
    asset = ctx.get("asset")
    calendar_days = 365.0 / 252.0
    if date is not None and ctx.get("cur_i") is not None:
        calendar_days = max((date - ctx["dates"][ctx["cur_i"]]).days, 1)
    if _asset_cost(asset, "holding_pct", HOLDING_COST_PCT_PER_YEAR) > 0:
        open_notional_today = sum(u["capital_deployed"] for s in ctx["systems"].values() for u in s["units"])
        if open_notional_today > 0:
            day_cost = daily_holding_cost(open_notional_today, asset, calendar_days)
            equity_state.equity -= day_cost
            equity_state.total_holding_cost += day_cost


def _process_systems_and_exposure(ctx: dict, equity_state: "_EquityState", i: int,
                                     open_unit, close_unit, finalize_episode):
    """Everything that happens for ONE asset on ONE bar, after that bar's holding cost has already
    been applied: stop/exit checks, pyramid-add/new-entry checks (per system), then the per-asset
    S1+S2 exposure-trim check. This is the exact per-bar body simulate_asset() always ran, now
    reusable by simulate_portfolio() too -- called once per asset per date instead of once per
    bar for a single asset, against whichever equity_state (private or shared) it's given."""
    systems = ctx["systems"]
    sys_params = ctx["sys_params"]
    open_, high, low, close = ctx["open"], ctx["high"], ctx["low"], ctx["close"]
    date = ctx["dates"][i]
    breakout_level, exit_low_level = ctx["breakout_level"], ctx["exit_low_level"]
    trailing_stop_low = ctx["trailing_stop_low"]
    sys_n, sys_tp_n, sys_adx = ctx["sys_n"], ctx["sys_tp_n"], ctx["sys_adx"]
    sys_ema_trend, sys_choppiness = ctx["sys_ema_trend"], ctx["sys_choppiness"]
    sys_all_time_high = ctx["sys_all_time_high"]
    sys_pyramid_breakout, sys_swing_high = ctx["sys_pyramid_breakout"], ctx["sys_swing_high"]
    sys_mh_add = ctx["sys_mh_add"]

    for sysdef in SYSTEMS:
        name = sysdef["name"]
        sys = systems[name]
        exit_lvl = exit_low_level[name]
        entry_lvl = breakout_level[name]

        if sys["in_position"]:
            units = sys["units"]

            if units and (sysdef["use_scale_out"] or sysdef["use_time_stop"] or sysdef["use_regime_scale_out"]
                           or sysdef["use_giveback_stop"]):
                remaining = []
                for u in units:
                    # Per-unit trailing profit give-back -- see the config comment on
                    # use_giveback_stop for the full reasoning. Tracked/checked BEFORE scale-out
                    # below so a unit that both qualifies for a scale-out AND has breached its own
                    # give-back level this same bar exits via give-back (a full close) rather than
                    # a partial -- the more protective of the two outcomes.
                    if sysdef["use_giveback_stop"] and u["shares"] > 1e-9:
                        u["peak_price_since_entry"] = max(u.get("peak_price_since_entry", u["entry_price"]), high[i])
                        risk_per_share = u["stop_n_mult"] * u["n_value"]
                        current_r = (close[i] - u["entry_price"]) / risk_per_share if risk_per_share > 0 else 0.0
                        if current_r >= sysdef["giveback_min_profit_r"]:
                            giveback_level = (u["entry_price"] + sysdef["giveback_fraction"]
                                                * (u["peak_price_since_entry"] - u["entry_price"]))
                            if low[i] <= giveback_level:
                                # same gap-through-stop realism as the shared SL/trailing/chandelier
                                # exits below -- can't fill better than the open once the market has
                                # already gapped past the level overnight
                                fill_price = open_[i] if open_[i] < giveback_level else giveback_level
                                u_closed = close_unit(u, fill_price, "giveback_stop", date)
                                sys["episode_closed_units"].append(u_closed)
                                sys["exit_markers"].append((i, u_closed["exit_price"], "giveback_stop"))
                                continue
                    if sysdef["use_scale_out"] and u["shares"] > 1e-9:
                        tp_distance = u["tp_n_mult"] * u["tp_n_value"]
                        for level_r, frac in sysdef["scale_out_levels"]:
                            if level_r in u["triggered_scale_levels"]:
                                continue
                            level_price = u["entry_price"] + level_r * tp_distance
                            if high[i] >= level_price:
                                partial_shares = _round_partial(
                                    ctx.get("asset"), min(u["initial_shares"] * frac, u["shares"]), u["shares"])
                                if partial_shares > 1e-9:
                                    # a resting sell order at the target fills at the open when the bar
                                    # gapped up through it, never below the bar's own range
                                    u_partial = close_unit(u, max(level_price, open_[i]), "scale_out", date,
                                                            shares_override=partial_shares)
                                    sys["episode_closed_units"].append(u_partial)
                                    sys["exit_markers"].append((i, u_partial["exit_price"], "scale_out"))
                                    u["shares"] -= partial_shares
                                    u["capital_deployed"] = u["shares"] * u["entry_price"]
                                    u["triggered_scale_levels"].add(level_r)
                    # regime-triggered scale-out: ADX spiking while ALREADY profitable predicts worse
                    # forward performance (confirmed across every asset in the regime diagnostic) --
                    # take an extra partial profit here, on top of the fixed R-multiple scale-out above.
                    # One-time per unit (won't keep firing every day ADX stays elevated).
                    if (sysdef["use_regime_scale_out"] and u["shares"] > 1e-9
                            and not u.get("regime_scale_out_triggered", False)
                            and close[i] > u["entry_price"]
                            and sys_adx[name][i] > sysdef["regime_scale_out_adx_threshold"]):
                        partial_shares = _round_partial(
                            ctx.get("asset"), min(u["initial_shares"] * sysdef["regime_scale_out_fraction"], u["shares"]),
                            u["shares"])
                        if partial_shares > 1e-9:
                            u_partial = close_unit(u, close[i], "regime_scale_out", date,
                                                    shares_override=partial_shares)
                            sys["episode_closed_units"].append(u_partial)
                            sys["exit_markers"].append((i, u_partial["exit_price"], "regime_scale_out"))
                            u["shares"] -= partial_shares
                            u["capital_deployed"] = u["shares"] * u["entry_price"]
                        u["regime_scale_out_triggered"] = True
                    if (sysdef["use_time_stop"] and u["shares"] > 1e-9
                            and (i - u["entry_iloc"]) >= sysdef["time_stop_days"]):
                        risk_per_share = u["stop_n_mult"] * u["n_value"]
                        current_r = (close[i] - u["entry_price"]) / risk_per_share if risk_per_share > 0 else 0.0
                        if current_r < sysdef["time_stop_min_r"]:
                            u_closed = close_unit(u, close[i], "time_stop", date)
                            sys["episode_closed_units"].append(u_closed)
                            sys["exit_markers"].append((i, u_closed["exit_price"], "time_stop"))
                            continue
                    if u["shares"] > 1e-9:
                        remaining.append(u)
                sys["units"] = units = remaining

            if not units:
                finalize_episode(sys, equity_state.equity)
                if sysdef["use_cooldown"] and sys["trades"][-1]["pnl"] < 0:
                    sys["last_loss_exit_iloc"] = i
                sys["episode_closed_units"] = []
                sys["in_position"] = False
                sys["last_add_price"] = None
                sys["last_add_iloc"] = None
                sys["last_swing_level_used"] = None
                sys["current_stop"] = None
                sys["trailing_stop_level"] = None
                sys["highest_high_since_entry"] = None
                sys["peak_trail_peak"] = sys["peak_trail_level"] = None
                sys["breakeven_triggered"] = False
                sys["first_entry_price"] = None
                sys["first_entry_n_value"] = None
            else:
                n_here = sys_n[name][i]

                if sysdef["use_n_day_low_trailing_stop"] and not np.isnan(trailing_stop_low[name][i]):
                    sys["trailing_stop_level"] = (trailing_stop_low[name][i] if sys["trailing_stop_level"] is None
                                                    else max(sys["trailing_stop_level"], trailing_stop_low[name][i]))

                # --- Chandelier exit: ATR-trailing stop from the highest HIGH since entry (not a
                # rolling N-day low -- a genuinely different, more standard trailing mechanism) ---
                chandelier_stopped = False
                if sysdef["use_chandelier_exit"]:
                    sys["highest_high_since_entry"] = (high[i] if sys["highest_high_since_entry"] is None
                                                         else max(sys["highest_high_since_entry"], high[i]))
                    if not np.isnan(n_here):
                        chandelier_level = sys["highest_high_since_entry"] - sysdef["chandelier_atr_mult"] * n_here
                        chandelier_stopped = low[i] <= chandelier_level

                # --- breakeven stop: once the position has proven itself by breakeven_trigger_r,
                # ratchet the stop up to (at least) the very first entry price -- caps how much of
                # a proven trade's cushion can give back before locking in no-loss ---
                if (sysdef["use_breakeven_stop"] and not sys["breakeven_triggered"]
                        and sys["first_entry_price"] is not None and sys["first_entry_n_value"] is not None):
                    risk_per_share = sysdef["stop_n_mult"] * sys["first_entry_n_value"]
                    trigger_price = sys["first_entry_price"] + sysdef["breakeven_trigger_r"] * risk_per_share
                    if high[i] >= trigger_price:
                        sys["current_stop"] = max(sys["current_stop"], sys["first_entry_price"])
                        sys["breakeven_triggered"] = True

                stopped = (sysdef["use_fixed_stop"] and sys["current_stop"] is not None
                           and low[i] <= sys["current_stop"])
                trailing_stopped = (sysdef["use_n_day_low_trailing_stop"] and sys["trailing_stop_level"] is not None
                                     and low[i] <= sys["trailing_stop_level"])
                peak_trail_stopped = bool(sysdef.get("use_peak_trail") and sys["peak_trail_level"] is not None
                                          and low[i] <= sys["peak_trail_level"])
                channel_exit = (not np.isnan(exit_lvl[i])) and close[i] < exit_lvl[i]
                daily_level, daily_stopped = None, False
                if (DAILY_LOSS_STOP_N_MULT or DAILY_LOSS_STOP_PCT) and i > 0:
                    candidates = []
                    if DAILY_LOSS_STOP_N_MULT and np.isfinite(sys_n[name][i - 1]):
                        candidates.append(close[i - 1] - DAILY_LOSS_STOP_N_MULT * sys_n[name][i - 1])
                    if DAILY_LOSS_STOP_PCT:
                        candidates.append(close[i - 1] * (1.0 - DAILY_LOSS_STOP_PCT / 100.0))
                    if candidates:
                        daily_level = max(candidates)
                        daily_stopped = low[i] <= daily_level
                if stopped or trailing_stopped or chandelier_stopped or peak_trail_stopped or daily_stopped or channel_exit:
                    if STOP_PRIORITY == "highest_level" and (stopped or trailing_stopped or chandelier_stopped or peak_trail_stopped
                                                             or daily_stopped):
                        # several stop levels can sit under today's low at once; a falling price
                        # reaches the HIGHEST of them first, and that is the one that fills
                        touched = []
                        if stopped:
                            touched.append((sys["current_stop"], "SL"))
                        if trailing_stopped:
                            touched.append((sys["trailing_stop_level"], "trailing_stop_low"))
                        if chandelier_stopped:
                            touched.append((sys["highest_high_since_entry"] - sysdef["chandelier_atr_mult"] * n_here,
                                            "chandelier_exit"))
                        if peak_trail_stopped:
                            touched.append((sys["peak_trail_level"], "peak_trail"))
                        if daily_stopped:
                            touched.append((daily_level, "daily_loss_stop"))
                        trigger_price, reason = max(touched, key=lambda level: level[0])
                    elif stopped:
                        trigger_price, reason = sys["current_stop"], "SL"
                    elif trailing_stopped:
                        trigger_price, reason = sys["trailing_stop_level"], "trailing_stop_low"
                    elif chandelier_stopped:
                        trigger_price, reason = (sys["highest_high_since_entry"] - sysdef["chandelier_atr_mult"] * n_here,
                                                   "chandelier_exit")
                    elif peak_trail_stopped:
                        trigger_price, reason = sys["peak_trail_level"], "peak_trail"
                    elif daily_stopped:
                        trigger_price, reason = daily_level, "daily_loss_stop"
                    else:
                        trigger_price, reason = close[i], "exit_channel"
                    # Gap-through-stop realism: all three stop-type exits are checked via
                    # low[i] <= trigger_price, which only proves the level was touched SOMEWHERE
                    # during the day -- not that it was the first price available. If the market
                    # OPENED already below the stop (an overnight gap, a crash), a real
                    # stop-market order fills at/near the OPEN, not at a stop level that was
                    # never actually tradeable. Only applies to the three stop-type reasons --
                    # exit_channel is already a close-based signal filled at that same close, so
                    # there's no gap to model (you can't gap through a level evaluated using the
                    # day's own close).
                    if reason != "exit_channel" and open_[i] < trigger_price:
                        raw_price = open_[i]
                    elif (reason == "exit_channel" and EXECUTION_TIMING == "next_open"
                          and i + 1 < ctx["n_bars"] and np.isfinite(open_[i + 1])):
                        raw_price = open_[i + 1]   # a close-based signal is acted on at the next open
                    else:
                        raw_price = trigger_price * (1.0 - STOP_SLIPPAGE_PCT / 100.0) if reason != "exit_channel" else trigger_price
                    for u in units:
                        u_closed = close_unit(u, raw_price, reason, date)
                        sys["episode_closed_units"].append(u_closed)
                    sys["exit_markers"].append((i, exit_fill_price(raw_price, ctx.get("asset")), reason))
                    finalize_episode(sys, equity_state.equity)
                    if sysdef["use_cooldown"] and sys["trades"][-1]["pnl"] < 0:
                        sys["last_loss_exit_iloc"] = i
                    sys["episode_closed_units"] = []
                    sys["units"] = []
                    sys["in_position"] = False
                    sys["last_add_price"] = None
                    sys["last_add_iloc"] = None
                    sys["last_swing_level_used"] = None
                    sys["current_stop"] = None
                    sys["trailing_stop_level"] = None
                    sys["highest_high_since_entry"] = None
                    sys["peak_trail_peak"] = sys["peak_trail_level"] = None
                    sys["breakeven_triggered"] = False
                    sys["first_entry_price"] = None
                    sys["first_entry_n_value"] = None
                else:
                    n_here = sys_n[name][i]
                    if sysdef.get("use_peak_trail") and not np.isnan(n_here) and sys["peak_trail_peak"] is not None:
                        # the stop for the NEXT bar, from everything known at this close; it never falls
                        sys["peak_trail_peak"] = max(sys["peak_trail_peak"], high[i])
                        level = sys["peak_trail_peak"] - sysdef["peak_trail_atr_mult"] * n_here
                        if sys["peak_trail_level"] is None or level > sys["peak_trail_level"]:
                            sys["peak_trail_level"] = level
                    if sysdef.get("use_monthly_high_entry") or sysdef["pyramid_method"] == "monthly_high":
                        # MONTHLY-HIGH adds: EVERY time a new entry would have happened. With the monthly-high entry on (or pyramid_method "monthly_high") the position adds a unit on every
                        # breakout signal of that entry: a monthly high that is min..max bars old breaks out for the first time and the close is above its EMA. No spacing rule; the signal
                        # array is +infinity on every other day, so only a real new breakout can add, and the max units per trend still cap it.
                        mh_add_here = sys_mh_add[name][i]
                        spacing_ok = (not np.isnan(mh_add_here)) and close[i] > mh_add_here
                    elif sysdef["pyramid_method"] == "breakout":
                        # "breakout" method: add whenever price makes a fresh N-day high
                        # (its own lookback, separate from the entry breakout) AND at least
                        # N days have passed since the last add. The fresh-high check ALONE
                        # doesn't slow down rapid-fire adds during an uninterrupted thrust --
                        # during a continuous climb out of a long base, EVERY day is trivially
                        # a fresh N-day high (there's no older peak left in the window to
                        # compare against), so a longer lookback alone never blocks quick,
                        # repeated adds within one sustained push (confirmed directly: an
                        # 8-day uninterrupted rally added on all 8 days even with N=55). The
                        # time gate is what actually enforces "no more than one add every N
                        # days," independent of whether today happens to be a fresh high.
                        pyramid_level_here = sys_pyramid_breakout[name][i]
                        fresh_high_ok = (not np.isnan(pyramid_level_here)) and (close[i] > pyramid_level_here)
                        days_since_last_add = i - sys["last_add_iloc"] if sys["last_add_iloc"] is not None else None
                        time_gate_ok = (days_since_last_add is None) or (
                            days_since_last_add >= sysdef["pyramid_breakout_lookback"])
                        spacing_ok = fresh_high_ok and time_gate_ok
                    elif sysdef["pyramid_method"] == "swing_high":
                        # "swing_high" method: add only when price closes above the most
                        # recently CONFIRMED swing high -- genuine market structure, not a
                        # rolling window. BUT: once price crosses one confirmed level, it can
                        # easily STAY above it for many days before the next swing high even
                        # confirms -- "close > swing_level_here" alone would then fire an add
                        # EVERY one of those days, not once (confirmed directly: 15 adds in
                        # 20 days on real synthetic data, before this second check was added).
                        # The fix: an add is only valid against a level GENUINELY NEWER than
                        # whichever level the last add already used -- once a level has been
                        # used, staying above it doesn't count again; only a fresh, higher
                        # confirmed swing high does. This is what actually forces one add per
                        # genuine pullback-and-new-pivot, not one add per day of "still above."
                        swing_level_here = sys_swing_high[name][i]
                        is_above_swing = (not np.isnan(swing_level_here)) and (close[i] > swing_level_here)
                        is_new_level = (sys["last_swing_level_used"] is None) or (
                            not np.isnan(swing_level_here) and swing_level_here > sys["last_swing_level_used"])
                        spacing_ok = is_above_swing and is_new_level
                    else:
                        # ORIGINAL/default method: add once price has moved far enough past
                        # the LAST add's price (spacing_n_mult * N)
                        spacing_ok = (not np.isnan(n_here)) and (
                            close[i] - sys["last_add_price"] >= sysdef["pyramid_spacing_n_mult"] * n_here)
                    # blocks the ADD (not the initial entry) if the move already looks
                    # over-extended by ADX -- doesn't touch spacing/max-units logic at all
                    pyramid_adx_ok = (not sysdef["use_pyramid_adx_filter"]) or (
                        sys_adx[name][i] <= sysdef["pyramid_max_adx"])
                    # require_all_time_high now ALSO gates pyramid adds, not just the initial
                    # entry -- the "spacing" method was mathematically guaranteed to always
                    # already be at a fresh all-time-high (proven and tested), but the
                    # "breakout" method uses a ROLLING window, so an earlier peak can roll
                    # off the lookback and let a later, LOWER local high count as a fresh
                    # breakout even though it's still below the true all-time high. Checking
                    # explicitly here closes that gap for both methods, rather than relying
                    # on a property that only actually held for one of them.
                    pyramid_ath_ok = ((not sysdef["require_all_time_high"]) or (
                        not np.isnan(sys_all_time_high[name][i]) and close[i] > sys_all_time_high[name][i])) and (
                        ATH_MAX_DISTANCE_PCT is None or (not np.isnan(sys_all_time_high[name][i]) and
                        close[i] >= sys_all_time_high[name][i] * (1.0 - ATH_MAX_DISTANCE_PCT / 100.0)))
                    if spacing_ok and pyramid_adx_ok and pyramid_ath_ok and len(units) < sysdef["max_units_per_trend"]:
                        current_notional = sum(u["capital_deployed"] for u in units) + _other_systems_notional(systems, name)
                        new_unit = open_unit(i, date, close[i], n_here, current_notional, name,
                                              unit_number=len(units) + 1)
                        if new_unit is not None:
                            units.append(new_unit)
                            sys["last_add_price"] = close[i]
                            sys["last_add_iloc"] = i
                            if sysdef["pyramid_method"] == "swing_high":
                                sys["last_swing_level_used"] = sys_swing_high[name][i]
                            if sysdef["ratchet_stop_on_pyramid"]:
                                if sysdef["use_average_entry_stop"]:
                                    # anchor to the share-weighted AVERAGE entry price across all
                                    # currently open units, not just the latest one -- the latest-
                                    # unit anchor tightens the stop with every add regardless of
                                    # how wide the pyramid spacing is, which can put the stop BELOW
                                    # the original entry even while every unit is still profitable.
                                    # Averaging is inherently gentler: it only tightens as fast as
                                    # the position's own average cost basis actually improves.
                                    total_shares = sum(u["shares"] for u in units)
                                    avg_entry = sum(u["shares"] * u["entry_price"] for u in units) / total_shares
                                    new_stop_anchor = avg_entry - sysdef["stop_n_mult"] * n_here
                                    sys["current_stop"] = max(sys["current_stop"], new_stop_anchor)
                                else:
                                    # unconditional SET, not a ratchet-up-only max(): every add moves
                                    # ALL open units' stop to stop_n_mult * N below THIS add's own entry
                                    # price/N, full stop. If N has widened since the last add, this CAN
                                    # move the stop down -- deliberate, since the point is "the stop
                                    # always sits stop_n_mult*N below the newest add," not "the stop
                                    # never loosens." use_average_entry_stop above is the gentler,
                                    # tighten-only alternative for anyone who wants that instead.
                                    new_stop = new_unit["entry_price"] - sysdef["stop_n_mult"] * n_here
                                    sys["current_stop"] = (max(sys["current_stop"], new_stop) if NEVER_LOOSEN_STOPS
                                                           else new_stop)
                            # if ratchet_stop_on_pyramid is False, current_stop is left exactly
                            # as the first entry set it -- adding units doesn't tighten it at all.
                            # (use_breakeven_stop remains a fully separate, independently
                            # controlled ratchet -- this toggle only affects the pyramid-driven one)
                            sys["entry_markers"].append((i, new_unit["entry_price"]))
        else:
            n_here = sys_n[name][i]
            buffer = (sysdef["entry_buffer_n_mult"] * n_here
                      if not np.isnan(n_here) and not (
                          sysdef.get("use_monthly_high_entry") or sysdef.get("use_ma_regime") or sysdef.get("use_ema_entry")
                          or (ENTRY_EMA_PERIOD and ema_entry_settings(name, sysdef))) else 0)      # the EMA / monthly-high levels carry their own buffer logic
            raw_signal = (not np.isnan(entry_lvl[i])) and close[i] > entry_lvl[i] + buffer
            ema_here = sys_ema_trend[name][i]
            ema_ok = (not sysdef["use_ema_filter"]) or (not np.isnan(ema_here) and close[i] > ema_here)
            choppiness_here = sys_choppiness[name][i]
            choppiness_ok = (not sysdef["use_choppiness_filter"]) or (
                not np.isnan(choppiness_here) and choppiness_here <= sysdef["choppiness_max"])
            ath_here = sys_all_time_high[name][i]   # highest HIGH ever seen (see build_features)
            ath_ok = ((not sysdef["require_all_time_high"]) or (
                not np.isnan(ath_here) and close[i] > ath_here)) and (   # trigger stays CLOSE-based
                ATH_MAX_DISTANCE_PCT is None or (not np.isnan(ath_here) and
                                                 close[i] >= ath_here * (1.0 - ATH_MAX_DISTANCE_PCT / 100.0)))
            cooldown_ok = (not sysdef["use_cooldown"]) or (sys["last_loss_exit_iloc"] is None) or (
                (i - sys["last_loss_exit_iloc"]) >= sysdef["cooldown_days"])
            would_enter = (raw_signal and ema_ok and choppiness_ok and ath_ok and cooldown_ok
                           and not np.isnan(n_here) and n_here > 0)

            if would_enter:
                current_notional = _other_systems_notional(systems, name)
                new_unit = open_unit(i, date, close[i], n_here, current_notional, name, unit_number=1)
                if new_unit is not None:
                    sys["units"] = [new_unit]
                    sys["last_add_price"] = new_unit["entry_price"]
                    sys["last_add_iloc"] = i
                    sys["current_stop"] = new_unit["entry_price"] - sysdef["stop_n_mult"] * n_here
                    if sysdef["use_n_day_low_trailing_stop"] and not np.isnan(trailing_stop_low[name][i]):
                        sys["trailing_stop_level"] = trailing_stop_low[name][i]
                    sys["first_entry_price"] = new_unit["entry_price"]
                    sys["first_entry_n_value"] = n_here
                    sys["highest_high_since_entry"] = high[i]
                    if sysdef.get("use_peak_trail"):
                        sys["peak_trail_peak"] = new_unit["entry_price"]
                        sys["peak_trail_level"] = new_unit["entry_price"] - sysdef["peak_trail_atr_mult"] * n_here
                    sys["in_position"] = True
                    sys["entry_markers"].append((i, new_unit["entry_price"]))

    all_open = [(name, u) for name, s in systems.items() for u in s["units"]]
    total_notional_now = sum(u["capital_deployed"] for _, u in all_open)
    cap_notional_now = (MAX_TOTAL_EXPOSURE_PCT / 100.0) * (equity_state.sizing_equity() if all_open
                                                           else equity_state.equity)
    if total_notional_now > cap_notional_now > 0 and all_open:
        trim_fraction = (total_notional_now - cap_notional_now) / total_notional_now
        for name, u in all_open:
            shares_to_trim = _round_partial(ctx.get("asset"), u["shares"] * trim_fraction, u["shares"], up=True)
            if shares_to_trim <= 1e-9:
                continue
            u_trimmed = close_unit(u, close[i], "exposure_trim", date, shares_override=shares_to_trim)
            systems[name]["episode_closed_units"].append(u_trimmed)
            systems[name]["exit_markers"].append((i, u_trimmed["exit_price"], "exposure_trim"))
            u["shares"] -= shares_to_trim
            u["capital_deployed"] = u["shares"] * u["entry_price"]
        for name, sys in systems.items():
            sys["units"] = [u for u in sys["units"] if u["shares"] > 1e-9]
            if sys["in_position"] and not sys["units"]:
                finalize_episode(sys, equity_state.equity)
                if sys_params[name]["use_cooldown"] and sys["trades"][-1]["pnl"] < 0:
                    sys["last_loss_exit_iloc"] = i
                sys["episode_closed_units"] = []
                sys["in_position"] = False
                sys["last_add_price"] = None
                sys["last_add_iloc"] = None
                sys["last_swing_level_used"] = None
                sys["current_stop"] = None
                sys["trailing_stop_level"] = None
                sys["highest_high_since_entry"] = None
                sys["peak_trail_peak"] = sys["peak_trail_level"] = None
                sys["breakeven_triggered"] = False
                sys["first_entry_price"] = None
                sys["first_entry_n_value"] = None


def _finalize_open_positions(ctx: dict, equity_state: "_EquityState", close_unit, finalize_episode):
    close = ctx["close"]
    systems = ctx["systems"]
    n_bars = ctx["n_bars"]
    exit_date = ctx["dates"][-1]
    for sys in systems.values():
        if sys["in_position"] and sys["units"]:
            for u in sys["units"]:
                u_closed = close_unit(u, close[-1], "end_of_data", exit_date)
                sys["episode_closed_units"].append(u_closed)
            sys["exit_markers"].append((n_bars - 1, sys["episode_closed_units"][-1]["exit_price"], "end_of_data"))
            finalize_episode(sys, equity_state.equity)
            sys["episode_closed_units"] = []
            sys["units"] = []
            sys["in_position"] = False


def _build_asset_result(ctx: dict, equity_curve: pd.Series, equity_state: "_EquityState") -> dict:
    systems = ctx["systems"]
    all_trades = sorted(
        [t for sys in systems.values() for t in sys["trades"]],
        key=lambda t: t["entry_dates"][0])
    all_entry_markers = {name: s["entry_markers"] for name, s in systems.items()}
    all_exit_markers = {name: s["exit_markers"] for name, s in systems.items()}
    return {
        "trades": all_trades, "trades_by_system": {name: s["trades"] for name, s in systems.items()},
        "equity_curve": equity_curve,
        "entry_markers_by_system": all_entry_markers, "exit_markers_by_system": all_exit_markers,
        "final_equity": equity_curve.iloc[-1],
        "still_open": any(s["in_position"] for s in systems.values()),
        "open_units_by_system": {name: s["units"] for name, s in systems.items()},
        "total_slippage_spread_cost": equity_state.total_slippage_spread_cost,
        "total_commission_cost": equity_state.total_commission_cost,
        "total_holding_cost": equity_state.total_holding_cost,
    }


def simulate_asset(df: pd.DataFrame, starting_capital: float = STARTING_CAPITAL_PER_ASSET) -> dict:
    ctx = _build_asset_context(df)
    equity_state = _EquityState(starting_capital)
    open_unit, close_unit, finalize_episode = _make_unit_functions(ctx, equity_state)

    n_bars = ctx["n_bars"]
    equity_curve = np.full(n_bars, np.nan)
    # Idle-bar skip (exact, not an approximation): a flat system can only enter when
    # close > breakout_level + buffer, and buffer >= 0, so close > breakout_level is NECESSARY for
    # any entry (NaN levels compare False). On a bar where that holds for neither system AND no
    # system holds a position or units, the whole bar body provably changes nothing -- no holding
    # cost (no open notional), no stops, no entries, nothing to trim -- so only the unchanged
    # equity needs recording. Most bars of a trend system are like this.
    close_arr = ctx["close"]
    can_enter_any = np.zeros(n_bars, dtype=bool)
    for sysdef in SYSTEMS:
        can_enter_any |= close_arr > ctx["breakout_level"][sysdef["name"]]
    system_states = list(ctx["systems"].values())
    for i in range(n_bars):
        if not can_enter_any[i] and not any(s["in_position"] or s["units"] for s in system_states):
            equity_curve[i] = equity_state.equity
            continue
        _apply_holding_cost(ctx, equity_state)
        # Snapshotted AFTER this bar's own entries/exits/stops are processed, not before -- so
        # equity_curve[i] is bar i's true END-of-day equity (the standard convention: "today's
        # equity" includes today's own trading), not bar i-1's. Recording it before used to make
        # EVERY bar's value actually belong to the PREVIOUS bar's date; the only exception was the
        # very last bar, which got force-corrected to the true final value right after the loop --
        # meaning the returns series' final data point used to silently compress more than one
        # day's P&L into what performance_stats() treats as a single day's return (and, for
        # walk_forward_turtle.py, the same artifact landed on whichever window's test_end happened
        # to be the overall dataset's last date). Snapshotting here removes the need for that
        # special case entirely -- every bar, including the last, is now genuinely consistent.
        _process_systems_and_exposure(ctx, equity_state, i, open_unit, close_unit, finalize_episode)
        equity_curve[i] = equity_state.equity
        equity_state.mark_close(equity_state.equity, ctx["dates"][i])

    _finalize_open_positions(ctx, equity_state, close_unit, finalize_episode)
    equity_curve[-1] = equity_state.equity

    return _build_asset_result(ctx, pd.Series(equity_curve, index=df.index), equity_state)


def _record_day(es: "_EquityState", date, history: list, position_rows: list):
    """One row of account state at this date's close, plus one row per asset with open units."""
    per_asset = {}
    next_open = EXECUTION_TIMING == "next_open"
    for name, ctx, px, u in es._open_units():
        row = per_asset.setdefault(name, [0.0, 0.0, px])   # shares, cost, price
        row[0] += u["shares"]
        # a unit filled at the next open has no profit or loss yet on its signal bar (see unrealized())
        row[1] += u["shares"] * (px if next_open and u["entry_iloc"] == ctx["cur_i"] else u["entry_price"])
    unrealized = notional = margin_init = margin_maint = 0.0
    for name, (shares, cost, px) in per_asset.items():
        rates = MARGIN_SPEC.get(name, DEFAULT_MARGIN_RATES) if MARGIN_SPEC else (0.0, 0.0)   # none modelled
        value = shares * px
        pnl = value - cost
        unrealized += pnl
        notional += value
        margin_init += value * rates[0]
        margin_maint += value * rates[1]
        position_rows.append((date, name, shares, cost / shares, px, pnl, value, value * rates[0]))
    history.append((date, es.equity, unrealized, es.equity + unrealized, margin_init, margin_maint, len(per_asset),
                    notional, es.total_slippage_spread_cost, es.total_commission_cost, es.total_holding_cost))


def _history_frame(history: list):
    if not history:
        return None
    return pd.DataFrame(history, columns=[
        "date", "realized_balance", "unrealized", "equity", "margin_initial", "margin_maintenance",
        "open_positions", "gross_notional", "cum_slippage_spread", "cum_commission", "cum_holding"]).set_index("date")


def _positions_frame(rows: list):
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["date", "asset", "shares", "avg_entry", "price", "unrealized", "notional",
                                       "margin"])


def _enforce_margin(contexts: dict, unit_fns: dict, equity_state: "_EquityState", date):
    """Daily-close margin check. If account equity has fallen below the total maintenance margin,
    the broker liquidates whole units -- the one using the most margin first -- at that day's close
    (with the usual exit slippage) until equity covers the requirement again. Checked on closes only,
    not intraday lows, and a cure always means closing, never depositing more cash."""
    account = equity_state.equity + equity_state.unrealized()
    init_req, maint_req = equity_state.margin_required()
    if account > 0:
        equity_state.max_margin_use = max(equity_state.max_margin_use, init_req / account)
    if account >= maint_req:
        return
    event = {"date": date, "equity": account, "required": maint_req, "units_closed": 0}
    while True:
        biggest = None
        for name, ctx, px, u in equity_state._open_units():
            margin_used = u["shares"] * px * MARGIN_SPEC.get(name, DEFAULT_MARGIN_RATES)[1]
            if biggest is None or margin_used > biggest[0]:
                biggest = (margin_used, name, ctx, u)
        if biggest is None:
            break
        _, name, ctx, u = biggest
        i = ctx["cur_i"]
        _, close_unit, finalize_episode = unit_fns[name]
        for sys_name, sys in ctx["systems"].items():
            if any(unit is u for unit in sys["units"]):
                break
        u_closed = close_unit(u, ctx["close"][i], "margin_call", date)
        sys["episode_closed_units"].append(u_closed)
        sys["exit_markers"].append((i, u_closed["exit_price"], "margin_call"))
        sys["units"] = [unit for unit in sys["units"] if unit is not u]
        event["units_closed"] += 1
        if not sys["units"]:
            finalize_episode(sys, equity_state.equity)
            if ctx["sys_params"][sys_name]["use_cooldown"] and sys["trades"][-1]["pnl"] < 0:
                sys["last_loss_exit_iloc"] = i
            sys["episode_closed_units"] = []
            sys["in_position"] = False
            for key in ("last_add_price", "last_add_iloc", "last_swing_level_used", "current_stop",
                        "trailing_stop_level", "highest_high_since_entry", "first_entry_price",
                        "first_entry_n_value", "peak_trail_peak", "peak_trail_level"):
                sys[key] = None
            sys["breakeven_triggered"] = False
        account = equity_state.equity + equity_state.unrealized()
        _, maint_req = equity_state.margin_required()
        if account >= maint_req:
            break
    if equity_state.equity <= 0:
        # gapped through zero: the account is wiped out (a real broker would also chase the debit)
        equity_state.equity = 0.0
        equity_state.wiped_out = True
    equity_state.margin_events.append(event)


def simulate_portfolio(features: dict, starting_capital: float = None, progress_cb=None) -> dict:
    """Steps every asset in `features` (name -> already build_features()'d DataFrame) forward
    TOGETHER against ONE shared equity curve, instead of each asset getting simulated in total
    isolation with its own private starting capital the way simulate_asset() does. This is what
    actually answers "how much could the account really lose if several correlated positions get
    hit at once" -- something no amount of per-asset backtesting can show, since simulate_asset()
    never lets one asset's trades affect another's available capital or risk budget at all.

    starting_capital: None (default) = STARTING_CAPITAL_PER_ASSET * len(features), unless the
    module-level PORTFOLIO_STARTING_CAPITAL is set, which takes precedence.

    Risk control, both applied against the SAME shared equity/open-risk accounting:
      - MAX_TOTAL_EXPOSURE_PCT (module-level, unchanged): still caps S1+S2's combined NOTIONAL on
        ONE asset -- now relative to shared portfolio equity instead of a private per-asset pot.
      - USE_PORTFOLIO_RISK_CAP / MAX_PORTFOLIO_RISK_PCT (module-level): caps the sum of every
        OPEN position's risk_amount across EVERY asset and both systems at MAX_PORTFOLIO_RISK_PCT%
        of portfolio equity -- the actual "how bad could a correlated crash get" answer. New
        entries/adds that would breach it get sized down or rejected via size_unit()'s
        portfolio_risk_room_left.
      - USE_DRAWDOWN_SCALING (module-level): the Turtles' account adjustment, reacting to the shared
        portfolio equity's own drawdown, not any single asset's -- see _EquityState's docstring.

    Assets are processed in `features`' own iteration order each day -- the deterministic
    tie-break whenever more than one asset wants to enter/add on the same date and the remaining
    risk budget can't fit all of them; earlier-ordered assets get first claim that day, exactly
    like a single trader placing orders in a fixed sequence would.

    Returns {"equity_curve": pd.Series, "starting_capital": float, "trades_by_asset": {name: [...]},
    "trades_by_system": {name: {sys_name: [...]}}, "entry_markers_by_asset": {name: {sys_name: [...]}},
    "exit_markers_by_asset": {...}, "final_equity": float, "still_open_by_asset": {name: bool},
    "total_slippage_spread_cost"/"total_commission_cost"/"total_holding_cost": float (portfolio-wide)}.
    """
    if starting_capital is None:
        starting_capital = (PORTFOLIO_STARTING_CAPITAL if PORTFOLIO_STARTING_CAPITAL is not None
                              else STARTING_CAPITAL_PER_ASSET * len(features))
    risk_cap = MAX_PORTFOLIO_RISK_PCT if USE_PORTFOLIO_RISK_CAP else None
    equity_state = _EquityState(starting_capital, portfolio_risk_cap_pct=risk_cap)

    contexts = {name: _build_asset_context(df) for name, df in features.items()}
    risk_mults = asset_risk_multipliers(contexts)
    for name, ctx in contexts.items():
        ctx["asset"] = name
        ctx["risk_mult"] = risk_mults[name]
        ctx["cur_i"] = None   # index of the asset's latest processed bar -- where its open units are marked
    equity_state.contexts = contexts
    unit_fns = {name: _make_unit_functions(ctx, equity_state) for name, ctx in contexts.items()}
    mark_open_pnl = MARK_TO_MARKET or USE_MARGIN_CALLS
    history, position_rows = [], []
    vol_history = []
    asset_order = list(contexts)

    all_dates = pd.DatetimeIndex(sorted(set().union(*(ctx["dates"] for ctx in contexts.values()))))
    equity_curve = pd.Series(index=all_dates, dtype=float)
    carry_on = CARRY_MODE == "futures"
    if carry_on:
        carry_rates = carry_rates_for(all_dates)
        carry_days = np.diff(all_dates.values).astype("timedelta64[D]").astype(float)
    book_rows = None
    if BOOK_VOL_CAP_PCT is not None:
        # daily returns of every market on one calendar: the days most of the live markets traded (a weekend bar of the
        # crypto market alone would otherwise pad the others with zero-return days), prices carried over a holiday
        closes = pd.DataFrame({name: pd.Series(ctx["close"], index=ctx["dates"]) for name, ctx in contexts.items()})
        closes = closes[~closes.index.duplicated(keep="last")].reindex(all_dates)
        traded, started = closes.notna().sum(axis=1), closes.ffill().notna().sum(axis=1)
        calendar = closes.ffill()[(started > 0) & (traded >= 0.5 * started)]
        equity_state.ret_matrix = calendar.pct_change().fillna(0.0).to_numpy(dtype=float)
        equity_state.asset_col = {name: k for k, name in enumerate(calendar.columns)}
        book_rows = np.searchsorted(calendar.index.values, all_dates.values, side="right") - 1

    if ENTRY_TOP_K:
        # each market's trend strength and its rank among all markets, every day, from closes up to that day only
        aligned = pd.DataFrame({name: pd.Series(ctx["close"], index=ctx["dates"]) for name, ctx in contexts.items()})
        aligned = aligned[~aligned.index.duplicated(keep="last")].reindex(all_dates).ffill()
        log_close = np.log(aligned.where(aligned > 0))
        span = int(ENTRY_STRENGTH_LOOKBACK)
        vol = log_close.diff().rolling(span, min_periods=span).std()
        strength = ((log_close - log_close.shift(span)) / (vol * np.sqrt(span))).replace([np.inf, -np.inf], np.nan)
        rank = strength.rank(axis=1, ascending=False, method="first")
        for name, ctx in contexts.items():
            ctx["entry_strength"] = strength[name].reindex(ctx["dates"]).to_numpy(dtype=float)
            ctx["entry_rank"] = rank[name].reindex(ctx["dates"]).to_numpy(dtype=float)

    progress_step = max(1, len(all_dates) // 100)
    for date_idx, date in enumerate(all_dates):
        if progress_cb is not None and date_idx % progress_step == 0:
            progress_cb(date_idx / len(all_dates))   # fraction done, about every 1% of the dates
        if book_rows is not None:
            equity_state.cur_row = int(book_rows[date_idx]) if book_rows[date_idx] >= 0 else None
        if carry_on and date_idx > 0:
            # the interval since the previous date, on the account as it closed that day (before today's trades)
            equity_state.accrue_carry(carry_days[date_idx - 1], carry_rates[date_idx])
        for name, ctx in contexts.items():
            if ctx["date_to_iloc"].get(date) is not None:
                _apply_holding_cost(ctx, equity_state, date)
        # Snapshotted AFTER this date's own entries/exits/stops across every asset -- same fix,
        # same reasoning, as simulate_asset()'s own loop above (see its comment for the full
        # explanation of why "before" silently compressed more than one day's P&L into what looked
        # like a single day's return at the end of the curve).
        if ASSET_PRIORITY == "rotate" and len(asset_order) > 1:
            shift = date_idx % len(asset_order)
            todays_order = asset_order[shift:] + asset_order[:shift]
        else:
            todays_order = asset_order
        for name in todays_order:
            ctx = contexts[name]
            i = ctx["date_to_iloc"].get(date)
            if i is not None:
                ctx["cur_i"] = i
                open_unit, close_unit, finalize_episode = unit_fns[name]
                _process_systems_and_exposure(ctx, equity_state, i, open_unit, close_unit, finalize_episode)
        if USE_MARGIN_CALLS:
            _enforce_margin(contexts, unit_fns, equity_state, date)
        account_equity = equity_state.equity + equity_state.unrealized() if mark_open_pnl else equity_state.equity
        equity_curve.loc[date] = account_equity
        equity_state.mark_close(account_equity, date)
        if VOL_TARGET_PCT:
            vol_history.append(account_equity)
            if len(vol_history) > VOL_TARGET_WINDOW + 1:
                del vol_history[0]
            if len(vol_history) >= 41:
                values = np.asarray(vol_history, dtype=float)
                returns_ = values[1:] / values[:-1] - 1.0
                realized = float(np.std(returns_, ddof=1) * np.sqrt(252))
                equity_state.vol_scale = (min(max(VOL_TARGET_PCT / 100.0 / realized, VOL_TARGET_FLOOR), VOL_TARGET_CAP)
                                          if realized > 0 else VOL_TARGET_CAP)
        if RECORD_HISTORY:
            _record_day(equity_state, date, history, position_rows)

    for name, ctx in contexts.items():
        open_unit, close_unit, finalize_episode = unit_fns[name]
        _finalize_open_positions(ctx, equity_state, close_unit, finalize_episode)
    equity_curve.iloc[-1] = equity_state.equity

    return {
        "equity_curve": equity_curve,
        "starting_capital": starting_capital,
        "trades_by_asset": {name: sorted([t for sys in ctx["systems"].values() for t in sys["trades"]],
                                           key=lambda t: t["entry_dates"][0])
                              for name, ctx in contexts.items()},
        "trades_by_system": {name: {sn: s["trades"] for sn, s in ctx["systems"].items()}
                               for name, ctx in contexts.items()},
        "entry_markers_by_asset": {name: {sn: s["entry_markers"] for sn, s in ctx["systems"].items()}
                                     for name, ctx in contexts.items()},
        "exit_markers_by_asset": {name: {sn: s["exit_markers"] for sn, s in ctx["systems"].items()}
                                    for name, ctx in contexts.items()},
        "final_equity": equity_curve.iloc[-1],
        "still_open_by_asset": {name: any(s["in_position"] for s in ctx["systems"].values())
                                  for name, ctx in contexts.items()},
        "total_slippage_spread_cost": equity_state.total_slippage_spread_cost,
        "total_commission_cost": equity_state.total_commission_cost,
        "total_holding_cost": equity_state.total_holding_cost,
        "total_carry": equity_state.total_carry,
        "margin_modelled": bool(MARGIN_SPEC), "history": _history_frame(history), "positions_history": _positions_frame(position_rows),
        "mark_to_market": mark_open_pnl,
        "margin_calls": equity_state.margin_events,
        "n_rounding_skips": equity_state.n_rounding_skips,
        "n_margin_blocked": equity_state.n_margin_blocked, "n_gross_blocked": equity_state.n_gross_blocked,
        "n_stress_blocked": equity_state.n_stress_blocked, "n_rank_blocked": equity_state.n_rank_blocked,
        "max_margin_use": equity_state.max_margin_use,
        "wiped_out": equity_state.wiped_out,
    }


def compute_buy_and_hold(df: pd.DataFrame, starting_capital: float = STARTING_CAPITAL_PER_ASSET) -> pd.Series:
    close = df["Close"]
    shares = starting_capital / close.iloc[0]
    return shares * close


def performance_stats(equity_curve: pd.Series, starting_capital: float, freq: int = 252) -> dict:
    """freq (default 252, a 5-day trading week) is only a FALLBACK for a curve too short (<2
    points) to measure its own density -- otherwise periods-per-year is inferred directly from
    the curve's own date span. A hardcoded 252 is correct for an ordinary stock/futures calendar,
    but WRONG for anything that trades every calendar day (confirmed directly: yfinance's BTC-USD
    data has rows on Saturdays/Sundays too, ~365 bars/year, not ~252) and wrong for any "combined
    portfolio" curve built by reindexing several assets onto their shared UNION of dates (e.g.
    turtle_ui.py's compute_combined_equity) -- including even ONE weekend-trading asset inflates
    the WHOLE combined curve's bar density past 252/year. Treating that inflated bar count as if
    it were still 252/year UNDERSTATES CAGR/Calmar (dividing the same total return by an inflated
    elapsed-years estimate) and UNDERSTATES Ann. Vol/Sharpe/Sortino (annualizing by sqrt(252) when
    the true bar density is higher) for both cases above."""
    equity = equity_curve.dropna()
    returns = equity.pct_change().fillna(0)

    if len(equity) >= 2:
        elapsed_days = (equity.index[-1] - equity.index[0]).days
        n_years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
        periods_per_year = (len(equity) - 1) / n_years if n_years and n_years > 0 else freq
    else:
        n_years, periods_per_year = np.nan, freq

    cagr = (equity.iloc[-1] / starting_capital) ** (1 / n_years) - 1 if n_years and n_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan

    # downside deviation: root-mean-square of the negative returns over ALL periods (zeros included),
    # measured from zero -- not the standard deviation of the losing days alone, which is smaller
    # and made Sortino look better than it is
    negative = np.minimum(returns.values, 0.0)
    downside_vol = np.sqrt((negative ** 2).mean()) * np.sqrt(periods_per_year) if (negative < 0).any() else np.nan
    sortino = (returns.mean() * periods_per_year) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    underwater = drawdown < 0
    if underwater.any():
        # calendar days from the last equity peak to the latest underwater date, longest such stretch
        last_peak = pd.Series(equity.index.where(~underwater.values), index=equity.index).ffill()
        stretch = (pd.Series(equity.index, index=equity.index) - last_peak).dt.days
        max_dd_duration = float(stretch[underwater].max())
    else:
        max_dd_duration = 0

    monthly_equity = equity.resample("ME").last().dropna()
    monthly_prev = monthly_equity.shift(1)
    if len(monthly_prev) > 0:
        monthly_prev.iloc[0] = starting_capital
    monthly_returns = (monthly_equity / monthly_prev - 1) if len(monthly_prev) > 0 else pd.Series(dtype=float)
    best_month = monthly_returns.max() if len(monthly_returns) > 0 else np.nan
    worst_month = monthly_returns.min() if len(monthly_returns) > 0 else np.nan

    return {
        "CAGR": cagr, "Ann. Vol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
        "Max Drawdown": max_dd, "Max DD Duration (days)": max_dd_duration, "Calmar": calmar,
        "Best Month": best_month, "Worst Month": worst_month,
        "Final Equity": equity.iloc[-1], "Total Return": equity.iloc[-1] / starting_capital - 1,
    }


def _max_consecutive(pnls: np.ndarray, want_wins: bool) -> int:
    """Longest streak of consecutive winning (or losing) episodes, in EPISODE ORDER as given --
    caller must pass pnls already sorted by exit_date for this to mean 'in a row over time'."""
    best = cur = 0
    for pnl in pnls:
        is_match = (pnl > 0) if want_wins else (pnl <= 0)
        cur = cur + 1 if is_match else 0
        best = max(best, cur)
    return best


def episode_stats(trades: list, open_units: list = None) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "Episodes": 0, "Win Rate": np.nan, "Avg Units/Episode": np.nan,
            "Avg Holding (days)": np.nan, "Avg PnL": np.nan, "Avg R": np.nan,
            "Best Episode": np.nan, "Worst Episode": np.nan, "Profit Factor": np.nan,
            "Max Consecutive Wins": 0, "Max Consecutive Losses": 0,
            "Units Exited via SL": 0, "Units Exited via Channel": 0,
            "Units Exited via Scale-Out": 0, "Units Exited via Regime Scale-Out": 0,
            "Units Exited via Time-Stop": 0, "Units Exited via Trailing Stop": 0,
            "Units Exited via Chandelier": 0, "Units Exited via Giveback": 0,
            "Units Exited via Exposure Trim": 0, "Units Exited EOD": 0,
            "Currently Open (units)": len(open_units) if open_units else 0,
        }
    trades_by_exit = sorted(trades, key=lambda t: t["exit_date"])
    pnls = np.array([t["pnl"] for t in trades_by_exit])
    r_multiples = np.array([t.get("r_multiple", np.nan) for t in trades_by_exit], dtype=float)
    units_per_ep = np.array([t["n_units"] for t in trades_by_exit])
    holding_days = np.array([(t["exit_date"] - min(t["entry_dates"])).days for t in trades_by_exit])
    all_unit_exits = [u["exit_reason"] for t in trades_by_exit for u in t["unit_details"]]

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    return {
        "Episodes": n, "Win Rate": win_rate,
        "Units Exited via SL": all_unit_exits.count("SL"),
        "Units Exited via Channel": all_unit_exits.count("exit_channel"),
        "Units Exited via Scale-Out": all_unit_exits.count("scale_out"),
        "Units Exited via Regime Scale-Out": all_unit_exits.count("regime_scale_out"),
        "Units Exited via Time-Stop": all_unit_exits.count("time_stop"),
        "Units Exited via Trailing Stop": all_unit_exits.count("trailing_stop_low"),
        "Units Exited via Chandelier": all_unit_exits.count("chandelier_exit"),
        "Units Exited via Giveback": all_unit_exits.count("giveback_stop"),
        "Units Exited via Exposure Trim": all_unit_exits.count("exposure_trim"),
        "Units Exited EOD": all_unit_exits.count("end_of_data"),
        "Avg Units/Episode": units_per_ep.mean(), "Max Units in One Episode": units_per_ep.max(),
        "Avg Holding (days)": holding_days.mean(),
        "Avg PnL": pnls.mean(), "Avg R": np.nanmean(r_multiples) if np.isfinite(r_multiples).any() else np.nan,
        "Best Episode": pnls.max(), "Worst Episode": pnls.min(),
        "Profit Factor": profit_factor,
        "Max Consecutive Wins": _max_consecutive(pnls, want_wins=True),
        "Max Consecutive Losses": _max_consecutive(pnls, want_wins=False),
        "Currently Open (units)": len(open_units) if open_units else 0,
    }


def print_stats_table(stats_dict: dict):
    df = pd.DataFrame(stats_dict).T
    for c in ["CAGR", "Ann. Vol", "Max Drawdown", "Total Return", "Best Month", "Worst Month"]:
        if c in df.columns:
            df[c] = (df[c].astype(float) * 100).round(2).astype(str) + "%"
    for c in ["Sharpe", "Sortino", "Calmar"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)
    if "Max DD Duration (days)" in df.columns:
        df["Max DD Duration (days)"] = df["Max DD Duration (days)"].astype(float).round(0).astype(int)
    if "Final Equity" in df.columns:
        df["Final Equity"] = df["Final Equity"].astype(float).map(lambda v: f"${v:,.0f}")
    print(df.to_string())


def print_episode_stats_table(stats_dict: dict):
    df = pd.DataFrame(stats_dict).T
    if "Win Rate" in df.columns:
        df["Win Rate"] = (df["Win Rate"].astype(float) * 100).round(1).astype(str) + "%"
    for c in ["Avg Units/Episode", "Avg Holding (days)", "Profit Factor"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)
    for c in ["Avg PnL", "Best Episode", "Worst Episode"]:
        if c in df.columns:
            df[c] = df[c].astype(float).map(lambda v: f"${v:,.0f}" if pd.notna(v) else "n/a")
    print(df.to_string())


def print_trade_log(name: str, trades: list, max_rows: int = 30):
    if not trades:
        print(f"  {name}: no closed trades")
        return
    tdf = pd.DataFrame([{
        "system": t["system"], "first_entry": min(t["entry_dates"]).date(), "n_units": t["n_units"],
        "avg_entry": round(np.mean(t["entry_prices"]), 2), "exit_date": t["exit_date"].date(),
        "exit_price": round(t["exit_price"], 2), "exit_reason": t.get("exit_reason", "?"),
        "pnl": round(t["pnl"], 2), "equity_after": round(t["equity_after"], 2),
    } for t in trades])
    print(f"\n  {name} (showing first {min(max_rows, len(tdf))} of {len(tdf)} episodes):")
    print(tdf.head(max_rows).to_string(index=False))


def print_pyramid_detail(name: str, trades: list, max_episodes: int = 3):
    if not trades:
        print(f"  {name}: no closed episodes")
        return
    print(f"\n  {name} (showing up to {max_episodes} episodes):")
    for ep_num, t in enumerate(trades[:max_episodes], start=1):
        print(f"    Episode {ep_num} [{t['system']}] ({t['entry_dates'][0].date()} -> {t['exit_date'].date()}, "
              f"exit: {t['exit_reason']}):")
        rows = []
        for u_num, u in enumerate(t["unit_details"], start=1):
            pct_equity = u["capital_deployed"] / u["equity_at_entry"] * 100
            rows.append({
                "unit": u_num, "date": u["entry_date"].date(), "entry_price": round(u["entry_price"], 2),
                "N_at_entry": round(u["n_value"], 3),
                "equity_at_entry": f"${u['equity_at_entry']:,.0f}", "risk_$": f"${u['risk_amount']:,.0f}",
                "shares": round(u["shares"], 4), "capital_deployed_$": f"${u['capital_deployed']:,.0f}",
                "% of equity": f"{pct_equity:.1f}%",
                "exit_date": u["exit_date"].date(), "exit_price": round(u["exit_price"], 2),
                "exit_via": u["exit_reason"], "unit_pnl": round(u["pnl"], 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))


def collect_all_trades(results_by_asset: dict) -> list:
    enriched = []
    for asset, result in results_by_asset.items():
        for t in result["trades"]:
            r_multiple = t["pnl"] / t["total_risk_amount"] if t["total_risk_amount"] > 0 else np.nan
            if not np.isfinite(r_multiple):
                continue
            enriched.append({"asset": asset, "trade": t, "win": t["pnl"] > 0, "r_multiple": r_multiple})
    return enriched


def select_representative_trades(all_trades: list, n: int = 5, candidate_pool: int = 20,
                                    seed: int = None) -> dict:
    rng = random.Random(seed)
    winners = [t for t in all_trades if t["win"]]
    losers = [t for t in all_trades if not t["win"]]

    def pick_near_average(group):
        if not group:
            return []
        avg_r = np.mean([t["r_multiple"] for t in group])
        sorted_by_closeness = sorted(group, key=lambda t: abs(t["r_multiple"] - avg_r))
        pool = sorted_by_closeness[:min(candidate_pool, len(sorted_by_closeness))]
        return rng.sample(pool, min(n, len(pool)))

    return {"winners": pick_near_average(winners), "losers": pick_near_average(losers)}


def reconstruct_stop_history(trade: dict) -> list:
    units = sorted(trade["unit_details"], key=lambda u: u["entry_date"])
    segments = []
    running_stop = None
    for i, u in enumerate(units):
        candidate = u["entry_price"] - u.get("stop_n_mult", 2.0) * u["n_value"]
        running_stop = candidate if running_stop is None else max(running_stop, candidate)
        seg_start = u["entry_date"]
        seg_end = units[i + 1]["entry_date"] if i + 1 < len(units) else trade["exit_date"]
        segments.append((seg_start, seg_end, running_stop))
    return segments


def plot_trade_closeup(df: pd.DataFrame, entry_asset: str, trade: dict, filename: str, padding_bars: int = 15):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    first_entry = min(trade["entry_dates"])
    window_start_iloc = max(0, df.index.get_indexer([first_entry], method="nearest")[0] - padding_bars)
    window_end_iloc = min(len(df) - 1, df.index.get_indexer([trade["exit_date"]], method="nearest")[0] + padding_bars)
    plot_df = df.iloc[window_start_iloc:window_end_iloc + 1]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = mdates.date2num(plot_df.index.to_pydatetime())
    width = (x[1] - x[0]) * 0.6 if len(x) > 1 else 0.5

    for xi, (_, row) in zip(x, plot_df.iterrows()):
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([xi, xi], [row["Low"], row["High"]], color=color, linewidth=0.9, zorder=2)
        ax.add_patch(plt.Rectangle(
            (xi - width / 2, min(row["Open"], row["Close"])), width,
            max(abs(row["Close"] - row["Open"]), 1e-6),
            color=color, alpha=0.85, zorder=3))

    sorted_units = sorted(trade["unit_details"], key=lambda uu: uu["entry_date"])
    MAX_LABELED_ENTRIES = 8
    for i, u in enumerate(sorted_units, start=1):
        ax.scatter(u["entry_date"], u["entry_price"], marker="^", color="blue", s=90, zorder=5,
                   edgecolors="black", linewidths=0.7, alpha=0.85)
        if i <= MAX_LABELED_ENTRIES:
            ax.annotate(f"{i}", (u["entry_date"], u["entry_price"]), textcoords="offset points",
                        xytext=(4, 6), fontsize=7, color="blue")
    if len(sorted_units) > MAX_LABELED_ENTRIES:
        ax.scatter([], [], marker="^", color="blue", s=90, edgecolors="black", linewidths=0.7,
                   label=f"Entries (1-{MAX_LABELED_ENTRIES} labeled, {len(sorted_units)} total)")
    else:
        ax.scatter([], [], marker="^", color="blue", s=90, edgecolors="black", linewidths=0.7, label="Entries")

    for seg_start, seg_end, stop_level in reconstruct_stop_history(trade):
        ax.plot([seg_start, seg_end], [stop_level, stop_level], color="darkred", linewidth=1.6,
                linestyle="--", zorder=4)
    ax.plot([], [], color="darkred", linewidth=1.6, linestyle="--", label="Stop-loss (ratcheting)")

    exit_color = "darkgreen" if trade["pnl"] > 0 else "darkred"
    ax.scatter(trade["exit_date"], trade["exit_price"], marker="v" if trade["pnl"] <= 0 else "o",
               color=exit_color, s=160, zorder=5, edgecolors="black", linewidths=1,
               label=f"Exit ({trade['exit_reason']})")

    r_multiple = trade["pnl"] / trade["total_risk_amount"] if trade["total_risk_amount"] > 0 else float("nan")
    outcome = "WINNER" if trade["pnl"] > 0 else "LOSER"
    # entry_dates is already deduplicated at the source (finalize_episode) -- no set() needed
    # here anymore. This used to work around exactly the same duplication bug fixed there,
    # just locally and only for this one display, while n_units right below was still wrong.
    entry_dates_sorted = sorted(trade["entry_dates"])
    if len(entry_dates_sorted) <= 3:
        entry_summary = ", ".join(d.date().isoformat() for d in entry_dates_sorted)
    else:
        entry_summary = (f"{entry_dates_sorted[0].date().isoformat()} -> "
                          f"{entry_dates_sorted[-1].date().isoformat()} ({trade['n_units']} units)")
    info_lines = [
        f"{entry_asset}  |  System {trade['system']}  |  {outcome}",
        f"Entry: {entry_summary}",
        f"Exit: {trade['exit_date'].date().isoformat()}  ({trade['exit_reason']})",
        f"Units (pyramided): {trade['n_units']}",
        f"Risked: ${trade['total_risk_amount']:,.0f}",
        f"PnL: ${trade['pnl']:,.0f}",
        f"R-multiple: {r_multiple:.2f}R",
    ]
    ax.text(0.02, 0.97, "\n".join(info_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_title(f"{entry_asset}: {outcome} episode close-up -- {r_multiple:.2f}R")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def build_episode_dataframe(results_by_asset: dict) -> pd.DataFrame:
    rows = []
    for asset, result in results_by_asset.items():
        for t in result["trades"]:
            first_unit = t["unit_details"][0]
            is_win = t["pnl"] > 0
            r_multiple = t["pnl"] / t["total_risk_amount"] if t["total_risk_amount"] > 0 else np.nan
            holding_days = (t["exit_date"] - min(t["entry_dates"])).days
            rows.append({
                "asset": asset, "system": t["system"],
                "entry_date": min(t["entry_dates"]).date(), "exit_date": t["exit_date"].date(),
                "holding_days": holding_days, "n_units": t["n_units"],
                "exit_reason": t["exit_reason"],
                "win": is_win, "pnl": round(t["pnl"], 2), "r_multiple": round(r_multiple, 3),
                "entry_price": round(first_unit["entry_price"], 4),
                "N_at_entry": round(first_unit["n_value"], 4),
                "entry_adx": round(first_unit["entry_adx"], 1) if pd.notna(first_unit["entry_adx"]) else np.nan,
                "entry_day_of_week": first_unit["entry_day_of_week"],
                "entry_month": first_unit["entry_month"], "entry_year": first_unit["entry_year"],
            })
    return pd.DataFrame(rows)


def print_most_common_entry_conditions(episode_df: pd.DataFrame):
    if episode_df.empty:
        print("  No closed episodes to report on.")
        return

    adx_bins = [0, 20, 25, 30, 40, 100]
    adx_labels = ["<20 (weak/no trend)", "20-25", "25-30", "30-40", ">40 (very strong trend)"]

    df = episode_df.copy()
    df["adx_bucket"] = pd.cut(df["entry_adx"], bins=adx_bins, labels=adx_labels)

    def most_common(series):
        counts = series.value_counts()
        if len(counts) == 0 or counts.sum() == 0:
            return "n/a", 0, 0.0
        top_label = counts.idxmax()
        top_count = counts.max()
        top_pct = top_count / counts.sum() * 100
        return top_label, top_count, top_pct

    for sys_name in sorted(df["system"].unique()):
        print(f"\n  --- System {sys_name} ---")
        for outcome_label, mask in [("Winners", df["system"].eq(sys_name) & df["win"]),
                                      ("Losers", df["system"].eq(sys_name) & ~df["win"])]:
            sub = df[mask]
            n_sub = len(sub)
            if n_sub == 0:
                print(f"    {outcome_label}: n/a (no episodes)")
                continue
            adx_label, adx_count, adx_pct = most_common(sub["adx_bucket"])
            print(f"    {outcome_label} (n={n_sub}):")
            print(f"      Most common ADX at entry:  {adx_label:<24} ({adx_count}/{n_sub} = {adx_pct:.0f}%)"
                  f"   | median ADX: {sub['entry_adx'].median():.1f}")
            print(f"      Median N at entry: {sub['N_at_entry'].median():.4f}")


def print_episode_circumstance_stats(episode_df: pd.DataFrame):
    if episode_df.empty:
        print("  No closed episodes to report on.")
        return

    n = len(episode_df)
    n_win = int(episode_df["win"].sum())
    n_loss = n - n_win
    win_rate = n_win / n * 100

    print(f"\n  OVERALL: {n} episodes -- {n_win} winning ({win_rate:.1f}%), {n_loss} losing ({100-win_rate:.1f}%)")

    r_win = episode_df.loc[episode_df["win"], "r_multiple"]
    r_loss = episode_df.loc[~episode_df["win"], "r_multiple"]
    avg_r_all = episode_df["r_multiple"].mean()
    profit_factor = (r_win.sum() / abs(r_loss.sum())) if len(r_loss) > 0 and r_loss.sum() != 0 else np.nan
    print(f"  R-MULTIPLE (reward:risk of each episode, realized PnL / initial risk taken):")
    print(f"    Avg R (all episodes):   {avg_r_all:.2f}R   <- expectancy per trade")
    print(f"    Avg R (winners only):   {r_win.mean():.2f}R" if len(r_win) else "    Avg R (winners only):   n/a")
    print(f"    Avg R (losers only):    {r_loss.mean():.2f}R" if len(r_loss) else "    Avg R (losers only):    n/a")
    print(f"    Best episode:           {episode_df['r_multiple'].max():.2f}R")
    print(f"    Worst episode:          {episode_df['r_multiple'].min():.2f}R")
    print(f"    Profit factor:          {profit_factor:.2f}")

    print(f"\n  N (ATR) AT ENTRY:")
    print(f"    Mean: {episode_df['N_at_entry'].mean():.3f}, Median: {episode_df['N_at_entry'].median():.3f}, "
          f"Min: {episode_df['N_at_entry'].min():.3f}, Max: {episode_df['N_at_entry'].max():.3f}")
    print(f"    Winners' avg N: {episode_df.loc[episode_df['win'],'N_at_entry'].mean():.3f}  |  "
          f"Losers' avg N: {episode_df.loc[~episode_df['win'],'N_at_entry'].mean():.3f}")

    print("\n  " + "=" * 74)
    print("  BY SYSTEM -- winning vs. losing exits, kept fully separate per system")
    print("  (S1 = faster/shorter-term entry, S2 = slower/longer-term entry)")
    print("  " + "=" * 74)
    for sys_name in sorted(episode_df["system"].unique()):
        sub = episode_df[episode_df["system"] == sys_name]
        n_sub = len(sub)
        n_sub_win = int(sub["win"].sum())
        n_sub_loss = n_sub - n_sub_win
        sub_win_rate = n_sub_win / n_sub * 100 if n_sub > 0 else np.nan
        r_sub_win = sub.loc[sub["win"], "r_multiple"]
        r_sub_loss = sub.loc[~sub["win"], "r_multiple"]
        sub_profit_factor = (r_sub_win.sum() / abs(r_sub_loss.sum())
                              if len(r_sub_loss) > 0 and r_sub_loss.sum() != 0 else np.nan)

        print(f"\n  --- System {sys_name}: {n_sub} episodes -- {n_sub_win} winning ({sub_win_rate:.1f}%), "
              f"{n_sub_loss} losing ({100-sub_win_rate:.1f}%) ---")
        print(f"    Avg R (all {sys_name} episodes):   {sub['r_multiple'].mean():.2f}R")
        print(f"    Avg R (winners only):          {r_sub_win.mean():.2f}R" if len(r_sub_win)
              else "    Avg R (winners only):          n/a (no winners)")
        print(f"    Avg R (losers only):           {r_sub_loss.mean():.2f}R" if len(r_sub_loss)
              else "    Avg R (losers only):           n/a (no losers)")
        print(f"    Best {sys_name} episode:            {sub['r_multiple'].max():.2f}R")
        print(f"    Worst {sys_name} episode:           {sub['r_multiple'].min():.2f}R")
        print(f"    Profit factor:                 {sub_profit_factor:.2f}")
        print(f"    Avg holding days (winners):    {sub.loc[sub['win'],'holding_days'].mean():.1f}" if len(r_sub_win)
              else "    Avg holding days (winners):    n/a")
        print(f"    Avg holding days (losers):     {sub.loc[~sub['win'],'holding_days'].mean():.1f}" if len(r_sub_loss)
              else "    Avg holding days (losers):     n/a")
        print(f"    Avg units pyramided (winners): {sub.loc[sub['win'],'n_units'].mean():.2f}" if len(r_sub_win)
              else "    Avg units pyramided (winners): n/a")
        print(f"    Avg units pyramided (losers):  {sub.loc[~sub['win'],'n_units'].mean():.2f}" if len(r_sub_loss)
              else "    Avg units pyramided (losers):  n/a")

        print(f"\n    Exit reasons for System {sys_name} winners:")
        win_exit_counts = sub.loc[sub["win"], "exit_reason"].value_counts()
        if len(win_exit_counts):
            print("      " + ", ".join(f"{reason}: {count}" for reason, count in win_exit_counts.items()))
        else:
            print("      n/a (no winners)")
        print(f"    Exit reasons for System {sys_name} losers:")
        loss_exit_counts = sub.loc[~sub["win"], "exit_reason"].value_counts()
        if len(loss_exit_counts):
            print("      " + ", ".join(f"{reason}: {count}" for reason, count in loss_exit_counts.items()))
        else:
            print("      n/a (no losers)")

    print("\n  --- Side-by-side summary ---")
    by_system = episode_df.groupby("system").agg(
        episodes=("win", "count"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"),
        avg_holding_days=("holding_days", "mean"), avg_units=("n_units", "mean"),
    )
    by_system["win_rate"] = (by_system["win_rate"] * 100).round(1).astype(str) + "%"
    by_system["avg_r"] = by_system["avg_r"].round(2)
    by_system["avg_holding_days"] = by_system["avg_holding_days"].round(1)
    by_system["avg_units"] = by_system["avg_units"].round(2)
    print(by_system.to_string())

    print("\n  BY EXIT REASON:")
    by_exit = episode_df.groupby("exit_reason").agg(
        episodes=("win", "count"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"))
    by_exit["win_rate"] = (by_exit["win_rate"] * 100).round(1).astype(str) + "%"
    by_exit["avg_r"] = by_exit["avg_r"].round(2)
    print(by_exit.to_string())

    print("\n  BY DAY OF WEEK ENTERED:")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow = episode_df.groupby("entry_day_of_week").agg(
        episodes=("win", "count"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"))
    by_dow["win_rate"] = (by_dow["win_rate"] * 100).round(1).astype(str) + "%"
    by_dow["avg_r"] = by_dow["avg_r"].round(2)
    by_dow = by_dow.reindex([d for d in dow_order if d in by_dow.index])
    print(by_dow.to_string())

    print("\n  BY MONTH ENTERED (seasonality):")
    by_month = episode_df.groupby("entry_month").agg(
        episodes=("win", "count"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"))
    by_month["win_rate"] = (by_month["win_rate"] * 100).round(1).astype(str) + "%"
    by_month["avg_r"] = by_month["avg_r"].round(2)
    print(by_month.to_string())


def compute_period_returns(equity_curve: pd.Series, rule: str, starting_capital: float) -> pd.DataFrame:
    period_equity = equity_curve.resample(rule).last().dropna()
    prev_equity = period_equity.shift(1)
    prev_equity.iloc[0] = starting_capital
    dollar_return = period_equity - prev_equity
    pct_return = dollar_return / prev_equity
    return pd.DataFrame({"equity": period_equity, "return_$": dollar_return, "return_%": pct_return})


def print_yearly_table(name: str, equity_curve: pd.Series, starting_capital: float):
    yearly = compute_period_returns(equity_curve, "YE", starting_capital)
    view = pd.DataFrame({
        "Year": yearly.index.year,
        "End Equity": yearly["equity"].map(lambda v: f"${v:,.0f}"),
        "Return $": yearly["return_$"].map(lambda v: f"${v:,.0f}"),
        "Return %": (yearly["return_%"] * 100).round(2).astype(str) + "%",
    })
    print(f"\n  {name}:")
    print(view.to_string(index=False))


# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
def plot_asset(name: str, df: pd.DataFrame, result: dict, filename: str, bh_equity: pd.Series = None):
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df.index, df["Close"], color="black", linewidth=0.8, label="Close")
    colors = {"S1": ("green", "darkgreen"), "S2": ("blue", "darkblue")}
    for sysdef in SYSTEMS:
        sname = sysdef["name"]
        entry_color, exit_color = colors.get(sname, ("green", "red"))
        markers = result["entry_markers_by_system"].get(sname, [])
        if markers:
            idx, prices = zip(*markers)
            ax1.scatter(df.index[list(idx)], prices, marker="^", color=entry_color, s=45, zorder=5,
                       label=f"{sname} Buy/Add")
        exit_markers = result["exit_markers_by_system"].get(sname, [])
        if exit_markers:
            eidx, eprices, ereasons = zip(*exit_markers)
            ax1.scatter(df.index[list(eidx)], eprices, marker="v", color=exit_color, s=45, zorder=5,
                       label=f"{sname} Exit")

    ax1.set_title(f"{name}: Dual-System Turtle Breakout Strategy")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(result["equity_curve"].index, result["equity_curve"].values, color="black", linewidth=1, label="Strategy")
    if bh_equity is not None:
        ax2.plot(bh_equity.index, bh_equity.values, color="gray", linewidth=1, linestyle="--", label="Buy & Hold")
    ax2.set_title("Equity Curve")
    ax2.set_yscale("log")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Saved chart to: {filepath}")


def plot_all_episodes_interactive(name: str, df: pd.DataFrame, result: dict, filename: str,
                                    bh_equity: pd.Series = None):
    """
    Real, zoomable candlesticks covering EVERY episode on this asset, PLUS
    each system's own EMA (overlaid on price), RSI, ADX, and ATR/N -- each
    in its own panel, and every single one of them independently toggleable
    via the legend (click a legend entry to show/hide just that line),
    exactly like the existing entry/exit/stop-line categories.

    Layout (top to bottom, all panels share the same time axis and zoom
    together):
      1. Price candlesticks + S1/S2 EMA overlay + entries/exits/stop-lines
      2. RSI (S1 and S2, each with its own period) with 30/70 reference lines
      3. ADX (S1 and S2, each with its own period) with a 25 reference line
      4. ATR / N (S1 and S2, each with its own period) -- this is literally
         the same volatility measure each system sizes/stops off of
      5. Equity curve (strategy vs. buy & hold)

    SCROLLING: opens already zoomed to an episode-scale window (~4 months),
    not the full history. Click-and-drag PANS freely on both axes at the
    current zoom level (dragmode="pan"). Scroll-wheel/trackpad zooms in and
    out smoothly. No range slider (conflicts with pan-by-drag). Double-click
    resets a panel back to autoscale.

    Requires: pip install plotly --break-system-packages
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed -- skipping interactive episode chart. "
              "Run: pip install plotly --break-system-packages")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.48, 0.16, 0.16, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{name}: every episode (drag to pan, scroll to zoom, double-click to reset)",
            "ADX", "ATR / N", "Equity Curve",
        ),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="green", decreasing_line_color="red",
        showlegend=False,
    ), row=1, col=1)

    system_colors = {"S1": "blue", "S2": "orange"}
    for sysdef in SYSTEMS:
        sname = sysdef["name"]
        entry_x, entry_y, entry_text = [], [], []
        for t in result["trades_by_system"].get(sname, []):
            for u_num, u in enumerate(sorted(t["unit_details"], key=lambda uu: uu["entry_date"]), start=1):
                entry_x.append(u["entry_date"])
                entry_y.append(u["entry_price"])
                entry_text.append(
                    f"{name} | System {sname} | Unit {u_num}/{t['n_units']}<br>"
                    f"Date: {u['entry_date'].date()}<br>Price: {u['entry_price']:.2f}<br>"
                    f"N (ATR) at entry: {u['n_value']:.3f}<br>Risk: ${u['risk_amount']:,.0f}<br>"
                    f"ADX: {u['entry_adx']:.1f}"
                )
        if entry_x:
            fig.add_trace(go.Scattergl(
                x=entry_x, y=entry_y, mode="markers", name=f"{sname} Entries",
                marker=dict(symbol="triangle-up", color=system_colors.get(sname, "blue"), size=9,
                            line=dict(width=1, color="black")),
                text=entry_text, hovertemplate="%{text}<extra></extra>",
                legendgroup=f"entries_{sname}",
            ), row=1, col=1)

    # exit markers, kept SEPARATE per system AND per outcome (4 independently toggleable
    # categories: S1 Win, S1 Loss, S2 Win, S2 Loss) -- click any legend entry to isolate it
    exit_win_color = {"S1": "darkgreen", "S2": "teal"}
    exit_loss_color = {"S1": "red", "S2": "darkorange"}
    exit_data = {sysdef["name"]: {"win": {"x": [], "y": [], "text": []},
                                    "loss": {"x": [], "y": [], "text": []}} for sysdef in SYSTEMS}
    stop_x, stop_y = [], []
    for sysdef in SYSTEMS:
        sname = sysdef["name"]
        for t in result["trades_by_system"].get(sname, []):
            r_multiple = t["pnl"] / t["total_risk_amount"] if t["total_risk_amount"] > 0 else float("nan")
            text = (f"{name} | System {sname} | Exit: {t['exit_reason']}<br>"
                    f"Date: {t['exit_date'].date()}<br>Price: {t['exit_price']:.2f}<br>"
                    f"Units: {t['n_units']} | Holding: {(t['exit_date']-min(t['entry_dates'])).days}d<br>"
                    f"PnL: ${t['pnl']:,.0f} | R-multiple: {r_multiple:.2f}R")
            bucket = exit_data[sname]["win"] if t["pnl"] > 0 else exit_data[sname]["loss"]
            bucket["x"].append(t["exit_date"]); bucket["y"].append(t["exit_price"]); bucket["text"].append(text)
            for seg_start, seg_end, level in reconstruct_stop_history(t):
                stop_x.extend([seg_start, seg_end, None])
                stop_y.extend([level, level, None])

    for sysdef in SYSTEMS:
        sname = sysdef["name"]
        win_data = exit_data[sname]["win"]
        loss_data = exit_data[sname]["loss"]
        if win_data["x"]:
            fig.add_trace(go.Scattergl(
                x=win_data["x"], y=win_data["y"], mode="markers", name=f"{sname} Exit: Winner",
                marker=dict(symbol="circle", color=exit_win_color.get(sname, "darkgreen"), size=9,
                            line=dict(width=1, color="black")),
                text=win_data["text"], hovertemplate="%{text}<extra></extra>",
                legendgroup=f"exit_win_{sname}",
            ), row=1, col=1)
        if loss_data["x"]:
            fig.add_trace(go.Scattergl(
                x=loss_data["x"], y=loss_data["y"], mode="markers", name=f"{sname} Exit: Loser",
                marker=dict(symbol="triangle-down", color=exit_loss_color.get(sname, "darkred"), size=9,
                            line=dict(width=1, color="black")),
                text=loss_data["text"], hovertemplate="%{text}<extra></extra>",
                legendgroup=f"exit_loss_{sname}",
            ), row=1, col=1)

    if stop_x:
        fig.add_trace(go.Scattergl(
            x=stop_x, y=stop_y, mode="lines", name="Stop-loss (all episodes)",
            line=dict(color="darkred", width=1, dash="dot"), opacity=0.6, connectgaps=False,
        ), row=1, col=1)

    # --- ADX panel, one line per system, each independently toggleable (ADX is kept only for
    # the optional regime_scale_out take-profit mechanic -- no entry-side filter uses it) ---
    adx_colors = {"S1": "blue", "S2": "orange"}
    for sysdef in SYSTEMS:
        sname = sysdef["name"]
        col = f"adx_{sname}"
        if col in df.columns:
            fig.add_trace(go.Scattergl(
                x=df.index, y=df[col], mode="lines", name=f"{sname} ADX({sysdef['adx_period']})",
                line=dict(color=adx_colors.get(sname, "blue"), width=1.2),
                legendgroup=f"adx_{sname}",
            ), row=2, col=1)
    fig.add_hline(y=25, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(title="ADX", row=2, col=1)

    # --- ATR / N panel, one line per system, each independently toggleable ---
    n_colors = {"S1": "blue", "S2": "orange"}
    for sysdef in SYSTEMS:
        sname = sysdef["name"]
        col = f"n_{sname}"
        if col in df.columns:
            fig.add_trace(go.Scattergl(
                x=df.index, y=df[col], mode="lines", name=f"{sname} N/ATR({sysdef['n_atr_period']})",
                line=dict(color=n_colors.get(sname, "blue"), width=1.2),
                legendgroup=f"n_{sname}",
            ), row=3, col=1)
    fig.update_yaxes(title="ATR (N)", row=3, col=1)

    fig.add_trace(go.Scatter(x=result["equity_curve"].index, y=result["equity_curve"].values,
                              mode="lines", name="Strategy Equity", line=dict(color="black", width=1.3)),
                  row=4, col=1)
    if bh_equity is not None:
        fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values, mode="lines", name="Buy & Hold Equity",
                                  line=dict(color="gray", width=1.3, dash="dash")), row=4, col=1)

    fig.update_yaxes(title="Price", row=1, col=1)
    fig.update_yaxes(title="Equity ($)", type="log", row=4, col=1)

    # Open already zoomed to an episode-scale window (~4 months) instead of the full multi-year
    # history -- individual entries/exits/stop-lines/indicators are readable from the moment it loads.
    initial_window_days = 120
    initial_start = df.index[0]
    initial_end = min(df.index[-1], initial_start + pd.Timedelta(days=initial_window_days))

    # NOTE: deliberately NOT using a range slider (conflicts with dragmode="pan"). shared_xaxes=True
    # means zooming/panning the price panel automatically syncs RSI/ADX/N/equity to the same window.
    fig.update_xaxes(rangeslider_visible=False, fixedrange=False, row=1, col=1)
    fig.update_xaxes(range=[initial_start, initial_end], row=4, col=1)   # sets the SHARED initial range

    # Sensible initial Y-range on the price panel matching the initial zoomed window, so it isn't
    # flat the moment the page loads -- NOT locked (fixedrange=False): fully, manually draggable.
    initial_window_mask = (df.index >= initial_start) & (df.index <= initial_end)
    if initial_window_mask.any():
        y_low = df.loc[initial_window_mask, "Low"].min()
        y_high = df.loc[initial_window_mask, "High"].max()
        y_pad = (y_high - y_low) * 0.08 if y_high > y_low else y_high * 0.05
        fig.update_yaxes(range=[y_low - y_pad, y_high + y_pad], fixedrange=False, row=1, col=1)

    fig.update_layout(
        height=1500, hovermode="closest",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig.write_html(filepath, config={"scrollZoom": True}, div_id="turtle_episode_chart")
    print(f"Saved interactive episode chart to: {filepath}  (open in a browser -- "
          f"opens zoomed in; drag to pan freely, scroll-wheel/trackpad to zoom, "
          f"double-click to reset. Click any legend entry -- EMA/RSI/ADX/N/entries/exits/stops -- "
          f"to toggle it on or off)")
    return fig


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("DUAL-SYSTEM TURTLE BREAKOUT STRATEGY -- BACKTEST")
    print("=" * 78)
    print(f"Assets: {', '.join(f'{n} ({t})' for n, t in TICKERS.items())}")
    print(f"Portfolio-level (shared, both systems combined): exposure cap = {MAX_TOTAL_EXPOSURE_PCT}% of equity")
    for s in SYSTEMS:
        print(f"\n--- System {s['name']} ---")
        if s.get("use_ma_regime"):
            print(f"  Entry: close above the {s['ma_regime_period']}-day average (+{s['ma_regime_band_pct']}%) "
                  f"-> exit on a close below it (-{s['ma_regime_band_pct']}%)")
        elif monthly_high_settings(s["name"], s):
            mn, mx, sm, mt = monthly_high_settings(s["name"], s)
            print(f"  Entry: the first breakout of each monthly high that is {mn}..{mx} bars old"
                  + (f" and the close is above its {sm}-day {mt.upper()}" if sm else "") + f" (buffer={s['entry_buffer_n_mult']}N), and a unit is added on every such breakout"
                  + (f" -> exit on a close below the {ema_exit_period(s['name'], s)}-day EMA" if ema_exit_period(s["name"], s)
                     else f" -> exit on close below {s['exit_lookback']}-day low"))
        elif ema_entry_settings(s["name"], s) or ema_exit_period(s["name"], s):
            es, xl = ema_entry_settings(s["name"], s), ema_exit_period(s["name"], s)
            entry_txt = (f"close breaks above the {es[0]}-day EMA ({'only on the crossing day' if es[1] == 'cross' else 'whenever flat and above it'})"
                         if es else f"{s['breakout_lookback']}-day breakout (buffer={s['entry_buffer_n_mult']}N)")
            exit_txt = f"exit on a close below the {xl}-day EMA" if xl else f"exit on close below {s['exit_lookback']}-day low"
            print(f"  Entry: {entry_txt} -> {exit_txt}")
        else:
            print(f"  Entry: {s['breakout_lookback']}-day breakout (buffer={s['entry_buffer_n_mult']}N) "
                  f"-> exit on close below {s['exit_lookback']}-day low")
        print(f"  N (volatility): {s['n_atr_period']}-day ATR")
        print(f"  Sizing: (equity x {s['risk_per_trade_pct']}%) / N per unit"
              + (f", graduated schedule {s['risk_per_trade_pct_schedule']}"
                 if s['risk_per_trade_pct_schedule'] else ""))
        ratchet_desc = "" if not s['use_fixed_stop'] else (
            f" -- initial entry - {s['stop_n_mult']}N"
            + (f", {'tightens to (average unit' if s['use_average_entry_stop'] else 'moves to (latest add'} - "
               f"{s['stop_n_mult']}N) on each add" if s['ratchet_stop_on_pyramid']
               else ", stays fixed at the first entry's level regardless of pyramid adds"))
        print(f"  Fixed stop (SL): {s['use_fixed_stop']}{ratchet_desc}")
        print(f"  Trailing stop (N-day low): {s['use_n_day_low_trailing_stop']}"
              + (f" ({s['trailing_stop_lookback_days']}d)" if s['use_n_day_low_trailing_stop'] else ""))
        print(f"  Chandelier exit: {s['use_chandelier_exit']}"
              + (f" ({s['chandelier_atr_mult']}x ATR from highest high)" if s['use_chandelier_exit'] else ""))
        print(f"  Breakeven stop: {s['use_breakeven_stop']}"
              + (f" (triggers at {s['breakeven_trigger_r']}R)" if s['use_breakeven_stop'] else ""))
        print(f"  Time stop: {s['use_time_stop']}"
              + (f" ({s['time_stop_days']}d, min {s['time_stop_min_r']}R)" if s['use_time_stop'] else ""))
        print(f"  Scale-out (TP): {s['use_scale_out']}"
              + (f" {s['scale_out_levels']}" if s['use_scale_out'] else ""))
        print(f"  Regime-triggered scale-out: {s['use_regime_scale_out']}"
              + (f" (ADX>{s['regime_scale_out_adx_threshold']}, takes {s['regime_scale_out_fraction']*100:.0f}%)"
                 if s['use_regime_scale_out'] else ""))
        print(f"  Entry filter -- EMA trend: {s['use_ema_filter']}"
              + (f" (close > EMA({s['ema_period']}))" if s['use_ema_filter'] else ""))
        print(f"  Entry filter -- Choppiness Index: {s['use_choppiness_filter']}"
              + (f" (CI({s['choppiness_period']}) <= {s['choppiness_max']}, i.e. must be trending)"
                 if s['use_choppiness_filter'] else ""))
        print(f"  Entry filter -- require all-time high: {s['require_all_time_high']}")
        if s['pyramid_method'] == 'breakout':
            print(f"  Pyramiding: +1 unit on a fresh {s['pyramid_breakout_lookback']}-day high, "
                  f"AND at least {s['pyramid_breakout_lookback']} days since the last add, "
                  f"up to {s['max_units_per_trend']} units per trend"
                  + (f" -- but NOT if ADX > {s['pyramid_max_adx']} at that moment"
                     if s['use_pyramid_adx_filter'] else ""))
        elif s['pyramid_method'] == 'monthly_high' or s.get('use_monthly_high_entry'):
            print(f"  Pyramiding: +1 unit every time a monthly high ({s['mh_min_age']}..{s['mh_max_age']} bars old) breaks out for the first time"
                  + (f" with the close above its {s['mh_sma_period']}-day {str(s.get('mh_ma_type', 'ema')).upper()}" if s['mh_sma_period'] else "")
                  + f", up to {s['max_units_per_trend']} units per trend"
                  + (f" -- but NOT if ADX > {s['pyramid_max_adx']} at that moment" if s['use_pyramid_adx_filter'] else ""))
        elif s['pyramid_method'] == 'swing_high':
            print(f"  Pyramiding: +1 unit on a close above a CONFIRMED swing high "
                  f"(pivot needs {s['swing_lookback']} bars on both sides), a genuinely newer "
                  f"level than the last add used, up to {s['max_units_per_trend']} units per trend"
                  + (f" -- but NOT if ADX > {s['pyramid_max_adx']} at that moment"
                     if s['use_pyramid_adx_filter'] else ""))
        else:
            print(f"  Pyramiding: +1 unit every {s['pyramid_spacing_n_mult']}N favorable move, "
                  f"up to {s['max_units_per_trend']} units per trend"
                  + (f" -- but NOT if ADX > {s['pyramid_max_adx']} at that moment"
                     if s['use_pyramid_adx_filter'] else ""))
        print(f"  Cooldown after a loss: {s['use_cooldown']}"
              + (f" ({s['cooldown_days']}d)" if s['use_cooldown'] else ""))
    print(f"\nStarting capital: ${STARTING_CAPITAL_PER_ASSET:,.0f} per asset "
          f"(${STARTING_CAPITAL_PER_ASSET * len(TICKERS):,.0f} total)")

    print("\nDownloading data...")
    raw = download_data(TICKERS, START_DATE, END_DATE)

    ath_references = {}
    if any(s.get("require_all_time_high") for s in SYSTEMS):
        print("\nFetching full available history for a genuine all-time-high reference "
              "(separate from the backtest window above)...")
        ath_references = fetch_all_time_highs({name: TICKERS[name] for name in raw})

    print("\nComputing indicators...")
    features = {name: build_features(df, full_history_ath=ath_references.get(name))
                for name, df in raw.items()}

    print("Running simulation...")
    results = {name: simulate_asset(df) for name, df in features.items()}
    bh_equity = {name: compute_buy_and_hold(df) for name, df in features.items()}

    print("\n" + "=" * 78)
    print("TRADE LOGS (all episodes, both systems combined, sorted by entry date)")
    print("=" * 78)
    for name, result in results.items():
        print_trade_log(name, result["trades"])

    print("\n" + "=" * 78)
    print("PYRAMID DETAIL")
    print("=" * 78)
    for name, result in results.items():
        print_pyramid_detail(name, result["trades"])

    print("\n" + "=" * 78)
    print("EPISODE STATISTICS -- win/loss, R-multiple, and entry circumstances for EVERY trade")
    print("(combined across all assets; split by System 1 = faster/shorter-term vs System 2 = slower/longer-term)")
    print("=" * 78)
    episode_df = build_episode_dataframe(results)
    print_episode_circumstance_stats(episode_df)

    print("\n" + "=" * 78)
    print("MOST COMMON ENTRY CONDITIONS -- RSI, ADX, and ATR/volatility regime, for winners")
    print("and losers, per system (RSI/ADX bucketed 0-100; ATR bucketed by its own percentile,")
    print("since raw N isn't comparable across assets with very different price scales)")
    print("=" * 78)
    print_most_common_entry_conditions(episode_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    episode_csv_path = os.path.join(OUTPUT_DIR, "episode_statistics.csv")
    episode_df.to_csv(episode_csv_path, index=False)
    print(f"\nFull per-episode data ({len(episode_df)} episodes, every circumstance captured) "
          f"written to: {episode_csv_path}")

    print("\n" + "=" * 78)
    print("SAMPLE TRADE CHARTS -- 5 representative (near-average-R) winners and losers")
    print("=" * 78)
    all_trades_enriched = collect_all_trades(results)
    selected = select_representative_trades(all_trades_enriched, n=5, seed=42)
    print(f"Winners selected (near the winners' own average R): "
          f"{[round(t['r_multiple'], 2) for t in selected['winners']]}")
    print(f"Losers selected (near the losers' own average R): "
          f"{[round(t['r_multiple'], 2) for t in selected['losers']]}")
    for i, item in enumerate(selected["winners"], start=1):
        plot_trade_closeup(features[item["asset"]], item["asset"], item["trade"],
                            filename=f"trade_winner_{i}_{item['asset']}.png")
    for i, item in enumerate(selected["losers"], start=1):
        plot_trade_closeup(features[item["asset"]], item["asset"], item["trade"],
                            filename=f"trade_loser_{i}_{item['asset']}.png")

    print("\n" + "=" * 78)
    print("PER-SYSTEM EPISODE STATISTICS")
    print("=" * 78)
    for name, result in results.items():
        print(f"\n--- {name} ---")
        ep_stats_by_system = {
            sysdef["name"]: episode_stats(result["trades_by_system"][sysdef["name"]],
                                            result["open_units_by_system"][sysdef["name"]])
            for sysdef in SYSTEMS
        }
        print_episode_stats_table(ep_stats_by_system)

    print("\n" + "=" * 78)
    print("STRATEGY vs. BUY & HOLD -- PER ASSET")
    print("=" * 78)
    strat_stats, bh_stats = {}, {}
    for name in results:
        strat_stats[name] = performance_stats(results[name]["equity_curve"], STARTING_CAPITAL_PER_ASSET)
        bh_stats[name] = performance_stats(bh_equity[name], STARTING_CAPITAL_PER_ASSET)
        print(f"\n--- {name} ---")
        print_stats_table({"Strategy": strat_stats[name], "Buy & Hold": bh_stats[name]})

    date_range = pd.date_range(
        min(r["equity_curve"].index.min() for r in results.values()),
        max(r["equity_curve"].index.max() for r in results.values()), freq="D")
    combined_strat_equity = sum(
        results[name]["equity_curve"].reindex(date_range).ffill().fillna(STARTING_CAPITAL_PER_ASSET)
        for name in results)
    combined_bh_equity = sum(
        bh_equity[name].reindex(date_range).ffill().fillna(STARTING_CAPITAL_PER_ASSET)
        for name in results)
    total_capital = STARTING_CAPITAL_PER_ASSET * len(TICKERS)
    combined_strat_stats = performance_stats(combined_strat_equity, total_capital)
    combined_bh_stats = performance_stats(combined_bh_equity, total_capital)
    print("\n" + "=" * 78)
    print("STRATEGY vs. BUY & HOLD -- COMBINED PORTFOLIO")
    print("=" * 78)
    print_stats_table({"Strategy (combined)": combined_strat_stats, "Buy & Hold (combined)": combined_bh_stats})

    print("\n" + "=" * 78)
    print("YEARLY RETURNS")
    print("=" * 78)
    for name in results:
        print_yearly_table(name, results[name]["equity_curve"], STARTING_CAPITAL_PER_ASSET)
    print_yearly_table("Combined Portfolio", combined_strat_equity, total_capital)

    print("\n" + "=" * 78)
    print("TRANSACTION COST SUMMARY")
    print("=" * 78)
    cost_rows = []
    total_slip = total_comm = total_hold = 0.0
    for name, result in results.items():
        cost_rows.append({
            "Asset": name, "Slippage/Spread": f"${result['total_slippage_spread_cost']:,.0f}",
            "Commission": f"${result['total_commission_cost']:,.0f}",
            "Holding/Financing": f"${result['total_holding_cost']:,.0f}",
        })
        total_slip += result["total_slippage_spread_cost"]
        total_comm += result["total_commission_cost"]
        total_hold += result["total_holding_cost"]
    cost_rows.append({"Asset": "TOTAL", "Slippage/Spread": f"${total_slip:,.0f}",
                       "Commission": f"${total_comm:,.0f}", "Holding/Financing": f"${total_hold:,.0f}"})
    print(pd.DataFrame(cost_rows).to_string(index=False))

    print("\nPlotting...")
    for name, df in features.items():
        plot_asset(name, df, results[name], filename=f"turtle_{name}.png", bh_equity=bh_equity[name])
        plot_all_episodes_interactive(name, df, results[name], filename=f"turtle_{name}_interactive.html",
                                        bh_equity=bh_equity[name])

    print(f"\nAll output files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()