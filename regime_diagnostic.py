"""
Regime Diagnostic for the Dual-System Turtle Strategy
==========================================================
Tests specific hypotheses about WHAT KIND of market conditions the
strategy's edge actually depends on, across every asset -- not just "is it
choppy or trending" but several independent, complementary angles on the
same question, so a pattern that shows up in multiple measures is much
more trustworthy than one that only shows up in a single indicator.

Each REGIME MEASURE below is computed fresh per asset (NOT tied to either
system's own period settings -- an honest, system-agnostic read of each
asset's actual character), bucketed into 3 categories, and reported two
ways:
  PART 1: what % of the whole backtest period each asset spent in each
          bucket -- "did this asset actually see more of X than the others?"
  PART 2: the strategy's episode stats (win rate, avg R, PnL), split by
          which bucket was active AT ENTRY, per asset -- "does the
          strategy's edge actually depend on this, consistently, across
          every asset?"

MEASURES INCLUDED:
  - Choppiness Index(14):    is price moving directly to its destination,
                              or zigzagging to get there
  - ADX(14):                 trend STRENGTH (note: a LAGGING indicator --
                              see the regime age measure below, which
                              exists specifically to test that lag)
  - Efficiency Ratio(14):    a differently-constructed trend/noise ratio
                              (net move / sum of absolute daily moves) --
                              a genuinely independent measure from the two
                              above, so agreement across all three is a
                              much stronger signal than any one alone
  - Regime Age (Choppiness): how many days the CURRENT Choppiness regime
                              has been running -- tests whether early
                              entries into a fresh regime outperform late
                              entries into an already-extended one (this
                              is the direct follow-up to the finding that
                              high-ADX "confirmed trend" entries were the
                              WORST bucket on every asset -- ADX lags, so
                              this asks the same question a cleaner way)
  - Extension from N-day high: how far price is already above (or still
                              below) its recent 20-day high at entry --
                              tests whether "fresh" breakouts outperform
                              "already extended" ones
  - Momentum / ROC(20):      20-day rate of change -- raw momentum
                              strength at entry, independent of any of
                              the above

Requires turtle_dual_system.py in the same folder.
"""

import os
import numpy as np
import pandas as pd

import turtle_dual_system as td

OUTPUT_DIR = td.OUTPUT_DIR


# ----------------------------------------------------------------------------
# REGIME MEASURES -- each is (name, compute_fn(df) -> Series, bucket_fn(Series) -> Series,
# ordered bucket labels). Add a new measure by adding one more entry to this list --
# both Part 1 and Part 2 below loop over it generically, nothing else needs to change.
# ----------------------------------------------------------------------------
def compute_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    tr_sum = tr.rolling(period).sum()
    range_high = high.rolling(period).max()
    range_low = low.rolling(period).min()
    span = (range_high - range_low).replace(0, np.nan)
    return 100 * np.log10(tr_sum / span) / np.log10(period)


def compute_efficiency_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df["Close"]
    net_move = (close - close.shift(period)).abs()
    total_move = close.diff().abs().rolling(period).sum()
    return 100 * net_move / total_move.replace(0, np.nan)


def compute_regime_age(df: pd.DataFrame, period: int = 14) -> pd.Series:
    ci = compute_choppiness_index(df, period)
    bucket = bucket_choppiness(ci)
    changed = bucket != bucket.shift(1)
    regime_id = changed.cumsum()
    age = bucket.groupby(regime_id).cumcount() + 1
    age = age.where(bucket.notna())
    return age.astype(float)


def compute_extension_from_high(df: pd.DataFrame, period: int = 20) -> pd.Series:
    recent_high = df["High"].rolling(period).max().shift(1)
    return (df["Close"] - recent_high) / recent_high * 100


def compute_roc(df: pd.DataFrame, period: int = 20) -> pd.Series:
    close = df["Close"]
    return (close / close.shift(period) - 1) * 100


def bucket_choppiness(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, 38.2, 61.8, np.inf], labels=["Trending", "Neutral", "Choppy"])


def bucket_adx(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, 20.0, 25.0, np.inf], labels=["Choppy", "Neutral", "Trending"])


def bucket_efficiency_ratio(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, 30, 60, np.inf], labels=["Choppy", "Neutral", "Trending"])


def bucket_regime_age(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, 5, 15, np.inf], labels=["Early (<=5d)", "Established (6-15d)", "Late (>15d)"])


def bucket_extension(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, 0, 2, np.inf],
                  labels=["At/Below Recent High", "Slightly Extended (0-2%)", "Very Extended (>2%)"])


def bucket_roc(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, 0, 10, np.inf], labels=["Negative/Flat", "Moderate (0-10%)", "Strong (>10%)"])


REGIME_MEASURES = [
    ("Choppiness Index(14)", compute_choppiness_index, bucket_choppiness,
     ["Trending", "Neutral", "Choppy"]),
    ("ADX(14)", lambda df: td.compute_adx(df, 14), bucket_adx,
     ["Choppy", "Neutral", "Trending"]),
    ("Efficiency Ratio(14)", compute_efficiency_ratio, bucket_efficiency_ratio,
     ["Choppy", "Neutral", "Trending"]),
    ("Regime Age (Choppiness-based)", compute_regime_age, bucket_regime_age,
     ["Early (<=5d)", "Established (6-15d)", "Late (>15d)"]),
    ("Extension from 20-day High", compute_extension_from_high, bucket_extension,
     ["At/Below Recent High", "Slightly Extended (0-2%)", "Very Extended (>2%)"]),
    ("Momentum / ROC(20)", compute_roc, bucket_roc,
     ["Negative/Flat", "Moderate (0-10%)", "Strong (>10%)"]),
]


# ----------------------------------------------------------------------------
# MAIN DIAGNOSTIC
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("REGIME DIAGNOSTIC -- strategy performance by market regime, per asset")
    print(f"({len(REGIME_MEASURES)} independent measures: " + ", ".join(m[0] for m in REGIME_MEASURES) + ")")
    print("=" * 78)

    print("\nDownloading data...")
    raw = td.download_data(td.TICKERS, td.START_DATE, td.END_DATE)

    print("\nComputing indicators and running the strategy (current SYSTEMS config)...")
    features = {}
    results = {}
    regime_data = {}
    for name, df in raw.items():
        feat = td.build_features(df)
        features[name] = feat
        results[name] = td.simulate_asset(feat)

        rd_cols = {"daily_return": df["Close"].pct_change()}
        for measure_name, compute_fn, bucket_fn, _ in REGIME_MEASURES:
            raw_series = compute_fn(df)
            rd_cols[f"{measure_name}__raw"] = raw_series
            rd_cols[f"{measure_name}__bucket"] = bucket_fn(raw_series)
        regime_data[name] = pd.DataFrame(rd_cols, index=df.index)

    # ------------------------------------------------------------------------
    # PART 1: what % of time was each asset actually IN each regime bucket?
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PART 1: % OF BACKTEST PERIOD SPENT IN EACH REGIME BUCKET, PER ASSET")
    print("(did one asset actually see more of a given condition than the others?)")
    print("=" * 78)
    for measure_name, _, _, labels in REGIME_MEASURES:
        print(f"\n--- By {measure_name} ---")
        col = f"{measure_name}__bucket"
        rows = []
        for name, rd in regime_data.items():
            counts = rd[col].value_counts(normalize=True) * 100
            row = {"Asset": name}
            for lbl in labels:
                row[f"{lbl} %"] = round(counts.get(lbl, 0.0), 1)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------------
    # PART 2: strategy episode stats, split by regime AT ENTRY, per asset
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PART 2: STRATEGY PERFORMANCE BY REGIME AT ENTRY, PER ASSET")
    print("(does the strategy's edge depend on this, CONSISTENTLY, across every asset?)")
    print("=" * 78)

    episode_df = td.build_episode_dataframe(results)
    if episode_df.empty:
        print("No closed episodes -- nothing to break down.")
    else:
        for measure_name, _, _, labels in REGIME_MEASURES:
            col = f"{measure_name}__bucket"
            episode_df[col] = None
            for idx, row in episode_df.iterrows():
                rd = regime_data[row["asset"]]
                entry_ts = pd.Timestamp(row["entry_date"])
                if entry_ts in rd.index:
                    episode_df.loc[idx, col] = rd.loc[entry_ts, col]

        for measure_name, _, _, labels in REGIME_MEASURES:
            col = f"{measure_name}__bucket"
            print(f"\n--- By {measure_name} at entry ---")
            for asset in sorted(episode_df["asset"].unique()):
                sub = episode_df[episode_df["asset"] == asset]
                print(f"\n  {asset}:")
                rows = []
                for regime in labels:
                    r = sub[sub[col] == regime]
                    if len(r) == 0:
                        rows.append({"Regime": regime, "Episodes": 0, "Win Rate": "n/a",
                                     "Avg R": "n/a", "Total PnL": "n/a"})
                        continue
                    win_rate = (r["win"].sum() / len(r)) * 100
                    rows.append({
                        "Regime": regime, "Episodes": len(r),
                        "Win Rate": f"{win_rate:.1f}%",
                        "Avg R": f"{r['r_multiple'].mean():.2f}R",
                        "Total PnL": f"${r['pnl'].sum():,.0f}",
                    })
                print("    " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n    "))

    # ------------------------------------------------------------------------
    # PART 3: buy & hold's average daily return conditional on regime -- context
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PART 3: BUY & HOLD -- AVERAGE DAILY RETURN CONDITIONAL ON REGIME, PER ASSET")
    print("(context: how much of the asset's raw price movement happened in each bucket)")
    print("=" * 78)
    for measure_name, _, _, labels in REGIME_MEASURES:
        print(f"\n--- By {measure_name} ---")
        col = f"{measure_name}__bucket"
        rows = []
        for name, rd in regime_data.items():
            row = {"Asset": name}
            for lbl in labels:
                sub = rd[rd[col] == lbl]
                avg_ret = sub["daily_return"].mean() * 100 if len(sub) else np.nan
                row[f"{lbl} avg daily %"] = f"{avg_ret:.3f}%" if pd.notna(avg_ret) else "n/a"
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------------
    # save full per-episode regime data for further digging
    # ------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not episode_df.empty:
        csv_path = os.path.join(OUTPUT_DIR, "regime_diagnostic_episodes.csv")
        episode_df.to_csv(csv_path, index=False)
        print(f"\nFull per-episode regime data ({len(episode_df)} episodes, "
              f"{len(REGIME_MEASURES)} regime measures each) saved to: {csv_path}")

    print(f"\nDone. Output files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()