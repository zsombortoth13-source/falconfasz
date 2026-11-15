"""THE BASELINE TEST: the default settings of every UI (the lean Monthly-high lab and the Strategy Lab), in one place.

A walk-forward started without touching any setting runs exactly this test. Its result (14 Nov 2026): CAGR 22.4%, maximum drawdown -64.9%, Sharpe 0.91, final equity $62,087,957,828, 14 windows,
13 of them positive. The documents in docs/ describe it and prove it. Change a number here and both UIs follow (the walk-forward result will of course change too).

Markets            the nine below (the Strategy Lab has fifteen; these are the ones the baseline uses)
Prices             TradingView daily files (data_tradingview/), from 1951-01-01
Account            $100,000, Interactive Brokers futures costs, micro contracts (whole contracts, at least one), margin calls and the equal margin slice on
Risk               1% of the account per unit, at most 300% of equity in one market, at most 100% of equity of open risk in the account
Walk-forward       10-year rolling train window, 5-year test window, 40 Optuna trials per window ranked by CAGR, 3-year warm start, middle-of-top-N off, seed 42
Strategy           the eight monthly-high numbers below, all searched over the candidate lists below"""

MARKETS = ["SP500", "Gold", "Silver", "Nasdaq", "Bitcoin", "Ger40", "Dow", "Nikkei225", "Russell2000"]
PRICE_SOURCE = "tradingview"
HISTORY_START = "1951-01-01"

ACCOUNT = 100_000.0
CONTRACT_SIZE = "micro"
MARGIN_CALLS = True
MARGIN_SLICE = True
TAKE_MIN_ONE = True
EXPOSURE_PCT = 300
MAX_OPEN_RISK_PCT = 100.0
RISK_PER_UNIT_PCT = 1.0

TRAIN_YEARS = 10
TEST_YEARS = 5
TRIALS = 40
RANK_METRIC = "cagr"
SEARCH_METHOD = "optuna"
ANCHORED = False
WARM_START = True
PICK_MIDDLE = False
TOP_N = 10
SEED = 42

# the eight numbers of the monthly-high strategy: the value used when a number is not searched, and the candidate list the search draws from
MH_VALUES = {"mh_sma_period": 200, "mh_min_age": 20, "mh_max_age": 365, "entry_buffer_n_mult": 0.0, "n_atr_period": 185, "stop_n_mult": 6.0, "tp_atr_period": 185, "tp_n_mult": 30.0}
MH_CANDIDATES = {"mh_sma_period": "100, 150, 200, 250, 300", "mh_min_age": "10, 15, 20, 30, 40, 60", "mh_max_age": "120, 180, 250, 365, 500", "entry_buffer_n_mult": "0, 0.25, 0.5, 1, 1.5, 2",
                 "n_atr_period": "14, 20, 40, 60, 100, 185", "stop_n_mult": "1.5, 2, 3, 4, 6, 8", "tp_atr_period": "14, 20, 40, 100, 185", "tp_n_mult": "3, 6, 10, 20, 30"}


def ordered(markets, all_markets=None) -> list:
    """The markets in the ONE order every run uses: the baseline's markets first, in the baseline's order, then any others in the order of `all_markets`.
    The order is not cosmetic: when two markets signal on the same day and a cap (account open risk, margin slice, margin) cannot serve both, the one listed first is served first,
    so the same markets in another order can give a slightly different result. A fixed order makes the result depend on WHICH markets are chosen, not on the order they were clicked."""
    markets = list(markets)
    first = [m for m in MARKETS if m in markets]
    rest_order = list(all_markets) if all_markets is not None else markets
    rest = [m for m in rest_order if m in markets and m not in MARKETS]
    rest += [m for m in markets if m not in first and m not in rest]
    return first + rest
