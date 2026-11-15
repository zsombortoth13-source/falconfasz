# Monthly-high lab

Backtest, optimizer and walk-forward of the monthly-high breakout strategy (entry above its EMA on the first close over a monthly high that is min..max days old, pyramiding on every further signal, an ATR stop that moves up with every add, an ATR take-profit per unit), with Monte Carlo and a chart of every position.

Run locally:

    pip install -r requirements.txt
    streamlit run mh_walkforward_ui.py

Main file for Streamlit Community Cloud: `mh_walkforward_ui.py`.

`cache_prices/` holds saved daily prices so the app starts without downloading; the app refreshes them from Yahoo Finance when they are older than 12 hours and falls back to the saved copy if Yahoo refuses the download.

`data_tradingview/` holds daily price files fetched from TradingView's data feed (one CSV per market, daily bars, the S&P 500 from 1950 and the Dow from 1896). Choose **Price source > TradingView files** in the sidebar to use them instead of Yahoo Finance (which Streamlit Community Cloud sometimes refuses); the **Download the latest TradingView data** button there replaces them with fresh ones (`tv_prices.py`, `tv_fetch.py`). Delayed futures data, bars only, an unofficial feed.


**Defaults = the baseline test.** `baseline_defaults.py` holds the baseline settings (9 markets, TradingView files from 1951, $100,000, Interactive Brokers futures costs with micro contracts, margin calls and the equal margin slice on, 1% risk per unit, 300% notional cap, 100% open-risk cap, walk-forward 10 years train / 5 years test, 40 Optuna trials ranked by CAGR, seed 42). A walk-forward started without changing anything runs exactly that test (CAGR 22.4%, max drawdown -64.9%, Sharpe 0.91, 14 windows). Change a number in that one file to change the defaults.

**To update the app:** upload everything in this folder to the GitHub repository (replace the files that exist, add `baseline_defaults.py`); Streamlit Community Cloud restarts the app by itself. The `.py` files (including `brokers_builtin.py`) and `requirements.txt` are all the app needs: the Interactive Brokers cost profile is built into `brokers_builtin.py`, so the `brokers/` folder is optional (a file there wins over the built-in copy). `data_tradingview/` and `cache_prices/` only save time. The sidebar ends with a **Setup check** line (a code fingerprint, where the cost profile comes from, how many TradingView files are here): it must read the same on the cloud as on your own computer, otherwise some files were not uploaded. If `data_tradingview/` is not uploaded the app does not fall back to Yahoo: it shows a warning in the sidebar and downloads the missing files from TradingView at the start of the first run (1-2 minutes), and every result says where its prices came from ("prices: TradingView files, 1951-01-02 to ..."). If that download fails the run stops with an error.

